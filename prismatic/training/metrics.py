"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
度量模块：统一管理训练/微调过程中的指标记录与追踪器（JSONL、本地日志、W&B 等）。
"""

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union  # 常用类型别名定义

import jsonlines  # 写入 jsonl 文件的轻量库
import numpy as np  # 用于统计时间等均值
import torch  # 用于管理张量化指标（loss 等）
import wandb  # Weights & Biases 日志追踪 SDK

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...  # 保存运行配置，例如超参

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None: ...  # 写入单条日志

    def finalize(self) -> None: ...  # 结束追踪器，释放资源


# === Individual Tracker Definitions ===
class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        # run_id：运行标识；run_dir：输出目录；hparams：配置字典
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        # 仅在 rank0 进程写入 run_config，避免多进程重复写文件
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        # 追加写入训练过程中每一步的指标记录
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return


class WeightsBiasesTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "prismatic",
        entity: Optional[str] = None,
        group: str = "align",
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Get W&B-Specific Initialization Parameters
        self.project, self.entity, self.group, self.wandb_dir = project, entity, group, self.run_dir  # W&B 初始化参数

        # Call W&B.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        # 仅在 rank0 初始化 W&B run，否则多进程会重复创建
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)  # 留足时间让 W&B 同步上传


# === Core Metrics Container :: Initializes Trackers => Compiles/Pushes Metrics ===


class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        stage: str,
        wandb_project: str = "prismatic",
        wandb_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group=self.stage
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()  # 先写入超参配置
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()  # 记录训练起点、计时
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),  # 梯度累积窗口内的原始 loss
            "loss": deque(maxlen=window_size),  # 平滑后的 loss
            "step_time": deque(maxlen=window_size),  # 记录 step 耗时
            "lr": [],  # 学习率历史
        }

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        # 将指标写入所有激活的 tracking 后端
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        # 拼装进度条状态文本，默认包含当前学习率，若提供 loss 则追加展示
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self, *, global_step: Optional[int] = None, lr: Optional[float] = None, update_step_time: bool = False, **kwargs
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())  # 其他指标也默认转为张量后记录

    @overwatch.rank_zero_only
    def push(self) -> str:
        # 对窗口内的 loss、step_time 等进行平均，准备写入日志
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        step_time, lr = np.mean(list(self.state["step_time"])), self.state["lr"][-1]
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Loss": loss,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
            },
        )
        return status

    def finalize(self) -> str:
        # 依次调用所有 tracker 的 finalize 方法，关闭资源
        for tracker in self.trackers:
            tracker.finalize()


class VLAMetrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        wandb_project: str = "openvla",
        wandb_entity: Optional[str] = "stanford-voltron",
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Initialize Trackers
        self.trackers = []  # 与 Metrics 类似，支持多种 tracker 并行输出
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group="vla-train"
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()  # 写入配置
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step  # 支持从断点恢复步数
        self.epoch = 0 if resume_epoch is None else resume_epoch  # 支持从断点恢复 epoch
        self.start_time, self.step_start_time = time.time(), time.time()  # 用于统计整体耗时与单步耗时
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),  # 存储未平滑 loss（累积窗口）
            "loss": deque(maxlen=window_size),  # 存储平滑后的 loss
            "l1_loss": deque(maxlen=window_size),  # 连续动作 L1 误差
            "action_accuracy": deque(maxlen=window_size),  # 动作 token 准确率
            "step_time": deque(maxlen=window_size),  # 单步耗时样本
            "lr": [],  # 学习率历史记录
        }

        # Created metrics buffers for individual tracked datasets
        self.dataset_trackers = defaultdict(lambda: VLAMetrics([], "", "", {}))  # 针对多数据集场景记录细粒度指标

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        # 将本次指标写入所有注册的追踪器
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        # 生成进度条状态字符串，包含当前 epoch、全局步与学习率
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} - Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    def commit_for_dataset(self, dataset_name: str, **kwargs) -> None:
        # 将指标提交到指定数据集对应的 tracker
        self.dataset_trackers[dataset_name].commit(**kwargs)

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()  # 梯度累积窗口内的原始 loss
        loss = torch.stack(list(self.state["loss"])).mean().item()  # 平滑后的 loss
        l1_loss = torch.stack(list(self.state["l1_loss"])).mean().item()  # 行为 L1 误差
        action_accuracy = torch.stack(list(self.state["action_accuracy"])).mean().item()  # 动作 token 准确率
        step_time = np.mean(list(self.state["step_time"]))  # 平均 step 时间
        lr = self.state["lr"][-1]
        status = self.get_status(loss)

        # Get metrics per dataset
        dataset_metrics = {}
        for ds, tracker in self.dataset_trackers.items():
            dataset_metrics.update(
                {
                    f"{ds}/L1 Loss": torch.stack(list(tracker.state["l1_loss"])).mean().item(),  # 分数据集 L1
                    f"{ds}/Action Token Accuracy": torch.stack(list(tracker.state["action_accuracy"])).mean().item(),
                }
            )

        # Fire to Trackers
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": self.global_step,
                f"{prefix}/Epoch": self.epoch,
                f"{prefix}/Loss": loss,
                f"{prefix}/L1 Loss": l1_loss,
                f"{prefix}/Action Token Accuracy": action_accuracy,
                f"{prefix}/Loss (Raw)": loss_raw,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
                **dataset_metrics,
            },
        )
        return status

    def finalize(self) -> str:
        for tracker in self.trackers:
            tracker.finalize()
