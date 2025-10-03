"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
抽象基类：分布式训练策略的公共封装，统筹参数初始化、优化器管理、训练/评估循环等通用逻辑。
"""

from abc import ABC, abstractmethod  # 定义抽象基类与抽象方法接口
from pathlib import Path  # 表示运行目录、检查点路径等
from typing import Callable, Optional  # 回调函数与可选类型注解

import torch  # 张量运算框架
import torch.distributed as dist  # 分布式通信原语（barrier、world_size 等）
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset  # 数据加载与采样工具
from tqdm import tqdm  # CLI 进度条
from transformers.modeling_outputs import CausalLMOutputWithPast  # LLM 前向输出结构

from prismatic.models.vlms import PrismaticVLM  # 视觉语言模型基类
from prismatic.overwatch import initialize_overwatch  # 自定义日志/监控器
from prismatic.training.metrics import Metrics, VLAMetrics  # 训练指标管理器
from prismatic.util import check_bloat16_supported  # BF16 支持性检查
from prismatic.util.batching_utils import SplitModalitySampler  # 文本/多模态混合采样器
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling  # collator
from prismatic.vla.action_tokenizer import ActionTokenizer  # 动作 tokenizer，用于指标计算

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage  # 保存模型、设备 ID 与训练阶段信息

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys  # 记录模块名称
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls  # Transformer 层类型（用于包装策略）

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps  # 训练轮数与最大步数（步数优先）
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size  # 全局/单卡 batch size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm  # 优化器超参
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio  # 学习率调度策略与预热比例

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing  # 是否开启梯度检查点以节省显存
        self.enable_mixed_precision_training = enable_mixed_precision_training  # 是否启用混合精度训练
        self.reduce_in_full_precision = reduce_in_full_precision  # 是否在 FP32 中执行梯度规约
        self.mixed_precision_dtype = mixed_precision_dtype  # 混合精度使用的数据类型（默认 BF16）

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn  # DataLoader worker 初始化函数

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None  # 优化器与学习率调度器占位

        # Lightweight Validation
        # 基本合法性校验：全局 batch size 必须能被单卡 batch size * 世界尺寸整除
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        # 根据世界规模推导梯度累积步数（全局 batch / 单卡 batch / GPU 数）
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,  # 运行目录，用于保存检查点
        global_step: int,  # 当前全局训练步数
        epoch: int,  # 当前训练轮数
        train_loss: Optional[float] = None,  # 可选：记录保存时的训练损失
        only_trainable: bool = True,  # 是否只保存可训练参数
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...  # 策略初始化：参数分片、优化器构建等

    @abstractmethod
    def clip_grad_norm(self) -> None: ...  # 梯度裁剪逻辑，由子类实现（区分 DDP/FSDP 等）

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`
        通用训练循环：根据 stage 与采样策略构建 DataLoader，执行梯度累积、优化器更新和指标记录。
        """
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            # 微调阶段支持按模态长度拆分 batch，提升语言/视觉样本的均衡度
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(  # 默认使用 PyTorch 分布式采样器
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(  # 构建 DataLoader：使用上面初始化的采样器与 collator
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps  # 计算每轮有效梯度步数（每完成 grad_accumulation_steps 次迭代记 1 步）
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()  # 获取初始状态字符串，用于进度条描述
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):  # 外层循环遍历 epoch
                self.vlm.train()  # 切换模型到训练模式
                sampler.set_epoch(epoch)  # 通知分布式采样器当前 epoch，确保多 GPU shuffle 同步

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()  # 清空梯度缓存

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],  # 图像/视觉输入
                            labels=batch["labels"],  # 自监督标签
                            multimodal_indices=batch["multimodal_indices"],  # 标记哪些样本包含视觉数据
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)  # 先记录原始 loss（未归一化）

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps  # 按梯度累积步数归一化
                    normalized_loss.backward()  # 反向传播，梯度会累积在模型参数上

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)  # 更新耗时信息，用于估计训练速度

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()  # 策略自定义的梯度裁剪（区分 DDP/FSDP）

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()  # 更新参数
                        self.lr_scheduler.step()  # 更新学习率
                        self.optimizer.zero_grad()  # 清空梯度，准备下一轮累积

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])  # 记录步数与学习率
                        status = metrics.push()  # 推送到外部追踪器

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:  # 达到最大步数则保存检查点并退出
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()  # 推进进度条
                        progress.set_description(status)  # 更新显示文本

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:  # 若未设置 max_steps，每个 epoch 结束也保存一次
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())  # 保存当前 epoch 检查点
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"  # RLDS 预期为可迭代数据集
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(  # RLDS 的 IterableDataset 自带循环逻辑，num_workers 设为 0
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()  # 切换到训练模式

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()  # 清空上轮残留梯度

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss  # 自监督/自回归损失由模型内部返回

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)  # 记录 loss
                loss.backward()  # 反向传播

                # === Compute Action Token Accuracy & L1 Loss ===
                # 计算离散动作 token 的准确率，以及连续动作的 L1 误差

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)  # 预测的动作 token id
                action_gt = batch["labels"][:, 1:].to(action_preds.device)  # 去掉 BOS，与预测对齐
                mask = action_gt > action_tokenizer.action_token_begin_idx  # 仅计算动作区间的准确率

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask  # 逐位置比对
                action_accuracy = correct_preds.sum().float() / mask.sum().float()  # 得到动作 token 准确率

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)  # 连续动作 L1 误差

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)  # 更新指标

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():  # 仅在主进程按数据集维度记录细粒度指标
                    datasets = set(batch["dataset_names"])  # 将 batch 内的数据集名称去重
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)  # 依据步数近似计算 epoch

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
