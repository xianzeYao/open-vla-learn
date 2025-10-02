"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""
# Vision-Language-Action 训练脚本说明：依托预训练 VLM，使用 PyTorch FSDP 多机多卡训练，并提供常用运行方式与先决条件。

import json  # JSON 配置读写工具
import os  # 访问环境变量与路径的 OS 工具
import re  # 正则解析检查点命名
from dataclasses import dataclass, field  # 定义配置数据类
from pathlib import Path  # 跨平台路径操作
from typing import Optional, Tuple, Union  # 类型注解提升可读性

import draccus  # 把数据类映射到命令行的配置库
import torch  # PyTorch 张量与自动求导
import torch.distributed as dist  # 分布式通信工具
import yaml  # YAML 解析与序列化

from prismatic.conf import VLAConfig, VLARegistry  # VLA 配置与注册表
from prismatic.models import load, load_vla  # 加载基础模型或增量模型
from prismatic.overwatch import initialize_overwatch  # 初始化统一日志
from prismatic.training import VLAMetrics, get_train_strategy  # 指标与训练策略工厂
from prismatic.util import set_global_seed  # 设置随机种子
from prismatic.vla import get_vla_dataset_and_collator  # 构建 VLA 数据管线
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # 保存数据统计

# 环境默认配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 初始化 Overwatch，统一管理多进程日志与状态
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig 定义位于 prismatic/conf/vla.py，可通过 --vla.type 指定其它预设
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # 目录相关参数
    data_root_dir: Path = Path(                                     # Open-X 数据集所在位置
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # 训练日志与模型输出目录

    # 断点续训参数
    pretrained_checkpoint: Optional[Path] = None                    # 预训练或断点权重路径
    is_resume: bool = True                                          # 指示是否继续之前的训练
    resume_step: Optional[int] = None                               # 恢复时的全局步数
    resume_epoch: Optional[int] = None                              # 恢复时的轮次编号

    # 运行参数
    run_id: Optional[str] = None                                    # 日志使用的运行 ID
    run_id_note: Optional[str] = None                               # 额外的运行标记
    save_interval: int = 2500                                       # 检查点保存间隔（按步数）
    image_aug: bool = False                                         # 是否启用图像增强
    seed: int = 7                                                   # 随机种子

    # Hugging Face 凭证
    hf_token: Union[str, Path] = Path(".hf_token")                  # HF 访问令牌（路径或环境变量名）

    # 日志追踪参数
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # 启动的追踪器类型
    wandb_project: str = "openvla"                                  # W&B 项目名
    wandb_entity: str = "stanford-voltron"                          # W&B workspace 名称

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size
        # 将优化相关超参提升到顶层字段，方便命令行覆盖

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # 校验当前进程组规模是否符合配置中的 expected_world_size
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


# draccus.wrap() 将数据类配置映射成命令行参数
@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")
    # 训练入口：初始化设备、模型、数据集与训练策略

    # 在 torchrun 环境下初始化 overwatch 会同步初始化 torch.distributed
    torch.cuda.set_device(device_id := overwatch.local_rank())
    # local_rank 指示当前进程绑定的 GPU 序号
    torch.cuda.empty_cache()

    # 组装唯一的运行 ID 并创建输出目录
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # 创建运行目录并设置随机种子
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(
        cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # 支持从文件或环境变量读取 Hugging Face 凭证
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # 保存配置文件（同时生成 YAML 与 JSON 两份）
    if overwatch.is_rank_zero():
        # 仅由 rank0 负责写盘，避免多进程同时写文件
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # 加载基础 VLM 或已有的 VLA 检查点，并确保参数以 FP32 载入
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # 断点恢复时解析文件名，校验 step 与 epoch 是否匹配
        if cfg.is_resume:
            assert int(re.search(
                "step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search(
                "epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        vlm = load_vla(cfg.pretrained_checkpoint,
                       hf_token=hf_token, load_for_training=True)

    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    # 校验模型参数均为 FP32，避免混合精度带来的不一致
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # 根据各分支是否冻结来确定训练阶段，支持多种微调策略
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:  # 无分支冻结，执行全量微调
        stage = "vla-full-train"
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:  # 冻结视觉分支，仅更新语言相关层
        stage = "vla-train"
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:  # 冻结大模型主体，仅保留末层可训练
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        # 该模式微调视觉编码器、投影层以及语言模型最后一层
        stage = "vla-sandwich-train"
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # 只训练语言模型的最后一层
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )
    # 通过 stage 控制哪些子模块可训练，实现从全量到“夹心”式等不同解冻策略

    # 显式调用 freeze_backbones，便于在日志中确认各分支冻结情况
    overwatch.info(
        f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    # 统计模型总参数量与可训练参数量，记录到日志
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel()
                               for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # 构建 VLA 数据集与 collator
    overwatch.info(
        f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    # 按配置的混合比例组装 RLDS 数据集，统一处理图像、文本与动作
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),  # 复用视觉骨干的图像标准化
        tokenizer=vlm.llm_backbone.get_tokenizer(),  # 使用语言模型自带的 tokenizer
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,  # 构建与模型格式匹配的提示词
        default_image_resolution=vlm.vision_backbone.default_image_resolution,  # 控制输入图像分辨率
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,  # RLDS 数据混合使用的乱序缓冲区
        image_aug=cfg.image_aug,
    )
    # action_tokenizer 将连续动作离散化，collator 负责 PAD 与掩码以适配自回归训练

    # 保存数据集统计信息，供推理阶段反归一化动作使用
    if overwatch.is_rank_zero():
        # 仅由主进程写入统计文件，避免重复操作
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # 创建训练策略
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    # TrainStrategy 内部封装 FSDP、优化器和学习率调度等组件
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(
        run_dir=run_dir, n_train_examples=len(vla_dataset))
    # run_setup 会完成参数分片、检查点钩子与随机数初始化等准备工作

    # 创建指标记录器，用于实时追踪并写入指定的日志后端
    overwatch.info(
        f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )
    # Metrics 负责写入 JSONL、W&B 等后端，并持久化运行配置

    # 启动训练循环
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )
    # 训练循环由策略对象托管，负责驱动数据加载、前向后向以及定期保存检查点

    # 训练结束，收尾日志
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()
    # finalize 会关闭追踪器并刷新日志，确保数据完整写出

    # 训练流程结束
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
