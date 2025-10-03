"""
materialize.py

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
工厂函数：根据训练策略 ID 构建具体的训练策略实例，目前主要支持 FSDP 相关配置。
"""

from typing import Callable, Optional  # 回调函数与可选类型注解

import torch  # 提供 dtype 等类型定义

from prismatic.models.vlms import PrismaticVLM  # VLM 基类，封装视觉语言模型
from prismatic.training.strategies import FSDPStrategy, TrainingStrategy  # 训练策略抽象与 FSDP 实现

# Registry =>> Maps ID --> {cls(), kwargs} :: supports FSDP for now, but DDP handler is also implemented!
# 训练策略注册表：根据策略名查找对应类与默认参数，目前提供 FSDP 两种分片策略
TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {
        "cls": FSDPStrategy,  # 使用 FSDP 策略类
        "kwargs": {"sharding_strategy": "shard-grad-op"},  # shard-grad-op：启用梯度与参数的部分分片
    },
    "fsdp-full-shard": {
        "cls": FSDPStrategy,
        "kwargs": {"sharding_strategy": "full-shard"},  #   ：梯度、参数与优化器状态全部分片
    },
}


def get_train_strategy(
    train_strategy: str,  # 策略标识符，如 fsdp-shard-grad-op
    vlm: PrismaticVLM,  # 已加载的视觉语言模型实例
    device_id: int,  # 当前进程绑定的 GPU ID
    stage: str,  # 训练阶段，用于决定冻结层等策略
    epochs: int,  # 总训练轮数
    max_steps: Optional[int],  # 可选的最大训练步数
    global_batch_size: int,  # 多卡汇总后的全局 batch size
    per_device_batch_size: int,  # 单卡 batch size
    learning_rate: float,  # 初始学习率
    weight_decay: float,  # 权重衰减系数
    max_grad_norm: float,  # 梯度裁剪阈值
    lr_scheduler_type: str,  # 学习率调度器类型
    warmup_ratio: float,  # 预热占总训练步的比例
    enable_gradient_checkpointing: bool = True,  # 是否开启梯度检查点，省显存
    enable_mixed_precision_training: bool = True,  # 是否启用混合精度训练
    reduce_in_full_precision: bool = False,  # 是否在全精度下进行梯度规约
    mixed_precision_dtype: torch.dtype = torch.bfloat16,  # 混合精度训练使用的数据类型
    worker_init_fn: Optional[Callable[[int], None]] = None,  # DataLoader worker 初始化函数
) -> TrainingStrategy:
    """根据策略名创建训练策略实例，注入训练超参数与模型引用"""
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]  # 取出注册信息，包括类与额外参数
        strategy = strategy_cfg["cls"](
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            **strategy_cfg["kwargs"],
        )
        return strategy
    else:
        raise ValueError(f"Train Strategy `{train_strategy}` is not supported!")
