"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).
简述：基于 HF AutoClasses 加载 OpenVLA，并用 PEFT 的 LoRA 在小数据/低显存场景下进行参数高效微调；
同时支持可选的 4bit 量化与多卡 DDP 训练、按步保存与合并适配器权重。

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os  # 与实验目录、环境变量相关的工具函数
from collections import deque  # 简单滑窗数据结构，用于平滑指标
from dataclasses import dataclass  # 定义配置 dataclass
from pathlib import Path  # 平台无关的文件路径操作
from typing import Optional  # 类型注解：可选值

import draccus  # 将 dataclass 映射为命令行参数/配置文件
import torch  # 核心张量与自动求导库
import torch.distributed as dist  # PyTorch 分布式通信原语
import tqdm  # 控制台进度条
from accelerate import PartialState  # Accelerate 的轻量分布式状态封装
# PEFT/LoRA 核心 API
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP  # PyTorch 原生 DDP 包装器
from torch.optim import AdamW  # 经典 AdamW 优化器
from torch.utils.data import DataLoader  # 数据加载器
# Vision2Seq 模型及 4bit 配置
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor  # 注册 AutoClasses 时需要
# HF 自回归模型的输出结构（含 loss/logits）
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb  # Weights & Biases 日志追踪
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder  # Prompt 构造器
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # 动作预测专用 collator
from prismatic.vla.action_tokenizer import ActionTokenizer  # 连续动作 ↔ 离散 token
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset  # RLDS 数据集与批量变换
# 保存数据统计（反归一化用）
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # HF AutoConfig 注册
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # HF 推理模型类
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # HF 处理器

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 并行化警告，保持日志干净


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # 要微调的 OpenVLA 模型路径（Hub/本地均可）

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # RLDS 数据所在根目录
    dataset_name: str = "droid_wipe"                                # 子数据集名称（决定加载的任务）
    run_root_dir: Path = Path("runs")                               # 微调日志与权重的保存目录
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # LoRA 适配器临时目录（合并前存放）

    # Fine-tuning Parameters
    batch_size: int = 16                                            # 单进程/单卡 batch 大小
    max_steps: int = 2000_000                                        # 最多训练多少个梯度步
    save_steps: int = 5000                                          # 每隔多少步保存检查点
    learning_rate: float = 5e-4                                     # AdamW 学习率
    grad_accumulation_steps: int = 1                                # 梯度累积步数（>1 可扩大有效 batch）
    image_aug: bool = True                                          # 是否开启图像增强
    shuffle_buffer_size: int = 100_000                              # RLDS 数据的 shuffle 缓冲区
    save_latest_checkpoint_only: bool = True                        # True=只保留最近一次检查点

    # LoRA Arguments
    use_lora: bool = True                                           # 是否启用 LoRA 微调
    lora_rank: int = 32                                             # LoRA 低秩矩阵的秩 r
    lora_dropout: float = 0.0                                       # LoRA 适配器 dropout
    use_quantization: bool = False                                  # 是否 4bit 量化 LoRA 模型（节省显存）

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # W&B 项目名
    wandb_entity: str = "stanford-voltron"                          # W&B 工作区
    run_id_note: Optional[str] = None                               # 额外标记（附加到 run_id）
    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # 打印微调基本信息
    print(
        f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    assert torch.cuda.is_available(
    ), "Fine-tuning assumes at least one GPU is available!"  # 确保至少有 1 张 GPU
    distributed_state = PartialState()  # 构建分布式状态，获取 local_process_index 等
    torch.cuda.set_device(
        device_id := distributed_state.local_process_index)  # 设置当前进程使用的 GPU
    torch.cuda.empty_cache()  # 清理显存碎片

    exp_id = (  # 组装实验 ID（模型名+数据集+有效 batch+学习率）
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        # 标记 LoRA 配置
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"  # 标记 4bit 量化
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"  # 附加自定义标记
    if cfg.image_aug:
        exp_id += "--image_aug"  # 记录是否使用图像增强

    run_dir = cfg.run_root_dir / exp_id  # 日志与权重保存目录
    adapter_dir = cfg.adapter_tmp_dir / exp_id  # LoRA 适配器临时目录
    os.makedirs(run_dir, exist_ok=True)  # 创建输出目录
    os.makedirs(adapter_dir, exist_ok=True)  # 创建适配器目录

    quantization_config = None  # 默认不量化
    if cfg.use_quantization:  # 开启 4bit 量化
        # 限制：只有 LoRA 场景使用
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(  # NF4 低比特配置
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    # 注册 OpenVLA 配置类，供 AutoConfig 使用
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(
        OpenVLAConfig, PrismaticImageProcessor)  # 注册自定义图像处理器
    # 注册同时处理图像/文本的 Processor
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction)  # 注册模型类

    processor = AutoProcessor.from_pretrained(
        cfg.vla_path, trust_remote_code=True)  # 加载图像处理 + tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(  # 加载 VLA 模型
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)  # 4bit 场景：冻结规范层、准备量化训练
    else:
        vla = vla.to(device_id)  # 非量化：直接送入目标 GPU

    if cfg.use_lora:
        lora_config = LoraConfig(  # LoRA 配置
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)  # 注入 LoRA 适配器
        vla.print_trainable_parameters()  # 打印可训练参数比例（便于确认注入结果）

    vla = DDP(  # 用 DDP 包装模型，支持多 GPU 训练
        vla,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    trainable_params = [param for param in vla.parameters()
                        if param.requires_grad]  # 收集可训练参数
    # 设置优化器（可按需加 weight_decay）
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(
        processor.tokenizer)  # 根据 tokenizer 构建动作离散化器

    batch_transform = RLDSBatchTransform(  # RLDS 批量转换：图像预处理 + prompt + 动作 token
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(  # 构建 RLDS IterableDataset
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(
            vla_dataset.dataset_statistics, run_dir)  # 保存数据归一化统计

    collator = PaddedCollatorForActionPrediction(  # 构建 collator：负责 padding 与 mask
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(  # 构建 DataLoader；RLDS 内部已并行因此 num_workers=0
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=None,  # 如需自定义 worker 初始化，可在此添加回调
    )

    if distributed_state.is_main_process:
        wandb.init(  # 初始化 W&B 追踪
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"ft+{exp_id}",
        )

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)  # 用于平滑 loss
    recent_action_accuracies = deque(
        maxlen=cfg.grad_accumulation_steps)  # 平滑动作准确率
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)  # 平滑 L1 误差

    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=not distributed_state.is_main_process) as progress:
        vla.train()  # 模型设为训练模式
        optimizer.zero_grad()  # 清空梯度
        # 迭代 RLDS 数据，直到达到 max_steps
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):  # 开启 BF16 混合精度
                output: CausalLMOutputWithPast = vla(  # 前向传播返回 HF 自回归输出
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(
                        torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss  # HF 内部交叉熵损失（仅动作 token 区域参与）

            normalized_loss = loss / cfg.grad_accumulation_steps  # 按累积步数缩放 loss
            normalized_loss.backward()  # 反向传播（梯度由 DDP 管理）

            action_logits = output.logits[:,
                                          vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
            action_preds = action_logits.argmax(dim=2)  # 动作 token 预测
            action_gt = batch["labels"][:, 1:].to(
                action_preds.device)  # 去掉 BOS 与预测对齐
            mask = action_gt > action_tokenizer.action_token_begin_idx  # 仅统计动作 token 区域
            # 每个 token 是否预测正确
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()  # 动作 token 准确率
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(
                continuous_actions_pred, continuous_actions_gt)  # 连续动作 L1

            recent_losses.append(loss.item())  # 记录当前 step loss
            recent_action_accuracies.append(action_accuracy.item())  # 记录动作准确率
            recent_l1_losses.append(action_l1_loss.item())  # 记录 L1

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps  # 当前梯度步编号
            smoothened_loss = sum(recent_losses) / \
                len(recent_losses)  # 平滑 loss
            smoothened_action_accuracy = sum(
                recent_action_accuracies) / len(recent_action_accuracies)  # 平滑准确率
            smoothened_l1_loss = sum(recent_l1_losses) / \
                len(recent_l1_losses)  # 平滑 L1

            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(  # 每 10 个梯度步推送一次 W&B 指标
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:  # 达到累积步，执行优化器更新
                optimizer.step()
                optimizer.zero_grad()
                if distributed_state.is_main_process:
                    progress.update()  # 更新进度条

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:  # 到达保存间隔
                if distributed_state.is_main_process:
                    print(
                        f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir  # LoRA：先保存适配器
                    processor.save_pretrained(run_dir)  # 保存处理器（图像+tokenizer）
                    vla.module.save_pretrained(save_dir)  # 保存模型/适配器权重
                dist.barrier()  # 等待主进程写完

                if cfg.use_lora:  # LoRA 额外合并适配器 → 完整模型
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                    merged_vla = PeftModel.from_pretrained(
                        base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)  # 覆写最新合并后权重
                            print(
                                f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            checkpoint_dir = Path(
                                str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(
                                vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(
                                f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                dist.barrier()  # 所有进程同步

            if gradient_step_idx == cfg.max_steps:  # 达到最大步数即终止
                print(
                    f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()  # 入口：draccus.wrap 会自动解析 CLI 参数并调用
