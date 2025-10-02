"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
加载预训练 VLM/VLA 模型的总入口：提供可用模型列表、模型描述查询，以及从本地或 Hugging Face Hub 获取权重的工具函数。
"""

import json  # JSON 解析：读取 config.json 获得模型结构配置
import os  # 文件系统操作：判断本地路径、环境变量等
from pathlib import Path  # Pathlib 封装的跨平台路径操作
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download  # 访问 HF Hub 的文件接口

from prismatic.conf import ModelConfig  # 模型配置注册表，描述视觉/语言骨干的元信息
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY  # 模型 ID 与别名映射
from prismatic.models.vlas import OpenVLA  # OpenVLA 模型定义
from prismatic.models.vlms import PrismaticVLM  # 通用 VLM 基类
from prismatic.overwatch import initialize_overwatch  # Logging utility for distributed setups / 日志工具封装分布式 logging
from prismatic.vla.action_tokenizer import ActionTokenizer  # 连续动作离散化成 token 的工具

# Initialize Overwatch =>> Wraps `logging.Logger` / 初始化 Overwatch 日志器，负责跨进程同步日志与控制台格式
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
# Repositories holding official checkpoints; used to download configs & weights / 存放官方发布模型的仓库路径，供下载配置与权重
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"


# === Available Models ===
# Enumerate registered models for CLI/script usage / 列举当前注册的模型信息，便于 CLI/脚本调用
def available_models() -> List[str]:
    """返回在模型注册表中可用的模型 ID 列表"""
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    """返回带有别名/描述的模型表，用于命令行展示"""
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    """打印并返回指定模型的详细介绍，便于快速了解架构与用途"""
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub.
    加载预训练的 PrismaticVLM，可从本地目录或 Hugging Face Hub 获取。"""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint / 构造 config.json 与 checkpoint 路径（通常位于 run_dir/checkpoints 内）
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]
    # config.json records vision/LLM backbones and arch metadata / 包含视觉骨干、语言骨干与结构说明，确保加载流程自描述

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    # Instantiate vision backbone + preprocessing transform per config / 根据配置实例化视觉骨干及图像预处理，保持训练/推理对齐

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )
    # inference_mode toggles caching & parameter freezing / 控制是否开启缓存、冻结参数；训练模式会返回可微 backbone

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )
    # from_pretrained loads weights into backbones and honors freeze flag / 根据 freeze_weights 决定是否允许微调

    return vlm


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
) -> OpenVLA:
    """Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub.
    加载预训练 OpenVLA，可指定本地 checkpoint 或 HF Hub 目录。"""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # Validate checkpoint layout `.../<RUN_ID>/checkpoints/*.pt` / 校验 checkpoint 路径格式需位于 <RUN_ID>/checkpoints 下且扩展名为 .pt
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify checkpoint using optional `step_to_load` (latest if None) / 根据 step_to_load 挑选权重，默认为最新
        step_to_load = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load if step_to_load is not None else ''}*.pt")
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()
    # VLA config references the base VLM choice / 指向基础 VLM 类型，可据此获取完整骨干描述

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)
    # norm_stats support action de-normalization during inference / 推理阶段用于动作反归一化

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )
    # Reuse the vision backbone from base VLM / 共用视觉骨干，并根据训练策略决定是否冻结

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )
    # Tokenizer reused by ActionTokenizer ensuring shared vocab / 供动作 tokenizer 复用，保证词表统一

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer())
    # Map continuous controls to tail tokens / 将连续控制量映射到词表末端的离散 token

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
    )
    # OpenVLA 继承自 PrismaticVLM，额外需要 norm_stats 和 action_tokenizer 来支持动作预测

    return vla
