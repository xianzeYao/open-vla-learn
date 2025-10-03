"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
工厂函数：根据数据混合配置，初始化 Open-X RLDS 数据集、动作 tokenizer 与 collator。
"""

from pathlib import Path  # 统一的路径表示，方便跨平台处理文件系统
from typing import Tuple, Type  # 类型注解：返回值与回调类型

from torch.utils.data import Dataset  # PyTorch 数据集抽象基类
from transformers import PreTrainedTokenizerBase  # HuggingFace tokenizer 抽象基类

from prismatic.models.backbones.llm.prompting import PromptBuilder  # 语言模型提示词构建器
from prismatic.models.backbones.vision import ImageTransform  # 视觉骨干定义的图像变换接口
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # 针对动作预测的 Padding collator
from prismatic.vla.action_tokenizer import ActionTokenizer  # 动作离散化 tokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset  # RLDS 数据集与预处理工具


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    # Step 1：构建动作 tokenizer，确保动作离散化与语言 tokenizer 的特殊符号保持一致
    action_tokenizer = ActionTokenizer(tokenizer)
    # Step 2：构建 batch_transform，将单条 RLDS 样本转换为模型输入
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
    )
    # Step 3：构建 collator，在 DataLoader 层完成 padding、mask 构建和标签对齐
    # 只是实例化一下collator
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Step 4：根据是否按 episode 输出决定使用普通 RLDSDataset 或 EpisodicRLDSDataset
    # episodic为True则就是一个完整的任务序列，反之则是一步步的动作
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )
    # Step 5：初始化数据集：默认只传入高度/宽度；train/image_aug 控制 TFDS 的 shuffle 与图像增强

    return dataset, action_tokenizer, collator
