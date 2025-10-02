"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
轻量级 PyTorch 数据集定义：把 RLDS/TFDS 管线包装成 OpenVLA 需要的格式，并提供可迭代数据集适配层。
"""

from dataclasses import dataclass  # 用于声明批量变换的轻量数据类
from pathlib import Path  # 统一路径处理工具
from typing import Any, Dict, Tuple, Type  # 类型注解，明确数据结构

import numpy as np  # 数值运算库，主要用于动作统计
import torch  # 张量运算与深度学习框架
from PIL import Image  # PIL 图像对象，便于自定义图像变换
from torch.utils.data import Dataset, IterableDataset  # PyTorch 数据集抽象类
from transformers import PreTrainedTokenizerBase  # HuggingFace tokenizer 抽象基类

from prismatic.models.backbones.llm.prompting import PromptBuilder  # 文本提示构造器
from prismatic.models.backbones.vision import ImageTransform  # 自定义图像变换接口
from prismatic.util.data_utils import tree_map  # 树状映射工具，用于批量操作
from prismatic.vla.action_tokenizer import ActionTokenizer  # 动作离散化 tokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset  # RLDS 数据集构建函数
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights  # OXE 数据混合配置
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType  # 动作归一化方式

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # Step 1：从 RLDS 批数据中取出数据集名称、动作、图像与语言指令
        dataset_name  = rlds_batch["dataset_name"]
        action = rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # Step 2：构造对话式提示词，其中回答部分直接使用离散化后的动作 token
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        # Step 3：使用基础 tokenizer 编码文本提示，得到 input_ids 与初始 labels
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # Step 4：转为张量；图像经过 image_transform 得到 pixel_values
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # Step 5：除动作 token 以外的部分全部设置为 IGNORE_INDEX；若不预测 stop_token，则最后一位也忽略
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        # 记录基础参数：数据根目录、数据混合配置以及 batch_transform
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        # Step 1：解析数据混合配置；支持 OXE 预设混合或单数据集
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),  # 只加载主相机视角
            load_depth=False,  # 不加载深度信息
            load_proprio=False,  # 不加载本体状态
            load_language=True,  # 需要语言指令
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,  # 动作归一化类型
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # 每次只取一个时间步
                future_action_window_size=0,  # 不进行未来动作拼块
                skip_unlabeled=True,  # 跳过没有语言指令的数据
                goal_relabeling_strategy="uniform",  # 暂未使用目标重标策略
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,  # 图像缩放尺寸（高、宽）
                num_parallel_calls=16,  # 并行处理线程数（解码、缩放等 CPU 密集操作）
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # 若启用图像增强，则在 frame_transform_kwargs 中追加增强配置
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Step 2：构建 RLDS 数据集，保存长度与统计信息
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        # 默认使用混合数据集接口，按权重交织采样
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        # 迭代器：逐批获取 numpy 数据，交给 batch_transform 转换为模型输入
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        # 返回数据集中样本总数（由 RLDS 提供）
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        # Episodic 模式只支持单数据集混合
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        # 使用单数据集版本的构建函数，按 episode 输出
        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        # 每次迭代返回一个 episode（列表形式），内部逐步调用 batch_transform
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        # 说明：正式数据集需提供动作反归一化统计，这里使用全 0/1 表示未做归一化。
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        # 需要替换为真实数据集的样本数量
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        # 示例实现：随机生成图像、动作与语言指令
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # 转为张量；图像经过 image_transform 得到模型输入
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # 仅保留动作 token 的损失，前面的文本设为 IGNORE_INDEX
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
