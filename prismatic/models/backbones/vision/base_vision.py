"""
base_vision.py

本模块定义视觉骨干网络的抽象基类，集中描述特征提取器的公共接口、工具函数与初始化流程，同时提供
TimmViTBackbone 作为 TIMM Vision Transformer 的通用封装。
"""

from abc import ABC, abstractmethod  # 定义抽象基类与抽象方法
from dataclasses import dataclass  # 轻量级数据容器
from functools import partial  # 生成携带默认参数的函数
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union  # 常用类型注解

import timm  # PyTorch 图像模型库，提供丰富的视觉骨干
import torch  # 张量与自动求导框架
import torch.nn as nn  # 神经网络组件
import torchvision.transforms.functional as TVF  # 图像变换函数
from PIL.Image import Image  # PIL 图像对象
from timm.models.vision_transformer import Block, VisionTransformer  # TIMM 中的 ViT 模块定义
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy  # FSDP 包裹策略工具
from torchvision.transforms import Compose, Resize  # Torchvision 图像变换组合与缩放


# === 用于猴子补丁的工具函数 ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    # TIMM 某些接口返回元组，此处包装成只取首个元素，方便后续调用
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === 图像变换接口定义，约束实现需可调用且返回张量或张量字典 ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === 自定义 Torchvision 图像变换 ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        # Letterbox 策略：利用对称填充补齐短边，保持长宽比不变
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


# === 视觉骨干网络的抽象基类，统一管理特征提取器、图像预处理和模型元信息 ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # 视觉骨干默认持有的组件：特征提取器与图像预处理器
        self.featurizer: nn.Module = None  # 视觉编码主干，负责将图像映射到 patch/grid 表征
        self.image_transform: ImageTransform = None  # 与主干匹配的图像预处理流程（裁剪、归一化等）

    def get_image_transform(self) -> ImageTransform:
        # 返回当前骨干默认使用的图像预处理函数
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


# === TIMM Vision Transformer 骨干的抽象基类，统一封装加载与预处理逻辑 ===
class TimmViTBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: Optional[str] = None,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # 初始化 ViT 特征提取器，必要时会自动从 TIMM 或 HF Hub 下载权重
        if self.override_act_layer is None:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
            )
        else:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size,
                act_layer=self.override_act_layer,
            )
        self.featurizer.eval()

        # 为确保与 FSDP 兼容，对 forward 进行猴子补丁，默认返回倒数第二层的 patch 表征
        # 待 PyTorch 相关问题修复后可移除此逻辑（参考 issue 109385）
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        # 目前仅支持 TIMM VisionTransformer，如需扩展其它视觉表征需补充实现
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # 读取 TIMM 的数据配置，并根据 default_image_size 覆写默认输入尺寸
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # 根据数据配置创建默认图像变换，后续会结合 resize 策略做调整
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # 部分模型（SigLIP/IN1K）默认缩放尺寸大于目标尺寸，这里重写为严格 resize
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # 按照配置的 resize 策略构建最终图像变换流水线
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # 计算填充颜色，基于归一化均值缩放至 0-255
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # 构建带有 letterbox 的新变换流水线
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回一个用于 FSDP 的包裹策略：逐个包装 ViT 模块，并保证整体特征提取器被覆盖。"""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """将预处理后的图像张量送入视觉骨干，返回完整的 patch 表征。"""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
