"""
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
视觉骨干：同时加载 DINOv2 与 SigLIP 的 ViT，并拼接它们的特征输出。
"""

from dataclasses import dataclass  # 描述双分支图像变换的简单数据类
from functools import partial  # 生成携带默认参数的函数，复用 wrap 策略
from typing import Callable, Dict, Tuple  # 类型注解，提升可读性

import timm  # TIMM 提供的预训练视觉模型库
import torch  # 张量运算与深度学习框架
from PIL import Image  # PIL 图像对象，用于自定义变换
from timm.models.vision_transformer import Block, VisionTransformer  # TIMM 中的 ViT 结构定义
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy  # FSDP 包装策略工具
from torchvision.transforms import Compose, Resize  # Torchvision 提供的图像变换组合与缩放

from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple  # 公共辅助类与工具函数

# 支持的 Dino+SigLIP 组合，使用 TIMM 的模型标识符
DINOSigLIP_VISION_BACKBONES = {
    # 224 分辨率组合：指定 DINO 与 SigLIP 在 TIMM 中的注册名
    "dinosiglip-vit-so-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    # 384 分辨率组合：同样列出两套骨干权重
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}


@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform  # DINO 分支的图像变换
    siglip_image_transform: ImageTransform  # SigLIP 分支的图像变换
    is_prismatic: bool = True  # 标记该变换兼容 prismatic 框架

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        # 调用时，分别执行两套图像变换并返回字典
        # 同时对同一张输入图像应用两套变换，返回字典以便后续分别送入两个分支
        return {"dino": self.dino_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class DinoSigLIPViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        # 调用父类构造器，记录骨干 ID、缩放策略与默认分辨率
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        # 根据注册表解析出 DINO 与 SigLIP 在 TIMM 的模型标识符
        self.dino_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["dino"]
        self.siglip_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["siglip"]

        # 分别实例化 DINO 与 SigLIP 的 ViT，并在必要时从 TIMM/HF Hub 下载权重
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dino_featurizer.eval()

        self.siglip_featurizer: VisionTransformer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.siglip_featurizer.eval()

        # Monkey-Patch forward：确保 FSDP 兼容，同时默认取倒数第二层的 patch 表征
        # PyTorch FSDP issue 109385 若修复，可移除此处补丁
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.siglip_featurizer.forward = unpack_tuple(
            partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
        )

        # 分别解析 TIMM 数据配置，并覆盖默认图像尺寸以匹配当前设定
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)
        self.siglip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # 初始化两套默认图像变换
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)

        # SigLIP 默认缩放尺寸大于目标尺寸，需重写为严格 resize
        assert isinstance(default_siglip_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose(
            [
                Resize(self.default_image_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                *default_siglip_transform.transforms[1:],
            ]
        )

        if self.image_resize_strategy == "resize-naive":
            # naive 策略：强行拉伸到方形输入，再延续原始归一化/标准化流程
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_image_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            siglip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                    *default_siglip_transform.transforms[1:],
                ]
            )

            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)

        elif self.image_resize_strategy == "resize-crop":
            # crop 策略：沿用 TIMM 默认配置（通常包含 resize + center crop）
            self.image_transform = DinoSigLIPImageTransform(default_dino_transform, default_siglip_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_transform`!"
            assert (
                "mean" in self.dino_data_cfg and "mean" in self.siglip_data_cfg
            ), "DinoSigLIP `data_cfg` missing `mean`!"

            # 通过均值计算像素填充值（缩放到 0-255），保证 letterbox 无缝衔接
            dino_fill = tuple([int(x * 255) for x in self.dino_data_cfg["mean"]])
            siglip_fill = tuple([int(x * 255) for x in self.siglip_data_cfg["mean"]])

            # 构建带 letterbox 的双分支图像变换
            self.image_transform = DinoSigLIPImageTransform(
                Compose([LetterboxPad(dino_fill), *default_dino_transform.transforms]),
                Compose([LetterboxPad(siglip_fill), *default_siglip_transform.transforms]),
            )

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers.
        返回适用于双分支 ViT 的 FSDP 包装策略：单个 ViT block 与整体模型都会被自动包装。
        """
        # 先设定整体 VisionTransformer 的包裹策略
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        # 再设定单个 transformer block 的自动包裹策略
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        # 使用 _or_policy 组合两种策略，保证块级与整网都能被 FSDP 包裹
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches.
        前向传播：分别将输入送入 DINO 与 SigLIP 骨干，拼接输出 patch 特征。
        """
        # 像素张量字典中键为 "dino"，对应 DINO 分支的输入
        dino_patches = self.dino_featurizer(pixel_values["dino"])
        # 键为 "siglip"，对应 SigLIP 分支的输入
        siglip_patches = self.siglip_featurizer(pixel_values["siglip"])

        # 将两路 patch 在通道维拼接，形成最终特征
        return torch.cat([dino_patches, siglip_patches], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        # 默认分辨率沿用 DINO 的数据配置（两者相同）
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        # 拼接后嵌入维度等于两支 ViT 的 embed_dim 之和
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        # 运行前先确保两支 ViT 的 patch 数量一致
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        # 双分支统一使用 bfloat16 作为半精度类型
        return torch.bfloat16
