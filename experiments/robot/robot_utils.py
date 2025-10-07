"""Utils for evaluating robot policies in various environments."""  # 描述：为多种机器人环境评估策略提供工具函数

import os  # 与文件路径、环境变量相关的操作
import random  # Python 内置随机数（设种子时用）
import time  # 时间戳工具，生成日志文件名等

import numpy as np  # 数值计算与数组操作
import torch  # 张量运算与深度学习库

from experiments.robot.openvla_utils import (  # OpenVLA 相关辅助函数
    get_vla,  # 加载 OpenVLA 模型
    get_vla_action,  # 调用 OpenVLA 预测动作
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7  # 动作向量维度（7 = xyz 平移 + xyz 旋转 + 夹爪）
DATE = time.strftime("%Y_%m_%d")  # 日期字符串，方便生成唯一目录
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")  # 日期+时间字符串，生成唯一文件名
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # 选择 GPU 或 CPU
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})  # 设置 numpy 打印格式，浮点保留 3 位

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (  # OpenVLA v0.1 使用的系统提示词
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""  # 统一设置随机种子，保证可复现
    torch.manual_seed(seed)  # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机数
    np.random.seed(seed)  # NumPy 随机数
    random.seed(seed)  # Python 内置随机数
    torch.backends.cudnn.deterministic = True  # cuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 benchmark，避免非确定性
    os.environ["PYTHONHASHSEED"] = str(seed)  # 控制 Python 哈希随机性


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""  # 根据配置加载评估模型
    if cfg.model_family == "openvla":  # 目前仅支持 OpenVLA
        model = get_vla(cfg)  # 调用工具函数加载模型
    else:
        raise ValueError("Unexpected `model_family` found in config.")  # 其他模型族暂不支持
    print(f"Loaded model: {type(model)}")  # 打印模型类型供调试
    return model  # 返回模型实例


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """  # 根据模型类型返回图像输入尺寸
    if cfg.model_family == "openvla":
        resize_size = 224  # OpenVLA 输入默认 224x224
    else:
        raise ValueError("Unexpected `model_family` found in config.")  # 未知模型族
    return resize_size  # 返回尺寸


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""  # 给定观测和指令，调用模型输出动作
    if cfg.model_family == "openvla":
        action = get_vla_action(  # 调用工具函数集成图像/语言处理并得到动作
            model,
            processor,
            cfg.pretrained_checkpoint,
            obs,
            task_label,
            cfg.unnorm_key,
            center_crop=cfg.center_crop,
        )
        assert action.shape == (ACTION_DIM,)  # 确保动作维度正确
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action  # 返回 numpy 动作向量


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """  # 将夹爪维度从 [0,1] 线性映射到 [-1,1]，可选二值化
    orig_low, orig_high = 0.0, 1.0  # 夹爪原始范围
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1  # 映射到 [-1,1]

    if binarize:
        action[..., -1] = np.sign(action[..., -1])  # 二值化：负值设为 -1，非负设为 +1

    return action  # 返回调整后的动作


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """  # 翻转夹爪动作符号（处理数据集约定差异）
    action[..., -1] = action[..., -1] * -1.0  # -1↔+1 对调
    return action  # 返回翻转后的动作
