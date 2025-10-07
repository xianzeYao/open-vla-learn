"""Utils for evaluating policies in LIBERO simulation environments."""

import math  # 提供 acos / sqrt 等数学函数给四元数转换使用
import os  # 处理路径拼接、目录创建等文件系统操作

import imageio  # 用于生成 MP4 视频回放
import numpy as np  # 图像、状态等数值处理
import tensorflow as tf  # 使用 TF 的图像编解码与缩放，保持与 RLDS 生成流程一致
from libero.libero import get_libero_path  # 获取 LIBERO 安装路径下的资源目录
from libero.libero.envs import OffScreenRenderEnv  # 离屏渲染的 MuJoCo 环境

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language  # 任务自带的自然语言指令
    task_bddl_file = os.path.join(  # BDDL 文件描述任务的约束/场景
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl_file,  # 指定 BDDL 文件
        "camera_heights": resolution,  # 设置渲染图像高度
        "camera_widths": resolution,  # 设置渲染图像宽度
    }
    env = OffScreenRenderEnv(**env_args)  # 创建与 OpenVLA 兼容的离屏渲染环境
    env.seed(0)  # 固定 seed，避免即使初始状态固定也出现对象位置漂移
    return env, task_description  # 返回环境实例与语言描述


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]  # 机械臂各自由度为 0，夹爪设为关闭（-1），即“原地等待”


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)  # resize_size 必须是 (height, width)
    img = tf.image.encode_jpeg(img)  # 先编码为 JPEG，RLDS 构建时的处理流程
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # 再解码以确保一致性
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)  # 使用 Lanczos3 插值缩放
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)  # 四舍五入并截断到 [0,255]
    img = img.numpy()  # 转回 numpy 数组供后续处理
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, (int, tuple))  # 支持 int 或 tuple
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)  # 若为 int 则扩展成正方形尺寸
    img = obs["agentview_image"]  # 取 agent 视角 RGB 图
    img = img[::-1, ::-1]  # 旋转 180°，与训练时的数据预处理保持一致
    img = resize_image(img, resize_size)  # 缩放到模型期望尺寸
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"  # 按日期建立回放目录
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )  # 处理成文件名友好的描述
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    )  # 输出视频文件名
    video_writer = imageio.get_writer(mp4_path, fps=30)  # 用 30 FPS 写 MP4
    for img in rollout_images:
        video_writer.append_data(img)  # 逐帧写入
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")  # 控制台提示
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")  # 日志记录
    return mp4_path


def quat2axisangle(quat):
    """将四元数转换为轴角形式，返回单位轴向量乘以旋转角度（弧度）。"""
    if quat[3] > 1.0:
        quat[3] = 1.0  # 夹断 w 分量，避免数值超界导致 acos 报错
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])  # 计算 sin(theta/2)
    if math.isclose(den, 0.0):
        return np.zeros(3)  # 旋转角非常小，直接返回零向量

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den  # axis * angle，axis=xyz/|xyz|
