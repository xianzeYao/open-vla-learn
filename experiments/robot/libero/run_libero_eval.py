"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os  # 文件与日志路径管理
import sys  # 修改 sys.path 以便导入上级目录模块
from dataclasses import dataclass  # 配置结构体
from pathlib import Path  # 跨平台路径操作
from typing import Optional, Union  # 类型注解：可选值/联合类型

import draccus  # 将 dataclass 暴露为命令行参数
import numpy as np  # 数值运算库
import tqdm  # 任务进度条
from libero.libero import benchmark  # 官方 LIBERO benchmark 接口

import wandb  # Weights & Biases 日志

# 让解释器能找到 experiments.robot 包（相对路径上移两级）
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (  # LIBERO 评估工具函数
    get_libero_dummy_action,  # 获取“空动作”用于等待阶段
    get_libero_env,  # 构建 LIBERO 环境
    get_libero_image,  # 图像预处理与缩放
    quat2axisangle,  # 四元数 → 轴角转换
    save_rollout_video,  # 保存回放视频
)
from experiments.robot.openvla_utils import get_processor  # OpenVLA Processor 加载工具
from experiments.robot.robot_utils import (  # 机器人通用工具（动作归一化、模型加载等）
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # 使用的模型族（默认 openvla，可扩展到其他策略）
    pretrained_checkpoint: Union[str, Path] = ""     # 微调或预训练权重路径
    load_in_8bit: bool = False                       # 是否以 8bit 量化载入（仅 openvla 支持）
    load_in_4bit: bool = False                       # 是否以 4bit 量化载入（仅 openvla 支持）

    center_crop: bool = True                         # 是否对输入图像做中心裁剪（若训练时用了随机裁剪，这里要 True）

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # 评估的 LIBERO 任务集
    num_steps_wait: int = 10                         # 刚开始等待的空步骤数（让物体稳定）
    num_trials_per_task: int = 50                    # 每个任务重复多少次实验

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # 运行 ID 追加说明，便于区分日志
    local_log_dir: str = "./experiments/logs"        # 本地日志输出目录

    use_wandb: bool = False                          # 是否启用 W&B 记录
    wandb_project: str = "YOUR_WANDB_PROJECT"        # W&B 项目名
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # W&B 工作区

    seed: int = 7                                    # 随机种子，保证可复现

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    """执行 LIBERO 评估流程，根据配置加载模型、环境并循环任务/episode"""
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"  # 必须提供模型权重
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"  # 若训练用了增强，评估要对齐裁剪策略
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"  # 不允许同时指定 8/4bit

    set_seed_everywhere(cfg.seed)  # 设置随机种子

    cfg.unnorm_key = cfg.task_suite_name  # 动作反归一化键默认为任务集名称
    model = get_model(cfg)  # 加载策略模型（openvla 或其他家族）

    if cfg.model_family == "openvla":  # 仅对 openvla 校验反归一化统计是否存在
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"  # 某些数据集后缀有 _no_noops
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = None  # Processor 包含图像处理 + tokenizer
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)  # 与训练保持一致的 Processor（决定图像变换与 prompt 构造）

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"  # 生成日志 ID
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"  # 附加备注
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")  # 打开本地日志文件
    print(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:  # 初始化 W&B
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    benchmark_dict = benchmark.get_benchmark_dict()  # 获取全部任务集
    task_suite = benchmark_dict[cfg.task_suite_name]()  # 实例化选定任务集
    num_tasks_in_suite = task_suite.n_tasks  # 任务数量
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = get_image_resize_size(cfg)  # 根据模型要求计算输入图像尺寸

    total_episodes, total_successes = 0, 0  # 累计 episode 与成功次数
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):  # 逐任务评估
        task = task_suite.get_task(task_id)  # 取任务描述
        initial_states = task_suite.get_task_init_states(task_id)  # 官方预设初始状态
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)  # 构建环境与文本描述

        task_episodes, task_successes = 0, 0  # 该任务内的统计
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):  # 重复多个 episode
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()  # 环境复位（机械臂回初始状态）
            obs = env.set_init_state(initial_states[episode_idx])  # 设置特定初始场景

            t = 0  # 时间步计数
            replay_images = []  # 记录帧，用于生成回放视频
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:  # 继续循环直到达到最大步数+等待步
                try:
                    if t < cfg.num_steps_wait:  # 前若干步执行“空动作”，等待物体稳定
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)  # 图像预处理（裁剪、缩放、归一化）
                    replay_images.append(img)  # 保存帧供回放视频使用

                    observation = {  # 构造模型输入（图像 + 状态）
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],  # 末端位置
                                quat2axisangle(obs["robot0_eef_quat"]),  # 姿态 (四元数→轴角)
                                obs["robot0_gripper_qpos"],  # 夹爪开合
                            )
                        ),
                    }

                    action = get_action(  # 让策略模型输出动作
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    action = normalize_gripper_action(action, binarize=True)  # 夹爪控制映射到 [-1,1]
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)  # OpenVLA 训练约定下夹爪符号需翻转

                    obs, reward, done, info = env.step(action.tolist())  # 执行动作
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:  # 捕获仿真异常，写日志并终止当前 episode
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(  # 保存回放视频（供人工回顾）
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    log_file.close()  # 关闭本地日志文件

    if cfg.use_wandb:  # 记录总成功率并上传日志文件到 W&B
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
