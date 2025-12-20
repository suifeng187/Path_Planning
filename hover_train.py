"""
无人机悬停训练脚本
使用PPO算法训练无人机学习悬停和目标追踪
"""
import argparse
import os
import pickle
import shutil
from importlib import metadata

# 检查rsl-rl-lib版本
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from hover_env import HoverEnv


# 对于PPO算法模型训练的配置
def get_train_cfg(exp_name, max_iterations):
    """
    获取PPO训练配置
    
    Args:
        exp_name: 实验名称
        max_iterations: 最大训练迭代次数
    Returns:
        train_cfg_dict: 训练配置字典
    """
    train_cfg_dict = {
        # PPO算法参数
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,           # PPO裁剪参数
            "desired_kl": 0.01,          # 目标KL散度
            "entropy_coef": 0.004,       # 熵正则化系数
            "gamma": 0.99,               # 折扣因子
            "lam": 0.95,                 # GAE λ参数
            "learning_rate": 0.0003,     # 学习率
            "max_grad_norm": 1.0,        # 梯度裁剪阈值
            "num_learning_epochs": 5,    # 每次更新的训练轮数
            "num_mini_batches": 4,       # 小批量数量
            "schedule": "adaptive",      # 学习率调度策略
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        # 策略网络配置
        "policy": {
            "activation": "tanh",              # 激活函数
            "actor_hidden_dims": [128, 128],   # Actor网络隐藏层
            "critic_hidden_dims": [128, 128],  # Critic网络隐藏层
            "init_noise_std": 1.0,             # 初始噪声标准差
            "class_name": "ActorCritic",
        },
        # 训练运行器配置
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,    # 每个环境每次收集的步数
        "save_interval": 100,        # 模型保存间隔
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict

# cfgs--- configuration 配置的
def get_cfgs():
    """
    获取环境、观测、奖励、指令配置
    
    Returns:
        env_cfg: 环境配置（终止条件、初始位置等）
        obs_cfg: 观测空间配置
        reward_cfg: 奖励函数配置
        command_cfg: 目标指令配置
    """
    # 无人机仿真环境的配置
    env_cfg = {
        "num_actions": 4,                        # 动作维度：4个电机转速
        # 终止条件
        "termination_if_roll_greater_than": 60,    # 横滚角超限(度)
        "termination_if_pitch_greater_than": 60,   # 俯仰角超限(度)
        "termination_if_close_to_ground": 0.05,    # 距地面最小高度(m)
        # 固定起点和终点（扩大距离以容纳更多障碍物）
        "base_init_pos": [0.0, -2.5, 0.15],        # 起点：地面附近
        "goal_position": [0.0, 2.5, 0.15],         # 终点：地面附近（Y方向5米外）
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],    # 初始姿态四元数
        "episode_length_s": 25.0,                  # 单回合最大时长(秒)
        "at_target_threshold": 0.3,                # 到达目标阈值(m)
        "simulate_action_latency": True,           # 是否模拟动作延迟
        "clip_actions": 1.0,                       # 动作裁剪范围
        # 可视化
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        # 障碍物配置：多排柱子形成障碍物阵列
        "obstacle_positions": [
            # 第一排障碍物 (Y = -1.5)
            [-0.5, -1.5, 1.0],
            [0.5, -1.5, 1.0],
            # 第二排障碍物 (Y = -0.5)
            [0.0, -0.5, 1.0],
            [-0.8, -0.5, 1.0],
            [0.8, -0.5, 1.0],
            # 第三排障碍物 (Y = 0.5)
            [-0.4, 0.5, 1.0],
            [0.4, 0.5, 1.0],
            # 第四排障碍物 (Y = 1.5)
            [0.0, 1.5, 1.0],
            [-0.7, 1.5, 1.0],
            [0.7, 1.5, 1.0],
        ],
        "obstacle_radius": 0.12,          # 柱子半径(m)
        "obstacle_height": 2.5,           # 柱子高度(m)
        "obstacle_safe_distance": 0.4,    # 安全距离阈值(m)
        "obstacle_collision_distance": 0.18,  # 碰撞距离阈值(m)
    }
    
    # 观测空间配置，神经网络训练前的预处理工作
    obs_cfg = {
        "num_obs": 17,                # 观测维度
        "obs_scales": {
            "rel_pos": 1 / 3.0,       # 相对位置缩放
            "lin_vel": 1 / 3.0,       # 线速度缩放
            "ang_vel": 1 / 3.14159,   # 角速度缩放
        },
    }
    
    # 奖励配置
    reward_cfg = {
        "yaw_lambda": -5.0,           # 偏航角惩罚系数
        "reward_scales": {
            "target": 20.0,           # 目标追踪奖励（进一步增大）
            "smooth": -1e-4,          # 动作平滑惩罚
            "yaw": 0.01,              # 偏航角奖励
            "angular": -2e-4,         # 角速度惩罚
            "crash": -30.0,           # 坠机惩罚（增大）
            "obstacle": -12.0,        # 避障惩罚（增大，更多障碍物需要更强惩罚）
        },
    }
    
    # 目标指令配置（固定终点模式下不使用随机范围）
    command_cfg = {
        "num_commands": 3,            # 指令维度：x, y, z
        "pos_x_range": [0.0, 0.0],    # 不使用
        "pos_y_range": [1.5, 1.5],    # 不使用
        "pos_z_range": [0.15, 0.15],  # 不使用
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    """
    主函数：解析参数 → 初始化Genesis → 创建环境 → 启动PPO训练
    """
    # 执行命令的格式
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")  # 实验名称
    parser.add_argument("-v", "--vis", action="store_true", default=False)       # 是否可视化
    parser.add_argument("-B", "--num_envs", type=int, default=8192)              # 并行环境数
    parser.add_argument("--max_iterations", type=int, default=301)               # 最大迭代次数
    args = parser.parse_args()

    # 初始化Genesis引擎
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    # 创建日志目录
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 启用可视化时显示目标点
    if args.vis:
        env_cfg["visualize_target"] = True

    # 保存配置文件
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # 创建悬停训练环境
    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    # 创建PPO训练器并开始训练
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# 训练命令示例
python hover_train.py -e drone-hovering -B 8192 --max_iterations 301 -v
"""
