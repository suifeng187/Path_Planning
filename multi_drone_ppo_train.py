"""
多无人机避障路径规划训练脚本 - PPO版本
将多架无人机视为一个整体，使用标准PPO算法训练
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
from multi_drone_ppo_env import MultiDronePPOEnv


def get_train_cfg(exp_name, max_iterations):
    """
    获取PPO训练配置
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.015,  # v1: 0.01 从 0.01 提高到 0.015（增加探索，让所有无人机都能学习）
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0002,  # v1: 0.0005 从 0.0005 降到 0.0002（更稳定的学习，避免策略分化）
            "max_grad_norm": 0.5,  # v1: 1.0 从 1.0 降到 0.5（更严格的梯度裁剪，提高稳定性）
            "num_learning_epochs": 4,  # v1: 4 从 4 提高到 5（每次更新更充分，提高学习效率）
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.2,  # v1: 1.0 从 1.0 降到 0.5（降低价值函数损失的影响，解决loss过大的问题）
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 0.5,
            "class_name": "ActorCritic",
        },
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
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    """
    获取环境配置
    """
    num_drones = 3
    
    env_cfg = {
        "num_drones": num_drones,
        "num_actions": 4,  # 每架无人机4个电机
        "termination_if_roll_greater_than": 80,
        "termination_if_pitch_greater_than": 80,
        "termination_if_close_to_ground": 0.02,
        "drone_init_positions": [
            [-1.0, -2.5, 0.8],
            [0.0, -2.5, 0.8],
            [1.0, -2.5, 0.8],
        ],
        "drone_goal_positions": [
            [-1.0, 2.5, 0.8],
            [0.0, 2.5, 0.8],
            [1.0, 2.5, 0.8],
        ],
        "episode_length_s": 35.0,
        "at_target_threshold": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "obstacle_positions": [
            # 第1排（y=-1.5）：6个柱子，间距0.9
            [-2.25, -1.5, 1.0], [-1.35, -1.5, 1.0], [-0.45, -1.5, 1.0], [0.45, -1.5, 1.0], [1.35, -1.5, 1.0], [2.25, -1.5, 1.0],
            # 第2排（y=-0.5）：5个柱子，间距0.9，插空排列
            [-1.8, -0.5, 1.0], [-0.9, -0.5, 1.0], [0.0, -0.5, 1.0], [0.9, -0.5, 1.0], [1.8, -0.5, 1.0],
            # 第3排（y=0.5）：6个柱子，同第1排
            [-2.25,  0.5, 1.0], [-1.35,  0.5, 1.0], [-0.45, 0.5, 1.0], [0.45, 0.5, 1.0], [1.35, 0.5, 1.0], [2.25, 0.5, 1.0],
            # 第4排（y=1.5）：5个柱子，同第2排
            [-1.8,  1.5, 1.0], [-0.9, 1.5, 1.0], [0.0, 1.5, 1.0], [0.9, 1.5, 1.0], [1.8, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,
        "obstacle_height": 2.5,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.1,
        "drone_safe_distance": 0.2,
        "drone_collision_distance": 0.1,
    }
    
    # PPO版本：观测维度 = 每架无人机观测 * 无人机数量 + 其他无人机相对位置
    # 每架无人机基础观测: 17维 (rel_pos:3 + quat:4 + lin_vel:3 + ang_vel:3 + last_action:4)
    # 加上其他无人机的相对位置: (num_drones-1) * 3
    obs_per_drone = 17 + (num_drones - 1) * 3  # 17 + 6 = 23
    
    obs_cfg = {
        "num_obs": obs_per_drone * num_drones,  # 23 * 3 = 69
        "num_obs_per_drone": obs_per_drone,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    
    reward_cfg = {
        "reward_scales": {
            "target": 75.0,  # v1: 50  从 50.0 提高到 80.0（更强调到达终点）
            "progress": 30.0,
            "alive": 3.0,
            "smooth": -1e-6,
            "crash": -10.0,  # v1: -5.0 从 -5.0 提高到 -8.0（更怕碰撞，强制学习避障）
            "obstacle": -4.0,  # v1 : -1 从 -1 提高到 -3.0（更怕撞柱子，特别是最后一排）
            "separation": -0.5,
            # 新增：朝向目标直线飞行的奖励（鼓励走“直线+穿柱子”的路线）
            "direction": 20.0, # v1:10 从 10.0 提高到 20.0（更强调直线飞行）
        },
    }
    
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-ppo")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=800)
    parser.add_argument("--resume", action="store_true", default=False, help="从检查点恢复训练")
    parser.add_argument("--ckpt", type=int, default=-1, help="检查点迭代数，-1表示加载最新模型（仅在--resume时有效）")
    parser.add_argument("--update_config", action="store_true", default=False, help="恢复训练时使用新的配置参数（而不是旧的cfgs.pkl）")
    args = parser.parse_args()

    # 初始化Genesis引擎（无头模式通过不创建viewer实现）
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    
    # 如果恢复训练，从已有配置加载；否则创建新配置
    if args.resume:
        if not os.path.exists(log_dir):
            print(f"错误：日志目录 {log_dir} 不存在，无法恢复训练")
            return
        
        cfg_path = f"{log_dir}/cfgs.pkl"
        
        # 如果指定了 --update_config，使用新配置；否则从旧配置加载
        if args.update_config:
            print("[更新配置] 使用新的配置参数，而不是旧的 cfgs.pkl")
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
            train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
            # 保存新配置
            pickle.dump(
                [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
                open(cfg_path, "wb"),
            )
        else:
            if not os.path.exists(cfg_path):
                print(f"错误：配置文件 {cfg_path} 不存在，无法恢复训练")
                return
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
            # 更新最大迭代次数（如果需要继续训练更多轮）
            train_cfg["runner"]["max_iterations"] = args.max_iterations
        
        # 确定检查点路径
        if args.ckpt > 0:
            resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        else:
            # 查找最新的模型
            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if not model_files:
                print(f"错误：在 {log_dir} 中未找到模型文件")
                return
            model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
            resume_path = os.path.join(log_dir, model_files[-1])
        
        if not os.path.exists(resume_path):
            print(f"错误：检查点文件 {resume_path} 不存在")
            available = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if available:
                print(f"可用的检查点: {available}")
            return
        
        print(f"从检查点恢复训练: {resume_path}")
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["resume_path"] = resume_path
    else:
        # 新训练：创建新配置
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
        
        # 如果目录存在，删除旧数据
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存配置
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)
        print(f"[Visualization Mode] Reduced num_envs to {args.num_envs}")

    env = MultiDronePPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # 如果恢复训练，加载检查点
    if args.resume:
        runner.load(resume_path)
        print(f"已加载检查点，继续训练...")
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# PPO多无人机训练命令

# 1. 新训练（无可视化，快速训练）
python multi_drone_ppo_train.py -e multi-drone-ppo-v2 -B 4096 --max_iterations 800

# 2. 新训练（带可视化）
python multi_drone_ppo_train.py -e multi-drone-ppo -B 64 --max_iterations 800 -v

# 3. 从最新检查点恢复训练（微调）
python multi_drone_ppo_train.py -e multi-drone-ppo -B 4096 --max_iterations 1000 --resume

# 4. 从指定检查点恢复训练（微调）
python multi_drone_ppo_train.py -e multi-drone-ppo-v2 -B 8192 --max_iterations 1000 --resume --ckpt 400

# 5. 微调训练（带可视化）
python multi_drone_ppo_train.py -e multi-drone-ppo -B 64 --max_iterations 1000 --resume -v
"""
