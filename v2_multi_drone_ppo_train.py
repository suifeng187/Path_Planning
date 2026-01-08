"""
多无人机避障路径规划训练脚本 - PPO版本 (Local Sensing + 4D Teammate + Top-2 Obstacle)
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
from v2_multi_drone_ppo_env import MultiDronePPOEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.00015,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.2,
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
        "num_actions": 4,
        "termination_if_roll_greater_than": 60,
        "termination_if_pitch_greater_than": 60,
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
        "at_target_threshold": 0.4,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        
        # === 感知配置 (核心修改) ===
        "sensing_radius": 3.0,          # 局部感知半径 3米
        "num_nearest_obstacles": 2,     # 只关注最近的2个障碍物
        # =========================

        "obstacle_positions": [
            # 第1排
            [-2.25, -1.5, 1.0], [-1.35, -1.5, 1.0], [-0.45, -1.5, 1.0], [0.45, -1.5, 1.0], [1.35, -1.5, 1.0], [2.25, -1.5, 1.0],
            # 第2排
            [-1.8, -0.5, 1.0], [-0.9, -0.5, 1.0], [0.0, -0.5, 1.0], [0.9, -0.5, 1.0], [1.8, -0.5, 1.0],
            # 第3排
            [-2.25,  0.5, 1.0], [-1.35,  0.5, 1.0], [-0.45, 0.5, 1.0], [0.45, 0.5, 1.0], [1.35, 0.5, 1.0], [2.25, 0.5, 1.0],
            # 第4排
            [-1.8,  1.5, 1.0], [-0.9, 1.5, 1.0], [0.0, 1.5, 1.0], [0.9, 1.5, 1.0], [1.8, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,
        "obstacle_height": 2.5,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.1,
        "drone_safe_distance": 0.2,
        "drone_collision_distance": 0.1,
    }
    
    # === 观测维度计算 (核心修改) ===
    num_nearest_obs = env_cfg["num_nearest_obstacles"]
    
    # 1. 基础状态 (17维)
    # 2. 队友感知 (4维): [rel_x, rel_y, rel_z, visibility_mask] * (num_drones - 1)
    # 3. 障碍物感知 (3维): [rel_x, rel_y, rel_z] * num_nearest_obs
    
    obs_per_drone = 17 + (num_drones - 1) * 4 + (num_nearest_obs * 3)
    
    obs_cfg = {
        "num_obs": obs_per_drone * num_drones,
        "num_obs_per_drone": obs_per_drone,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
            "obstacle": 1 / 3.0, 
        },
    }
    
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 30.0,      
            "progress": 1,     
            "alive": 0.01,
            "smooth": -1e-3,
            "yaw": 0.5,
            "angular": -0.05,
            "crash": -15,
            "obstacle": -5,
            "separation": -1,
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
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--update_config", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    
    if args.resume:
        if not os.path.exists(log_dir):
            print(f"错误：日志目录 {log_dir} 不存在，无法恢复训练")
            return
        
        cfg_path = f"{log_dir}/cfgs.pkl"
        
        if args.update_config:
            print("[更新配置] 使用新的配置参数")
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
            train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
            pickle.dump(
                [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
                open(cfg_path, "wb"),
            )
        else:
            if not os.path.exists(cfg_path):
                print(f"错误：配置文件 {cfg_path} 不存在")
                return
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
            train_cfg["runner"]["max_iterations"] = args.max_iterations
        
        if args.ckpt > 0:
            resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        else:
            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if not model_files:
                print(f"错误：在 {log_dir} 中未找到模型文件")
                return
            model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
            resume_path = os.path.join(log_dir, model_files[-1])
        
        print(f"从检查点恢复训练: {resume_path}")
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["resume_path"] = resume_path
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
        
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)

    env = MultiDronePPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume:
        runner.load(resume_path)
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# PPO多无人机训练命令

# 1. 新训练（无可视化，快速训练）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo-v2 -B 4096 --max_iterations 800

# 2. 新训练（带可视化）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo -B 64 --max_iterations 800 -v

# 3. 从最新检查点恢复训练（微调）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo -B 4096 --max_iterations 1000 --resume

# 4. 从指定检查点恢复训练（微调）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo -B 8192 --max_iterations 1000 --resume --ckpt 400

# 5. 微调训练（带可视化）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo -B 64 --max_iterations 1000 --resume -v

# 6. 微调（加载新配置）
python v2_multi_drone_ppo_train.py -e v2-multi-drone-ppo -B 8192 --max_iterations 1000 --resume --ckpt 400 --update_config
"""