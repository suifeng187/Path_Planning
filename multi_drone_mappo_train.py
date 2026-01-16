"""
多无人机避障路径规划训练脚本 - MAPPO版本 (圆形障碍物与多轮任务)
【修改说明】
1. 邻居观测移除相对姿态，维度降低。
2. 奖励函数取消 Yaw 约束，鼓励全向飞行。
"""
import argparse
import os
import pickle
import shutil
import math
import random
import genesis as gs
from multi_drone_mappo_env import MultiDroneMAPPOEnv
from rsl_rl.runners import OnPolicyRunner

try:
    from mappo_algorithm import MAPPORunner 
except ImportError:
    print("Error: Could not import 'mappo_algorithm.py'.")
    exit()

def generate_circular_obstacles(max_radius=3, obstacle_radius=0.1, spacing=1):
    """生成圆形障碍物阵列"""
    obstacles = []
    start_radius = 0.8 
    current_radius = start_radius
    while current_radius <= max_radius:
        circumference = 2 * math.pi * current_radius
        num_obs_in_ring = int(circumference / spacing)
        for i in range(num_obs_in_ring):
            angle = 2 * math.pi * i / num_obs_in_ring
            x = current_radius * math.cos(angle)
            y = current_radius * math.sin(angle)
            obstacles.append([x, y, 1.0])
        current_radius += spacing * 0.9 
    return obstacles

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "MAPPO", 
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.5,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64], 
            "critic_hidden_dims": [512, 256, 128], 
            "init_noise_std": 0.5,
            "class_name": "ActorCritic", 
        },
        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "log_interval": 1,
            "checkpoint": -1,
        },
        "num_steps_per_env": 100, 
        "save_interval": 50,
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    num_drones = 3
    
    # 生成圆形障碍物分布
    obstacles = generate_circular_obstacles(max_radius=3, obstacle_radius=0.1, spacing=1.0)
    
    env_cfg = {
        "num_drones": num_drones,
        "num_actions": 4,
        "team_spirit": 0.2, # 团队合作系数
        "termination_if_roll_greater_than": 60,
        "termination_if_pitch_greater_than": 60,
        "termination_if_close_to_ground": 0.02,
        # 初始位置：距离圆形区域(R=3.5) 1米，即 x = -4.5 附近
        "drone_init_positions": [
            [-4, -1.5, 1.0], [-4, 0.0, 1.0], [-4, 1.5, 1.0],
        ],
        # 目标位置不再固定，将在 Env 中随机生成，此处留空或作为占位
        "drone_goal_positions": [], 
        "episode_length_s": 50.0, 
        "rounds_per_episode": 5,   # 每轮 Episode 需要完成 5 次目标
        "obstacle_area_radius": 3, 
        "at_target_threshold": 0.3,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "sensing_radius": 3.0,
        "num_nearest_obstacles": 2,
        "obstacle_positions": obstacles,
        "obstacle_radius": 0.1,
        "obstacle_height": 2.5,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.1,
        "drone_safe_distance": 0.4,
        "drone_collision_distance": 0.1,
    }
    
    # === 观测维度计算 (CTDE) ===
    # 1. Self State (13维): RelPosBody(3) + LinVelBody(3) + AngVelBody(3) + LastAction(4)
    self_dim = 13
    
    # 2. Neighbor State (7维): RelPosBody(3) + RelVelBody(3) + Mask(1)
    neighbor_dim = 7 
    
    # 3. Obstacle State (6维): K=2 * 3 (Body Frame)
    obstacle_dim = env_cfg["num_nearest_obstacles"] * 3
    
    local_obs_dim = self_dim + neighbor_dim + obstacle_dim
    # Result: 13 + 7 + 6 = 26 维
    
    # 全局观测 (Critic): 拼接所有 Agent 的局部观测
    global_obs_dim = local_obs_dim * num_drones
    
    obs_cfg = {
        "num_obs": local_obs_dim, 
        "num_privileged_obs": global_obs_dim,
        "num_obs_per_drone": local_obs_dim,
        "obs_scales": {
            "rel_pos": 1 / 8.0, 
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
            "alive": 0,
            "smooth": -0.01,
            "yaw": 0.5,          
            "angular": -0.05,    
            "crash": -50.0,
            "obstacle": -5.0,
            "separation": -5.0,
            "team_coordination": 0.2,
        },
    }
    
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--update_config", action="store_true", default=False) 
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
    log_dir = f"logs/{args.exp_name}"
    
    new_env_cfg, new_obs_cfg, new_reward_cfg, new_command_cfg = get_cfgs()
    new_train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if args.resume:
        cfg_path = f"{log_dir}/cfgs.pkl"
        if not os.path.exists(cfg_path):
            print("Config not found.")
            return
        
        if args.update_config:
            print("Updating configuration from script...")
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = new_env_cfg, new_obs_cfg, new_reward_cfg, new_command_cfg, new_train_cfg
        else:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))

        if args.ckpt > 0:
            resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        else:
            files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if not files: return
            files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            resume_path = os.path.join(log_dir, files[-1])
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = new_env_cfg, new_obs_cfg, new_reward_cfg, new_command_cfg, new_train_cfg
        if os.path.exists(log_dir): shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))
        resume_path = None

    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 64)

    env = MultiDroneMAPPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )
    env.num_privileged_obs = obs_cfg["num_privileged_obs"] 

    algo_name = train_cfg["algorithm"]["class_name"]
    print("=" * 50)
    print(f"Algorithm: {algo_name}")
    print(f"Actor Dim: {env.num_obs}")
    print(f"Critic Dim: {env.num_privileged_obs}")
    print(f"Team Spirit: {env_cfg.get('team_spirit', 0.0)}")
    print("=" * 50)

    if algo_name == "MAPPO":
        runner = MAPPORunner(env, train_cfg, log_dir, device=gs.device)
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume and resume_path:
        runner.load(resume_path)
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# MAPPO多无人机训练命令

# 1. 新训练（无可视化，快速训练）
python multi_drone_mappo_train.py -e multi-drone-mappo-v4 -B 4096 --max_iterations 800

# 2. 新训练（带可视化）
python multi_drone_mappo_train.py -e multi-drone-mappo -B 64 --max_iterations 800 -v

# 3. 从最新检查点恢复训练（微调）
python multi_drone_mappo_train.py -e multi-drone-mappo -B 4096 --max_iterations 1000 --resume

# 4. 从指定检查点恢复训练（微调）
python multi_drone_mappo_train.py -e multi-drone-mappo -B 8192 --max_iterations 1000 --resume --ckpt 400

# 5. 微调训练（带可视化）
python multi_drone_mappo_train.py -e multi-drone-mappo -B 64 --max_iterations 1000 --resume -v

# 6. 微调（加载新配置）
python multi_drone_mappo_train.py -e multi-drone-mappo-v2 -B 6000 --max_iterations 2501 --resume --ckpt 1800 --update_config
"""