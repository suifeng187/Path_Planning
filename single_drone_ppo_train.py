import argparse
import os
import pickle
import shutil
import math
from importlib import metadata

# ... (Imports check logic remains the same) ...
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
from single_drone_ppo_env import SingleDronePPOEnv

# ... (get_train_cfg remains the same) ...
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

def generate_circular_obstacles(max_radius=2.5, obstacle_radius=0.12, spacing=0.9):
    """生成圆形障碍物阵列（原始训练配置）"""
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

def generate_grid_obstacles(arena_radius=3.5, obstacle_radius=0.1, grid_spacing=0.8, center_gap=1.0):
    """生成网格布局障碍物（用于泛化性验证）"""
    obstacles = []
    # 计算网格范围
    grid_range = arena_radius - center_gap
    num_cells = int(grid_range / grid_spacing)
    
    # 生成网格点
    for i in range(-num_cells, num_cells + 1):
        for j in range(-num_cells, num_cells + 1):
            x = i * grid_spacing
            y = j * grid_spacing
            
            # 跳过中心区域（给无人机起飞空间）
            dist_from_center = math.sqrt(x*x + y*y)
            if dist_from_center < center_gap:
                continue
            
            # 确保在arena范围内
            if dist_from_center <= arena_radius:
                obstacles.append([x, y, 1.0])
    
    return obstacles

def generate_random_obstacles(arena_radius=3.5, obstacle_radius=0.1, num_obstacles=30, center_gap=1.0, seed=None):
    """生成随机分布障碍物（用于泛化性验证）"""
    import random
    if seed is not None:
        random.seed(seed)
    
    obstacles = []
    attempts = 0
    max_attempts = num_obstacles * 10
    
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        # 在圆形区域内随机采样
        r = math.sqrt(random.random()) * arena_radius
        theta = random.random() * 2 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # 跳过中心区域
        if math.sqrt(x*x + y*y) < center_gap-0.8:
            continue
        
        # 检查是否与现有障碍物太近（确保障碍物中心距离大于0.6m）
        too_close = False
        min_spacing = 0.6  # 障碍物中心之间的最小距离（边缘到边缘约0.4m）
        for obs in obstacles:
            dist = math.sqrt((x - obs[0])**2 + (y - obs[1])**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if not too_close:
            obstacles.append([x, y, 1.0])
    
    return obstacles

def get_cfgs(obstacle_type="circular"):
    """
    获取环境配置
    Args:
        obstacle_type: 障碍物类型
            - "circular": 圆形阵列（原始训练配置）
            - "grid": 网格布局（泛化性验证）
            - "random": 随机分布（泛化性验证）
    """
    num_drones = 1
    
    arena_radius = 3.5
    
    # 根据类型生成不同的障碍物配置
    if obstacle_type == "circular":
        obstacle_pos_list = generate_circular_obstacles(
            max_radius=arena_radius, 
            obstacle_radius=0.1, 
            spacing=0.8
        )
    elif obstacle_type == "grid":
        obstacle_pos_list = generate_grid_obstacles(
            arena_radius=arena_radius,
            obstacle_radius=0.1,
            grid_spacing=0.8,
            center_gap=1.0
        )
    elif obstacle_type == "random":
        obstacle_pos_list = generate_random_obstacles(
            arena_radius=arena_radius,
            obstacle_radius=0.1,
            num_obstacles=30,
            center_gap=1.0,
            seed=42  # 固定种子以确保可重复性
        )
    else:
        raise ValueError(f"Unknown obstacle_type: {obstacle_type}. Choose from 'circular', 'grid', 'random'")

    env_cfg = {
        "num_drones": num_drones,
        "num_actions": 4,
        "termination_if_roll_greater_than": 60,
        "termination_if_pitch_greater_than": 60,
        "termination_if_close_to_ground": 0.02,
        
        "drone_init_positions": [[0.0, 0.0, 0.8]], 
        "drone_goal_positions": [[2.0, 0.0, 0.8]], 

        # === 修改 2: 目标生成参数调整 ===
        "arena_radius": arena_radius, # 传入这个参数，让目标生成在这个半径内
        "goal_height": 0.8,
        
        # 悬停参数
        "hover_duration_s": 0.5, # 悬停时间

        "episode_length_s": 45.0, # 增加一点时间，因为要悬停
        "at_target_threshold": 0.4,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        
        "sensing_radius": 3.0,          
        "num_nearest_obstacles": 2,     

        "obstacle_positions": obstacle_pos_list,
        "obstacle_radius": 0.1,
        "obstacle_height": 1.6,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.1,
    }
    
    obs_per_drone = 17 + (env_cfg["num_nearest_obstacles"] * 3)
    
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
            "target": 20.0,
            "progress": 1.5,     
            "alive": 0.1,
            "smooth": -1e-3,
            "yaw": 0.5,
            "angular": -0.05,
            "crash": -20.0,
            "obstacle": -5.0,
            # 注意：成功到达的奖励会在 env.step 中处理
        },
    }
    
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg

# ... (main function remains the same) ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-hover-nav") # 建议改个名
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500) # 任务变难了，增加训练步数
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--update_config", action="store_true", default=False)
    parser.add_argument("--obstacle_type", type=str, default="circular", 
                       choices=["circular", "grid", "random"],
                       help="障碍物类型: circular(圆形阵列), grid(网格布局), random(随机分布)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    
    if args.resume:
        if not os.path.exists(log_dir):
            print(f"错误：日志目录 {log_dir} 不存在，无法恢复训练")
            return
        
        cfg_path = f"{log_dir}/cfgs.pkl"
        
        if args.update_config:
            print(f"[更新配置] 使用新的配置参数，障碍物类型: {args.obstacle_type}")
            env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(obstacle_type=args.obstacle_type)
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
        print(f"[新训练] 障碍物类型: {args.obstacle_type}")
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(obstacle_type=args.obstacle_type)
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

    env = SingleDronePPOEnv(
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
# PPO单无人机训练命令

# 1. 新训练（无可视化，快速训练）- 使用圆形障碍物（默认）
python single_drone_ppo_train.py -e single-drone-ppo-v2 -B 4096 --max_iterations 800

# 2. 新训练（使用网格布局障碍物）
python single_drone_ppo_train.py -e single-drone-ppo-grid -B 4096 --max_iterations 800 --obstacle_type grid

# 3. 新训练（使用随机分布障碍物）
python single_drone_ppo_train.py -e single-drone-ppo-random -B 4096 --max_iterations 800 --obstacle_type random

# 4. 新训练（带可视化）
python single_drone_ppo_train.py -e single-drone-ppo -B 64 --max_iterations 800 -v

# 5. 从最新检查点恢复训练（微调）
python single_drone_ppo_train.py -e single-drone-ppo -B 4096 --max_iterations 1000 --resume

# 6. 从指定检查点恢复训练（微调）
python single_drone_ppo_train.py -e single-drone-ppo -B 8192 --max_iterations 1000 --resume --ckpt 400

# 7. 微调训练（带可视化）
python single_drone_ppo_train.py -e single-drone-ppo -B 64 --max_iterations 1000 --resume -v

# 8. 微调（加载新障碍物配置）
python single_drone_ppo_train.py -e single-drone-ppo -B 8192 --max_iterations 1000 --resume --ckpt 400 --update_config --obstacle_type grid
"""