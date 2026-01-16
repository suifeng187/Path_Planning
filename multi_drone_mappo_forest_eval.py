"""
multi_drone_mappo_forest_eval.py
森林穿越评估脚本 - MAPPO版本
配置更新：
1. 地图尺寸：12m x 5m
2. 观测缩放：rel_pos 改为 1/14.0 以适应更远的航程
"""
import argparse
import os
import pickle
import torch
import sys
import math
import random
import numpy as np

import genesis as gs
from multi_drone_mappo_forest_env import MultiDroneMAPPOEnv

from importlib import metadata
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from mappo_algorithm import ActorCritic

def generate_forest_obstacles(num_obs, x_range, y_range, min_spacing):
    """
    在矩形森林区域内随机生成障碍物
    """
    obstacles = []
    attempts = 0
    max_attempts = num_obs * 5000
    
    print(f"[Forest Map] Generating {num_obs} obstacles in X:{x_range} Y:{y_range}...")
    
    while len(obstacles) < num_obs and attempts < max_attempts:
        attempts += 1
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        
        valid = True
        for ox, oy, _ in obstacles:
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < min_spacing:
                valid = False
                break
        
        if valid:
            obstacles.append([x, y, 1.0])
            
    print(f"[Forest Map] Successfully generated {len(obstacles)} obstacles.")
    return obstacles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num_obstacles", type=int, default=60) 
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"
    
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    # 加载训练时的原始配置
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    
    # ==================== [配置修改区域] ====================
    # 1. 修改观测缩放 (关键修正)
    # 因为现在的跨度是 -7m 到 +7m = 14m，原先的 1/8.0 会导致数值溢出被截断
    print(f"[Config] Overriding obs_scales['rel_pos'] from {obs_cfg['obs_scales']['rel_pos']} to {1/14.0}")
    obs_cfg["obs_scales"]["rel_pos"] = 1.0 / 14.0

    # 2. 设置只运行一轮
    env_cfg["rounds_per_episode"] = 1
    
    # 3. 森林尺寸定义 (穿越方向为X轴，长度设为12米)
    forest_len_x = 12.0  
    forest_wid_y = 5.0   
    
    x_min, x_max = -forest_len_x/2.0, forest_len_x/2.0  # [-6.0, 6.0]
    y_min, y_max = -forest_wid_y/2.0, forest_wid_y/2.0  # [-2.5, 2.5]
    
    # 4. 生成障碍物
    forest_obstacles = generate_forest_obstacles(
        num_obs=args.num_obstacles,
        x_range=[x_min, x_max], 
        y_range=[y_min, y_max], 
        min_spacing=1
    )
    env_cfg["obstacle_positions"] = forest_obstacles
    env_cfg["obstacle_radius"] = 0.1
    env_cfg["obstacle_height"] = 5
    
    # 5. 设置固定起点 (X = -7.0)
    start_x = x_min - 1.0 
    start_z = 1.0
    env_cfg["drone_init_positions"] = [
        [start_x, -1.0, start_z],
        [start_x,  0.0, start_z],
        [start_x,  1.0, start_z]
    ]
    
    # 6. 设置固定终点 (X = +7.0)
    target_x = x_max + 1.0
    target_z = 1.0 
    env_cfg["fixed_target_positions"] = [
        [target_x, -1.0, target_z],
        [target_x,  0.0, target_z],
        [target_x,  1.0, target_z]
    ]

    # 可视化设置
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60
    # ==================== [配置结束] ====================

    # 初始化环境 (传入修改后的配置)
    env = MultiDroneMAPPOEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,  # 这里传入了修改后的 obs_cfg
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    policy_cfg = train_cfg["policy"]
    
    # 加载模型权重
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            print("No model found.")
            return
        model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        model_path = os.path.join(log_dir, model_files[-1])
    
    checkpoint = torch.load(model_path, map_location=gs.device)
    saved_config = checkpoint.get('config', {})
    
    # 兼容性处理：优先使用 checkpoint 中的维度，如果没有则使用 obs_cfg 中的
    ckpt_actor_dim = saved_config.get('num_actor_obs', obs_cfg["num_obs"])
    ckpt_critic_dim = saved_config.get('num_critic_obs', obs_cfg["num_privileged_obs"])
    
    print(f"Loading Model from: {model_path}")
    
    actor_critic = ActorCritic(
        num_actor_obs=ckpt_actor_dim,
        num_critic_obs=ckpt_critic_dim,
        num_actions=env.num_actions,
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        init_noise_std=policy_cfg["init_noise_std"],
    ).to(gs.device)
    
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()

    # 设置最大时间步 (路程14米 / 速度0.5m/s ≈ 28s -> 设为30s)
    max_steps = int(30.0 / env.dt) 
    
    stats = {
        "success": 0,
        "collision": 0,
        "timeout": 0
    }
    
    print(f"Start Forest Evaluation: {args.episodes} episodes")
    print(f"Scenario: Forest Size 12m(L) x 5m(W)")
    print(f"Task: Fly from X={start_x} to X={target_x} (Dist: {target_x - start_x}m)")

    if args.record:
        video_path = f"video/forest_crossing_12m_{args.exp_name}.mp4"
        os.makedirs("video", exist_ok=True)
        if env.cam:
            env.cam.start_recording()

    try:
        with torch.no_grad():
            for ep in range(args.episodes):
                obs, _ = env.reset()
                total_rew = 0
                outcome = "Unknown"
                
                for step in range(max_steps):
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    
                    total_rew += rews.sum().item()
                    if env.cam and args.record: env.cam.render()
                    
                    if dones.any():
                        is_crash = env.phys_crash_cond[0].item()
                        is_timeout = infos["time_outs"][0].item()
                        is_success = env.phys_success_cond[0].item()

                        if is_crash:
                            stats["collision"] += 1
                            outcome = "CRASH"
                        elif is_success:
                            stats["success"] += 1
                            outcome = "SUCCESS"
                        elif is_timeout:
                            stats["timeout"] += 1
                            outcome = "TIMEOUT"
                        else:
                            if env.success_rounds[0] >= 1:
                                stats["success"] += 1
                                outcome = "SUCCESS"
                            else:
                                stats["timeout"] += 1
                                outcome = "TIMEOUT"

                        print(f"Episode {ep+1} | Steps: {step} | Result: {outcome}")
                        break
                else:
                    stats["timeout"] += 1
                    print(f"Episode {ep+1} | Result: TIMEOUT (Max Steps)")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.record and env.cam:
            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"Video saved to {video_path}")

    print("\n" + "="*30)
    print("   FOREST CROSSING REPORT    ")
    print("="*30)
    print(f"Success Rate   : {stats['success']/args.episodes*100:.1f}%")
    print(f"Collision Rate : {stats['collision']/args.episodes*100:.1f}%")
    print("="*30)

if __name__ == "__main__":
    main()

# python multi_drone_mappo_forest_eval.py -e multi-drone-mappo --ckpt 800
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 799 --record
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 1400 --target_threshold 0.2  # 使用更严格的判定半径
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --random_obs

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --record --random_obs

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --random_obs --episodes 100