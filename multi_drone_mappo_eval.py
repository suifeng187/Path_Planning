"""
多无人机路径规划评估脚本 - MAPPO版本 (含成功率统计与随机障碍物泛化测试)
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
from multi_drone_mappo_env import MultiDroneMAPPOEnv

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

# 使用本地定义的 Network 类以确保兼容
from mappo_algorithm import ActorCritic

def generate_random_obstacles(num_obs, area_radius, min_spacing):
    """
    在圆形区域内随机生成互不重叠的障碍物 (拒绝采样)
    """
    obstacles = []
    attempts = 0
    max_attempts = num_obs * 1000  # 防止死循环
    
    while len(obstacles) < num_obs and attempts < max_attempts:
        attempts += 1
        
        # 1. 在圆内均匀随机采样
        r = math.sqrt(random.random()) * area_radius
        theta = random.random() * 2 * math.pi
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # 2. 检查与现有障碍物的距离
        valid = True
        for ox, oy, _ in obstacles:
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < min_spacing:
                valid = False
                break
        
        # 3. 如果位置有效，添加列表
        if valid:
            obstacles.append([x, y, 1.0])
            
    print(f"[Random Map] Generated {len(obstacles)}/{num_obs} obstacles with spacing > {min_spacing}m")
    return obstacles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=10) # 默认评估 10 轮
    parser.add_argument("--random_obs", action="store_true", default=False, help="Generate random obstacles for generalization test")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"
    
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    
    # ==================== [配置修改] ====================
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60
    
    # 随机障碍物生成逻辑
    if args.random_obs:
        print("="*50)
        print("Generalization Test: Generating Random Obstacles...")
        random_obstacles = generate_random_obstacles(num_obs=30, area_radius=3.0, min_spacing=0.8)
        env_cfg["obstacle_positions"] = random_obstacles
        env_cfg["obstacle_radius"] = 0.1
    # ==================== [修改结束] ====================

    # 评估只需一个物理环境
    env = MultiDroneMAPPOEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    policy_cfg = train_cfg["policy"]
    
    # 加载模型
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
    ckpt_actor_dim = saved_config.get('num_actor_obs', obs_cfg["num_obs"])
    ckpt_critic_dim = saved_config.get('num_critic_obs', obs_cfg["num_privileged_obs"])
    
    print(f"Loading Model from: {model_path}")
    print(f"Model Iteration: {checkpoint.get('iteration', 'Unknown')}")
    
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

    max_steps = int(env_cfg["episode_length_s"] / env.dt)
    
    # ==================== [统计变量初始化] ====================
    stats = {
        "success": 0,
        "collision": 0,
        "timeout": 0
    }
    total_episodes = args.episodes
    # ========================================================

    print(f"Start Evaluation: {args.episodes} episodes")
    
    if args.record:
        video_path = f"video/mappo_{args.exp_name}_{'random' if args.random_obs else 'fixed'}.mp4"
        os.makedirs("video", exist_ok=True)
        if env.cam:
            env.cam.start_recording()
            print(f"Recording to {video_path}...")

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
                    
                    # 检查是否结束
                    if dones.any():
                        # === 判定结束原因 ===
                        # 1. 检查是否碰撞 (phys_crash_cond)
                        # 注意：env.phys_crash_cond 是 tensor，需要取第0个环境的值
                        is_crash = env.phys_crash_cond[0].item()
                        
                        # 2. 检查是否超时 (time_outs)
                        # infos["time_outs"] 是 tensor
                        is_timeout = infos["time_outs"][0].item()
                        
                        if is_crash:
                            stats["collision"] += 1
                            outcome = "COLLISION"
                        elif is_timeout:
                            stats["timeout"] += 1
                            outcome = "TIMEOUT"
                        else:
                            # 既没碰撞也没超时，但 done 了，说明完成了所有 Round -> 成功
                            stats["success"] += 1
                            outcome = "SUCCESS"
                            
                        print(f"Episode {ep+1}/{total_episodes} | Steps: {step} | Reward: {total_rew:.2f} | Result: {outcome}")
                        break
                
                # 如果跑满 max_steps 还没 done (极少数情况)，视为超时
                else:
                    stats["timeout"] += 1
                    print(f"Episode {ep+1}/{total_episodes} | Steps: {max_steps} | Reward: {total_rew:.2f} | Result: TIMEOUT (Max Steps)")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        if args.record and env.cam:
            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"Video saved to {video_path}")

    # ==================== [输出最终统计报告] ====================
    print("\n" + "="*30)
    print("      EVALUATION REPORT      ")
    print("="*30)
    print(f"Total Episodes : {args.episodes}")
    print(f"Successes      : {stats['success']} ({stats['success']/args.episodes*100:.1f}%)")
    print(f"Collisions     : {stats['collision']} ({stats['collision']/args.episodes*100:.1f}%)")
    print(f"Timeouts       : {stats['timeout']} ({stats['timeout']/args.episodes*100:.1f}%)")
    print("="*30)

if __name__ == "__main__":
    main()

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 799 --record
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 1400 --target_threshold 0.2  # 使用更严格的判定半径
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --random_obs

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --record --random_obs

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800 --random_obs --episodes 100