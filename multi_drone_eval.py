"""
多无人机避障路径规划评估脚本 - MAPPO版本
支持可视化评估和视频录制
"""
import argparse
import os
import pickle
import torch

import genesis as gs
from multi_drone_env import MultiDroneEnv
from mappo_algorithm import MAPPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("--ckpt", type=int, default=-1, help="检查点迭代数，-1表示加载final模型")
    parser.add_argument("--record", action="store_true", default=False, help="录制视频")
    parser.add_argument("--episodes", type=int, default=5, help="评估的episode数量")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    
    # 加载配置
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, mappo_cfg = pickle.load(open(cfg_path, "rb"))
    
    # 评估时不计算奖励
    reward_cfg["reward_scales"] = {}

    # 可视化配置
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    env = MultiDroneEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # 加载MAPPO模型
    mappo = MAPPO(
        num_agents=env.num_drones,
        obs_dim=env.num_obs_per_drone,
        global_obs_dim=env.global_obs_dim,
        action_dim=env.num_actions_per_drone,
        device=gs.device,
        **mappo_cfg
    )
    
    # 确定模型路径
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"mappo_model_{args.ckpt}.pt")
    else:
        model_path = os.path.join(log_dir, "mappo_model_final.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # 尝试查找可用模型
        available = [f for f in os.listdir(log_dir) if f.startswith("mappo_model")]
        if available:
            print(f"Available models: {available}")
        return
    
    mappo.load(model_path)
    print(f"Loaded model from {model_path}")

    # 初始化观测
    obs, _ = env.reset()
    _init_observations(env)

    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)
    
    print("\n" + "="*50)
    print("MAPPO Multi-Drone Evaluation")
    print(f"Episodes: {args.episodes} | Max Steps: {max_steps_per_episode}")
    print("="*50 + "\n")
    
    with torch.no_grad():
        if args.record:
            # 录制模式
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"mappo_{args.exp_name}.mp4")
            
            print(f"Recording to {video_path}...")
            env.cam.start_recording()
            
            for ep in range(args.episodes):
                obs, _ = env.reset()
                _init_observations(env)
                
                for step in range(max_steps_per_episode):
                    local_obs = env.get_local_obs()
                    actions, _ = mappo.get_actions(local_obs, deterministic=True)
                    flat_actions = actions.view(env.num_envs, -1)
                    obs, rews, dones, infos = env.step(flat_actions)
                    env.cam.render()
                    
                    if dones.any():
                        if env.success_condition.any():
                            print(f"  Episode {ep+1}: SUCCESS at step {step}")
                        else:
                            print(f"  Episode {ep+1}: Failed at step {step}")
                        break
            
            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"\nVideo saved to {video_path}")
        
        else:
            # 交互式评估
            success_count = 0
            
            for ep in range(args.episodes):
                obs, _ = env.reset()
                _init_observations(env)
                episode_reward = 0
                
                for step in range(max_steps_per_episode):
                    local_obs = env.get_local_obs()
                    actions, _ = mappo.get_actions(local_obs, deterministic=True)
                    flat_actions = actions.view(env.num_envs, -1)
                    obs, rews, dones, infos = env.step(flat_actions)
                    episode_reward += rews.sum().item()
                    
                    if dones.any():
                        if env.success_condition.any():
                            success_count += 1
                            print(f"Episode {ep+1}: SUCCESS (step {step}, reward {episode_reward:.2f})")
                        else:
                            print(f"Episode {ep+1}: Failed (step {step}, reward {episode_reward:.2f})")
                        break
                else:
                    print(f"Episode {ep+1}: Timeout (reward {episode_reward:.2f})")
            
            print(f"\nSuccess Rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")


def _init_observations(env):
    """初始化观测（reset后调用）"""
    for i in range(env.num_drones):
        drone_obs = torch.cat([
            torch.clip(env.rel_pos[:, i, :] * env.obs_scales["rel_pos"], -1, 1),
            env.base_quat[:, i, :],
            torch.clip(env.base_lin_vel[:, i, :] * env.obs_scales["lin_vel"], -1, 1),
            torch.clip(env.base_ang_vel[:, i, :] * env.obs_scales["ang_vel"], -1, 1),
            env.last_actions[:, i*4:(i+1)*4],
        ], dim=-1)
        env.local_obs[:, i, :] = drone_obs
    
    obs_buf = env.local_obs.view(env.num_envs, -1)
    obstacle_info = []
    for obs in env.obstacles:
        obstacle_info.append(obs["pos"][:2].unsqueeze(0).expand(env.num_envs, -1))
    if obstacle_info:
        obstacle_tensor = torch.cat(obstacle_info, dim=-1)
        env.global_obs = torch.cat([obs_buf, obstacle_tensor], dim=-1)
    else:
        env.global_obs = obs_buf


if __name__ == "__main__":
    main()

"""
# MAPPO评估命令

# 基本评估（可视化窗口）
python multi_drone_eval.py -e multi-drone-mappo

# 指定检查点
python multi_drone_eval.py -e multi-drone-mappo --ckpt 400

# 录制视频
python multi_drone_eval.py -e multi-drone-mappo --record --episodes 3
"""
