"""
多无人机路径规划评估脚本 - MAPPO版本
"""
import argparse
import os
import pickle
import torch
import sys

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"
    
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

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
    
    # 尝试读取 Config 中的维度（如果保存了的话）
    saved_config = checkpoint.get('config', {})
    ckpt_actor_dim = saved_config.get('num_actor_obs', obs_cfg["num_obs"])
    ckpt_critic_dim = saved_config.get('num_critic_obs', obs_cfg["num_privileged_obs"])
    
    print(f"Loading Model from: {model_path}")
    print(f"Model Iteration: {checkpoint.get('iteration', 'Unknown')}")
    
    # 使用检查点中的维度创建模型（确保兼容性）
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
    
    print(f"Start Evaluation: {args.episodes} episodes")
    
    with torch.no_grad():
        if args.record:
            video_path = f"video/mappo_{args.exp_name}.mp4"
            os.makedirs("video", exist_ok=True)
            env.cam.start_recording()
            print(f"Recording to {video_path} (Ctrl+C to stop)...")
            
            try:
                for ep in range(args.episodes):
                    obs, _ = env.reset()
                    for step in range(max_steps):
                        actions = actor_critic.act_inference(obs)
                        obs, rews, dones, infos = env.step(actions)
                        env.cam.render()
                        if dones.any():
                            print(f"Episode {ep+1} ended at step {step}")
                            break
            except KeyboardInterrupt:
                print("Interrupted.")
            finally:
                env.cam.stop_recording(save_to_filename=video_path, fps=60)
        else:
            for ep in range(args.episodes):
                obs, _ = env.reset()
                total_rew = 0
                for step in range(max_steps):
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    total_rew += rews.sum().item()
                    
                    if dones.any():
                        print(f"Ep {ep+1} Done. Reward: {total_rew:.2f}")
                        break

if __name__ == "__main__":
    main()

# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 800
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 799 --record
# python multi_drone_mappo_eval.py -e multi-drone-mappo --ckpt 1400 --target_threshold 0.2  # 使用更严格的判定半径