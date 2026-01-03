"""
单无人机持续导航评估脚本
"""
import argparse
import os
import pickle
import torch
import sys

import genesis as gs
from single_drone_ppo_env import SingleDronePPOEnv

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

from rsl_rl.modules import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-circle-nav")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=3) # 减少episode数，因为每个episode时间长
    parser.add_argument("--target_threshold", type=float, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    
    if args.target_threshold is not None:
        env_cfg["at_target_threshold"] = args.target_threshold

    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60
    env_cfg["episode_length_s"] = 60.0 # 评估时给更多时间看连续飞行

    env = SingleDronePPOEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    policy_cfg = train_cfg["policy"]
    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        init_noise_std=policy_cfg["init_noise_std"],
    ).to(gs.device)
    
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            print(f"Error: No model found in {log_dir}")
            return
        model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        model_path = os.path.join(log_dir, model_files[-1])
    
    checkpoint = torch.load(model_path, map_location=gs.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    print(f"Loaded model from {model_path}")

    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)
    
    print("\n" + "="*50)
    print("Continuous Navigation Evaluation")
    print(f"Episodes: {args.episodes} | Max Steps: {max_steps_per_episode}")
    print("="*50 + "\n")
    
    with torch.no_grad():
        if args.record:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"ppo_{args.exp_name}.mp4")
            
            print(f"Recording to {video_path}...")
            env.cam.start_recording()
            try:
                for ep in range(args.episodes):
                    obs, _ = env.reset()
                    curr_goals_hit = 0
                    for step in range(max_steps_per_episode):
                        actions = actor_critic.act_inference(obs)
                        obs, rews, dones, infos = env.step(actions)
                        env.cam.render()
                        
                        # 检查目标到达情况 (通过比较计数器)
                        total_hits = env.goals_reached_count[0].item()
                        if total_hits > curr_goals_hit:
                            print(f"Episode {ep+1} Step {step}: Goal {total_hits} Reached! Moving to next...")
                            curr_goals_hit = total_hits

                        if dones.any():
                            print(f"  Episode {ep+1}: Crash or Timeout at step {step}. Total Goals: {curr_goals_hit}")
                            break
            except KeyboardInterrupt:
                pass
            finally:
                env.cam.stop_recording(save_to_filename=video_path, fps=60)
                print(f"\nVideo saved to {video_path}")
        else:
            for ep in range(args.episodes):
                obs, _ = env.reset()
                curr_goals_hit = 0
                
                print(f"--- Episode {ep+1} Start ---")
                for step in range(max_steps_per_episode):
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    
                    total_hits = env.goals_reached_count[0].item()
                    if total_hits > curr_goals_hit:
                        dist_to_obs = env.min_obstacle_dist[0,0].item()
                        print(f"  [Step {step}] Goal {total_hits} Reached! (Nearest Obs: {dist_to_obs:.2f}m)")
                        curr_goals_hit = total_hits

                    if dones.any():
                        reason = "Crash" if env.crash_condition[0] else "Timeout"
                        print(f"  Episode {ep+1} End: {reason}. Total Goals Reached: {curr_goals_hit}\n")
                        break

if __name__ == "__main__":
    main()

# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 800
# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 799 --record
# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 1400 --target_threshold 0.2  # 使用更严格的判定半径