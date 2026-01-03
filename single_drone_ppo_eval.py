"""
单无人机避障路径规划评估脚本 - PPO版本
"""
import argparse
import os
import pickle
import torch
import sys

import genesis as gs
# 修改：引入单无人机环境
from single_drone_ppo_env import SingleDronePPOEnv

# 检查rsl-rl-lib版本
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
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-ppo")
    parser.add_argument("--ckpt", type=int, default=-1, help="检查点迭代数，-1表示加载最新模型")
    parser.add_argument("--record", action="store_true", default=False, help="录制视频")
    parser.add_argument("--episodes", type=int, default=5, help="评估的episode数量")
    parser.add_argument("--target_threshold", type=float, default=None, 
                        help="评估时使用的目标点判定半径(米)，默认使用训练时的配置值")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # 加载配置
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    
    original_threshold = env_cfg.get("at_target_threshold", 0.5)
    if args.target_threshold is not None:
        env_cfg["at_target_threshold"] = args.target_threshold
        print(f"评估模式：目标点判定半径从 {original_threshold} 米调整为 {args.target_threshold} 米")
    else:
        print(f"使用训练时的目标点判定半径: {original_threshold} 米")

    # 可视化配置
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    # 实例化单无人机环境
    env = SingleDronePPOEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # 创建ActorCritic模型
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
    
    # 确定模型路径
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        # 查找最新的模型
        model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            print(f"Error: No model found in {log_dir}")
            return
        model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        model_path = os.path.join(log_dir, model_files[-1])
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        available = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if available:
            print(f"Available models: {available}")
        return
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=gs.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    print(f"Loaded model from {model_path}")

    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)
    
    print("\n" + "="*50)
    print("PPO Single Drone Evaluation")
    print(f"Episodes: {args.episodes} | Max Steps: {max_steps_per_episode}")
    print("="*50 + "\n")
    
    with torch.no_grad():
        if args.record:
            # 录制模式
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"ppo_{args.exp_name}.mp4")
            
            print(f"Recording to {video_path}...")
            print("提示：录制过程中，您可以随时在终端按 'Ctrl + C' 结束录制并保存视频。")

            env.cam.start_recording()
            
            try:
                for ep in range(args.episodes):
                    obs, _ = env.reset()
                    
                    for step in range(max_steps_per_episode):
                        actions = actor_critic.act_inference(obs)
                        obs, rews, dones, infos = env.step(actions)
                        env.cam.render()
                        
                        if dones.any():
                            print(f"  Episode {ep+1}: Done at step {step}")
                            break
            except KeyboardInterrupt:
                print("\n\n[用户中断] 检测到 Ctrl+C，正在停止录制并保存视频...")
            finally:
                env.cam.stop_recording(save_to_filename=video_path, fps=60)
                print(f"\nVideo saved to {video_path}")
        
        else:
            # 交互式评估 (无录制)
            success_count = 0
            
            for ep in range(args.episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for step in range(max_steps_per_episode):
                    actions = actor_critic.act_inference(obs)
                    
                    # 在调用step之前保存状态
                    prev_reached = env.drone_ever_reached_target.clone() if hasattr(env, 'drone_ever_reached_target') else None
                    
                    obs, rews, dones, infos = env.step(actions)
                    episode_reward += rews.sum().item()
                    
                    if dones.any():
                        # 单无人机数据获取
                        final_dist = torch.norm(env.rel_pos[0, 0, :]).item()
                        reached = prev_reached[0, 0].item() if prev_reached is not None else False
                        
                        crash_now = env.crash_condition.clone() if hasattr(env, 'crash_condition') else torch.tensor([False])
                        success_now = env.success_condition.clone() if hasattr(env, 'success_condition') else torch.tensor([False])
                        
                        if success_now.item():
                            reached = True
                        
                        if reached:
                            success_count += 1
                            print(f"Episode {ep+1}: SUCCESS (step {step}, reward {episode_reward:.2f})")
                            print(f"  最终距离: {final_dist:.3f}m")
                        else:
                            print(f"Episode {ep+1}: FAILED (step {step}, reward {episode_reward:.2f})")
                            print(f"  最终距离: {final_dist:.3f}m")
                            print(f"  结束原因: Crash={crash_now.any().item()}, Success={success_now.any().item()}")
                            
                            min_obs_dist = env.min_obstacle_dist[0, 0].item()
                            print(f"  最小障碍物距离: {min_obs_dist:.3f}m")
                        break
                else:
                    # Timeout情况
                    final_dist = torch.norm(env.rel_pos[0, 0, :]).item()
                    reached = env.drone_ever_reached_target[0, 0].item()
                    print(f"Episode {ep+1}: TIMEOUT (reward {episode_reward:.2f})")
                    print(f"  最终距离: {final_dist:.3f}m")
                    print(f"  到达状态: {reached}")
            
            print(f"\nCompleted: {args.episodes} episodes | Success Rate: {success_count}/{args.episodes}")

if __name__ == "__main__":
    main()

# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 800
# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 799 --record
# python single_drone_ppo_eval.py -e single-drone-ppo --ckpt 1400 --target_threshold 0.2  # 使用更严格的判定半径