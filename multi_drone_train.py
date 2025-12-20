"""
多无人机避障路径规划训练脚本 - MAPPO版本
支持可视化训练过程，类似PPO训练体验
"""
import argparse
import os
import pickle
import shutil
import torch
import time
from collections import deque

import genesis as gs
from multi_drone_env import MultiDroneEnv
from mappo_algorithm import MAPPO, MAPPOBuffer


def get_cfgs():
    """环境配置"""
    num_drones = 3
    
    env_cfg = {
        "num_drones": num_drones,
        "num_actions": 4,
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
            [-0.5, -1.5, 1.0], [0.5, -1.5, 1.0],
            [-1.0, -0.5, 1.0], [0.0, -0.5, 1.0], [1.0, -0.5, 1.0],
            [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0],
            [-1.0, 1.5, 1.0], [0.0, 1.5, 1.0], [1.0, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,
        "obstacle_height": 2.0,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.12,
        "drone_safe_distance": 0.35,
        "drone_collision_distance": 0.2,
    }
    
    obs_cfg = {
        "num_obs_per_drone": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    
    # 奖励配置（密集引导信号 + 存活奖励）
    reward_cfg = {
        "reward_scales": {
            "target": 50.0,       # 目标奖励（含密集距离信号）
            "progress": 30.0,     # 前进+高度+姿态奖励
            "alive": 5.0,         # 存活奖励（新增）
            "smooth": -1e-6,      # 极小平滑惩罚
            "crash": -5.0,        # 降低坠机惩罚
            "obstacle": -1.0,     # 降低避障惩罚
            "separation": -0.5,   # 降低间距惩罚
        },
    }
    
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_mappo_cfg():
    """MAPPO算法配置 - 大规模并行优化版"""
    return {
        "lr_actor": 5e-4,       # 更高学习率（大batch可以用更高lr）
        "lr_critic": 1e-3,      # critic学更快
        "gamma": 0.99,          # 折扣因子
        "lam": 0.95,            # GAE lambda
        "clip_param": 0.2,      # PPO裁剪参数
        "entropy_coef": 0.01,   # 适中熵系数
        "value_loss_coef": 0.5,
        "max_grad_norm": 1.0,
        "num_epochs": 4,        # 减少epoch（大batch不需要太多epoch）
        "batch_size": 2048,     # 增大batch size
        "share_actor": True,
    }


class MAPPORunner:
    """MAPPO训练Runner - 支持可视化"""
    
    def __init__(self, env, mappo_cfg, log_dir, device, num_steps_per_env=100):
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.num_steps_per_env = num_steps_per_env
        
        num_agents = env.num_drones
        obs_dim = env.num_obs_per_drone
        global_obs_dim = env.global_obs_dim
        action_dim = env.num_actions_per_drone
        
        self.mappo = MAPPO(
            num_agents=num_agents,
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=action_dim,
            device=device,
            **mappo_cfg
        )
        
        self.buffer = MAPPOBuffer(
            num_envs=env.num_envs,
            num_agents=num_agents,
            num_steps=num_steps_per_env,
            obs_dim=obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=action_dim,
            device=device
        )
        
        # 训练统计
        self.reward_history = deque(maxlen=100)
        self.success_history = deque(maxlen=100)
        self.total_timesteps = 0
        self.start_time = None
    
    def learn(self, max_iterations, save_interval=100, log_interval=1):
        """训练主循环 - 支持可视化"""
        self.start_time = time.time()
        obs_buf, _ = self.env.reset()
        
        # 初始化局部观测（reset后需要构建）
        self._update_observations()
        
        print("\n" + "="*60)
        print("MAPPO Multi-Drone Training Started")
        print(f"Num Envs: {self.env.num_envs} | Num Drones: {self.env.num_drones}")
        print(f"Obs Dim (per drone): {self.env.num_obs_per_drone} | Global Obs Dim: {self.env.global_obs_dim}")
        print("="*60 + "\n")
        
        for iteration in range(max_iterations):
            iter_start = time.time()
            episode_rewards = []
            episode_successes = []
            
            # 收集经验
            for step in range(self.num_steps_per_env):
                local_obs = self.env.get_local_obs()
                global_obs = self.env.get_global_obs()
                
                with torch.no_grad():
                    actions, log_probs = self.mappo.get_actions(local_obs)
                    values = self.mappo.get_value(global_obs)
                
                flat_actions = actions.view(self.env.num_envs, -1)
                obs_buf, rewards, dones, infos = self.env.step(flat_actions)
                
                self.buffer.store(
                    obs=local_obs,
                    global_obs=global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones.float(),
                    values=values
                )
                
                episode_rewards.append(rewards.mean().item())
                
                # 统计成功率
                if "episode" in infos:
                    episode_successes.append(self.env.success_condition.float().mean().item())
                
                self.total_timesteps += self.env.num_envs
            
            # 更新网络 - 使用最后收集的global_obs计算bootstrap value
            with torch.no_grad():
                # 获取最新的全局观测用于计算最后的value
                last_global_obs = self.env.get_global_obs().clone()
                last_value = self.mappo.get_value(last_global_obs)
            
            # 计算GAE并更新
            self.buffer.compute_gae(last_value.detach(), self.mappo.gamma, self.mappo.lam)
            losses = self.mappo.update(self.buffer)
            
            # 统计
            mean_reward = sum(episode_rewards) / len(episode_rewards)
            self.reward_history.append(mean_reward)
            
            if episode_successes:
                mean_success = sum(episode_successes) / len(episode_successes)
                self.success_history.append(mean_success)
            
            iter_time = time.time() - iter_start
            fps = (self.num_steps_per_env * self.env.num_envs) / iter_time
            
            # 日志输出
            if (iteration + 1) % log_interval == 0:
                elapsed = time.time() - self.start_time
                avg_reward = sum(self.reward_history) / len(self.reward_history)
                avg_success = sum(self.success_history) / len(self.success_history) if self.success_history else 0
                
                print(f"Iter {iteration+1:4d}/{max_iterations} | "
                      f"Reward: {mean_reward:7.3f} (avg: {avg_reward:7.3f}) | "
                      f"Success: {avg_success*100:5.1f}% | "
                      f"FPS: {fps:6.0f} | "
                      f"Time: {elapsed/60:5.1f}min")
                
                if (iteration + 1) % 10 == 0:
                    print(f"  └─ Actor Loss: {losses['actor_loss']:.4f} | "
                          f"Critic Loss: {losses['critic_loss']:.4f} | "
                          f"Entropy: {losses['entropy']:.4f}")
            
            # 保存模型
            if (iteration + 1) % save_interval == 0:
                self.save(os.path.join(self.log_dir, f"mappo_model_{iteration+1}.pt"))
        
        # 最终保存
        self.save(os.path.join(self.log_dir, "mappo_model_final.pt"))
        
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print(f"Training Complete! Total Time: {total_time/60:.1f} min")
        print(f"Final Avg Reward: {sum(self.reward_history)/len(self.reward_history):.3f}")
        print("="*60)
    
    def _update_observations(self):
        """更新观测（用于reset后初始化）"""
        for i in range(self.env.num_drones):
            drone_obs = torch.cat([
                torch.clip(self.env.rel_pos[:, i, :] * self.env.obs_scales["rel_pos"], -1, 1),
                self.env.base_quat[:, i, :],
                torch.clip(self.env.base_lin_vel[:, i, :] * self.env.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.env.base_ang_vel[:, i, :] * self.env.obs_scales["ang_vel"], -1, 1),
                self.env.last_actions[:, i*4:(i+1)*4],
            ], dim=-1)
            self.env.local_obs[:, i, :] = drone_obs
        
        obs_buf = self.env.local_obs.view(self.env.num_envs, -1)
        obstacle_info = []
        for obs in self.env.obstacles:
            obstacle_info.append(obs["pos"][:2].unsqueeze(0).expand(self.env.num_envs, -1))
        if obstacle_info:
            obstacle_tensor = torch.cat(obstacle_info, dim=-1)
            self.env.global_obs = torch.cat([obs_buf, obstacle_tensor], dim=-1)
        else:
            self.env.global_obs = obs_buf
    
    def save(self, path):
        self.mappo.save(path)
        print(f"  [Saved] {path}")
    
    def load(self, path):
        self.mappo.load(path)
        print(f"  [Loaded] {path}")
    
    def get_inference_policy(self):
        """获取推理策略（用于评估）"""
        return self.mappo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="启用可视化")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=800)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100, help="每次迭代的步数")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    mappo_cfg = get_mappo_cfg()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 可视化配置
    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)  # 可视化时适当减少环境数量
        print(f"[Visualization Mode] Reduced num_envs to {args.num_envs}")
    else:
        print(f"[Training Mode] Using {args.num_envs} parallel environments")

    # 保存配置
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, mappo_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = MultiDroneEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,  # 关键：启用可视化窗口
    )

    runner = MAPPORunner(
        env, 
        mappo_cfg, 
        log_dir, 
        device=gs.device,
        num_steps_per_env=args.num_steps
    )
    runner.learn(
        max_iterations=args.max_iterations, 
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()

"""
# MAPPO训练命令

# 无可视化（快速训练）
python multi_drone_train.py -e multi-drone-mappo -B 4096 --max_iterations 800

# 带可视化（观察训练过程）
python multi_drone_train.py -e multi-drone-mappo -B 64 --max_iterations 800 -v

# 自定义参数
python multi_drone_train.py -e my-exp -B 2048 --max_iterations 500 --save_interval 50 --num_steps 128
"""
