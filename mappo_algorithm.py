import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import deque
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gc

class ActorCritic(nn.Module):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 128, 64], 
                 critic_hidden_dims=[512, 256, 128],
                 activation="elu", init_noise_std=1.0):
        super().__init__()

        act_func = nn.ELU() if activation.lower() == "elu" else nn.ReLU()

        # Actor 网络
        actor_layers = []
        prev_dim = num_actor_obs
        for dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, dim))
            actor_layers.append(act_func)
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # 初始化 Actor 最后一层，使其输出接近 0
        last_layer = self.actor[-1]
        nn.init.orthogonal_(last_layer.weight, 0.01)
        nn.init.constant_(last_layer.bias, 0.0)
        
        self.log_std = nn.Parameter(np.log(init_noise_std) * torch.ones(num_actions))

        # Critic 网络
        critic_layers = []
        prev_dim = num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, dim))
            critic_layers.append(act_func)
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def act(self, obs):
        return self.actor(obs)
    
    def act_inference(self, obs):
        return self.actor(obs)

    def get_actions_log_prob_entropy_value(self, actor_obs, critic_obs, action=None):
        mean = self.actor(actor_obs)
        # Std Clamping: 防止方差过小导致数值不稳定
        std = self.log_std.exp().expand_as(mean)
        std = torch.clamp(std, min=0.05) 
        
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(critic_obs)

        return action, log_prob, entropy, value

class MAPPO:
    def __init__(self, actor_critic, cfg):
        self.ac = actor_critic
        self.cfg = cfg
        
        self.learning_rate = cfg["learning_rate"]
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.learning_rate)
        
    def update(self, storage):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        
        # 展平缓冲区数据
        b_obs = storage["obs"].reshape((-1, storage["obs"].shape[-1]))
        b_priv_obs = storage["priv_obs"].reshape((-1, storage["priv_obs"].shape[-1]))
        b_actions = storage["actions"].reshape((-1, storage["actions"].shape[-1]))
        b_logprobs = storage["logprobs"].reshape(-1)
        b_advantages = storage["advantages"].reshape(-1)
        b_returns = storage["returns"].reshape(-1)
        b_values = storage["values"].reshape(-1)

        # Advantage Normalization (Batch Level)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // self.cfg["num_mini_batches"]

        accumulated_kl = 0.0
        num_updates = 0

        for _ in range(self.cfg["num_learning_epochs"]):
            indices = torch.randperm(batch_size, device=b_obs.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = self.ac.get_actions_log_prob_entropy_value(
                    b_obs[mb_idx], b_priv_obs[mb_idx], b_actions[mb_idx]
                )
                
                log_ratio = new_logprob - b_logprobs[mb_idx]
                ratio = torch.exp(log_ratio)
                
                with torch.no_grad():
                    approx_kl = ((log_ratio ** 2) * 0.5).mean()
                    accumulated_kl += approx_kl.item()
                
                mb_advantages = b_advantages[mb_idx]

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg["clip_param"], 1.0 + self.cfg["clip_param"]) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                new_value = new_value.flatten()
                if self.cfg["use_clipped_value_loss"]:
                    v_clipped = b_values[mb_idx] + torch.clamp(new_value - b_values[mb_idx],
                                                             -self.cfg["clip_param"],
                                                             self.cfg["clip_param"])
                    v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                loss = policy_loss + self.cfg["value_loss_coef"] * v_loss - self.cfg["entropy_coef"] * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg["max_grad_norm"])
                self.optimizer.step()
                
                mean_value_loss += v_loss.item()
                mean_surrogate_loss += policy_loss.item()
                mean_entropy_loss += entropy.mean().item()
                num_updates += 1

        mean_kl = accumulated_kl / num_updates
        
        if self.cfg.get("schedule", "fixed") == "adaptive":
            target_kl = self.cfg["desired_kl"]
            if mean_kl > target_kl * 2.0:
                self.learning_rate = max(1e-6, self.learning_rate / 1.5)
            elif mean_kl < target_kl / 2.0 and mean_kl > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        return {
            "loss/value_function": mean_value_loss / num_updates,
            "loss/surrogate": mean_surrogate_loss / num_updates,
            "loss/entropy": mean_entropy_loss / num_updates,
            "policy/mean_kl": mean_kl,
            "policy/learning_rate": self.learning_rate
        }

class MAPPORunner:
    def __init__(self, env, train_cfg, log_dir, device="cuda:0"):
        self.env = env
        self.cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.runner_cfg = train_cfg["runner"]
        self.log_dir = log_dir
        self.device = device
        
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.max_iterations = self.runner_cfg["max_iterations"]
        self.save_interval = train_cfg["save_interval"]

        self.num_actor_obs = env.num_obs
        self.num_critic_obs = getattr(env, "num_privileged_obs", None)
        self.num_actions = env.num_actions

        self.alg = MAPPO(
            ActorCritic(
                self.num_actor_obs, 
                self.num_critic_obs, 
                self.num_actions,
                self.policy_cfg["actor_hidden_dims"],
                self.policy_cfg["critic_hidden_dims"],
                self.policy_cfg["activation"],
                self.policy_cfg["init_noise_std"]
            ).to(self.device),
            self.cfg
        )
        
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        self.num_envs = env.num_envs
        self.storage = {
            "obs": torch.zeros(self.num_steps_per_env, self.num_envs, self.num_actor_obs, device=self.device),
            "priv_obs": torch.zeros(self.num_steps_per_env, self.num_envs, self.num_critic_obs, device=self.device),
            "actions": torch.zeros(self.num_steps_per_env, self.num_envs, self.num_actions, device=self.device),
            "rewards": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "dones": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "logprobs": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "values": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "returns": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "advantages": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
            "time_outs": torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device),
        }
        
        self.ep_infos = deque(maxlen=100)
        
        self.start_iteration = 0

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
        if num_learning_iterations is None:
            num_learning_iterations = self.max_iterations

        obs, extras = self.env.get_observations()
        priv_obs = self.env.get_privileged_observations()
        
        print(f"Starting Training: {num_learning_iterations} iterations (Start from {self.start_iteration})")
        start_time = time.time()

        for it in range(self.start_iteration, num_learning_iterations):
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    self.storage["obs"][step] = obs
                    self.storage["priv_obs"][step] = priv_obs
                    
                    actions, log_prob, _, value = self.alg.ac.get_actions_log_prob_entropy_value(obs, priv_obs)
                    
                    self.storage["actions"][step] = actions
                    self.storage["logprobs"][step] = log_prob
                    self.storage["values"][step] = value.flatten()

                    obs, rewards, dones, infos = self.env.step(actions)
                    
                    if "privileged_obs" in infos:
                        priv_obs = infos["privileged_obs"]
                    else:
                        priv_obs = self.env.get_privileged_observations()
                    
                    self.storage["rewards"][step] = rewards
                    self.storage["dones"][step] = dones
                    
                    if "time_outs" in infos:
                        self.storage["time_outs"][step] = infos["time_outs"].repeat_interleave(self.env.num_drones)
                    else:
                        self.storage["time_outs"][step] = 0

                    if "episode" in infos:
                        self.ep_infos.append(infos["episode"])

                _, _, _, last_val = self.alg.ac.get_actions_log_prob_entropy_value(obs, priv_obs)
                last_val = last_val.flatten()

            self.compute_returns(last_val)
            stats = self.alg.update(self.storage)
            
            if it > 0 and it % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            self.log(it, stats, start_time)
            if (it + 1) % self.save_interval == 0:
                self.save(it + 1)

    def compute_returns(self, last_values):
        gae = 0
        gamma = self.cfg["gamma"]
        lam = self.cfg["lam"]
        for step in reversed(range(self.num_steps_per_env)):
            if step == self.num_steps_per_env - 1:
                next_values = last_values
                is_time_out = self.storage["time_outs"][step].float()
                next_is_not_terminal = 1.0 - (self.storage["dones"][step].float() * (1.0 - is_time_out))
            else:
                next_values = self.storage["values"][step + 1]
                is_time_out = self.storage["time_outs"][step].float()
                next_is_not_terminal = 1.0 - (self.storage["dones"][step].float() * (1.0 - is_time_out))
            
            delta = self.storage["rewards"][step] + gamma * next_values * next_is_not_terminal - self.storage["values"][step]
            gae = delta + gamma * lam * next_is_not_terminal * gae
            self.storage["advantages"][step] = gae
            self.storage["returns"][step] = gae + self.storage["values"][step]

    def log(self, it, stats, start_time):
        mean_rew = self.storage["rewards"].sum() / self.num_envs
        fps = int(self.num_steps_per_env * self.num_envs / (time.time() - start_time + 1e-6))
        
        log_string = f"Iter {it+1}/{self.max_iterations} | Rew: {mean_rew:.2f} | Loss: {stats['loss/surrogate']:.4f} | KL: {stats['policy/mean_kl']:.4f} | LR: {stats['policy/learning_rate']:.6f} | FPS: {fps}"
        
        if len(self.ep_infos) > 0:
            keys = self.ep_infos[0].keys()
            ep_stats = {}
            for k in keys:
                if k.startswith("rew_"): 
                    values = [ep_info[k] for ep_info in self.ep_infos if k in ep_info]
                    if values:
                        mean_val = np.mean(values)
                        ep_stats[k] = mean_val
                        self.writer.add_scalar(f"Episode/{k}", mean_val, it)
            
            core_keys = ["rew_target", "rew_progress", "rew_crash", "rew_obstacle","rew_team_coordination"]
            for k in core_keys:
                if k in ep_stats:
                    log_string += f" | {k[4:]}: {ep_stats[k]:.4f}"
        
        print(log_string)
        
        for k, v in stats.items():
            self.writer.add_scalar(k, v, it)
        self.writer.add_scalar("Train/mean_reward", mean_rew, it)
        self.writer.add_scalar("Train/FPS", fps, it)

    def save(self, it):
        path = os.path.join(self.log_dir, f"model_{it}.pt")
        torch.save({
            'model_state_dict': self.alg.ac.state_dict(),
            'iteration': it,
            'config': {'num_actor_obs': self.num_actor_obs, 'num_critic_obs': self.num_critic_obs}
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.alg.ac.load_state_dict(checkpoint['model_state_dict'])
        
        self.start_iteration = checkpoint.get('iteration', 0)
        
        print(f"Loaded model from {path}, resuming from iter: {self.start_iteration}")
