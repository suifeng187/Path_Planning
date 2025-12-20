"""
MAPPO (Multi-Agent PPO) 算法实现
核心思想：集中式训练（Centralized Critic）+ 分布式执行（Decentralized Actor）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class MAPPOActor(nn.Module):
    """分布式Actor网络 - 每个智能体独立决策"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256], init_std=0.5):
        super().__init__()
        
        self.action_dim = action_dim
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        # 可学习的log_std，初始值较大以鼓励探索 
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)
    
    def forward(self, obs):
        features = self.backbone(obs)
        # 使用tanh限制输出范围到[-1, 1]
        mean = torch.tanh(self.mean_head(features))
        # std可学习，范围更宽以允许更多探索
        std = torch.clamp(self.log_std.exp(), min=0.1, max=1.0).expand_as(mean)
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros(obs.shape[0], device=obs.device)
        dist = Normal(mean, std)
        # 采样后也要clamp到有效范围
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        # 计算log_prob时使用原始采样值（在clamp之前）
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """集中式Critic网络 - 使用全局状态评估价值"""
    
    def __init__(self, global_obs_dim, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = global_obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, global_obs):
        return self.network(global_obs).squeeze(-1)


class MAPPOBuffer:
    """MAPPO经验回放缓冲区"""
    
    def __init__(self, num_envs, num_agents, num_steps, obs_dim, global_obs_dim, action_dim, device):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.device = device
        self.ptr = 0
        
        # 每个智能体的局部观测 [steps, envs, agents, obs_dim]
        self.obs = torch.zeros((num_steps, num_envs, num_agents, obs_dim), device=device)
        # 全局观测 [steps, envs, global_obs_dim]
        self.global_obs = torch.zeros((num_steps, num_envs, global_obs_dim), device=device)
        # 动作 [steps, envs, agents, action_dim]
        self.actions = torch.zeros((num_steps, num_envs, num_agents, action_dim), device=device)
        # 对数概率 [steps, envs, agents]
        self.log_probs = torch.zeros((num_steps, num_envs, num_agents), device=device)
        # 奖励 [steps, envs] - 共享奖励
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        # 终止标志 [steps, envs]
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        # 价值估计 [steps, envs]
        self.values = torch.zeros((num_steps, num_envs), device=device)
        # GAE优势 [steps, envs]
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        # 回报 [steps, envs]
        self.returns = torch.zeros((num_steps, num_envs), device=device)
    
    def store(self, obs, global_obs, actions, log_probs, rewards, dones, values):
        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.num_steps
    
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """计算GAE优势估计"""
        gae = torch.zeros((self.num_envs,), device=self.device)
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
        
        # 标准化优势（按环境维度展平后标准化）
        adv_flat = self.advantages.view(-1)
        self.advantages = ((self.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8))
    
    def get_batches(self, batch_size):
        """生成训练批次"""
        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            # 转换索引
            step_idx = batch_indices // self.num_envs
            env_idx = batch_indices % self.num_envs
            
            yield {
                'obs': self.obs[step_idx, env_idx],           # [batch, agents, obs_dim]
                'global_obs': self.global_obs[step_idx, env_idx],  # [batch, global_obs_dim]
                'actions': self.actions[step_idx, env_idx],   # [batch, agents, action_dim]
                'log_probs': self.log_probs[step_idx, env_idx],  # [batch, agents]
                'advantages': self.advantages[step_idx, env_idx],  # [batch]
                'returns': self.returns[step_idx, env_idx],   # [batch]
            }
    
    def clear(self):
        self.ptr = 0


class MAPPO:
    """MAPPO算法主类"""
    
    def __init__(
        self,
        num_agents,
        obs_dim,
        global_obs_dim,
        action_dim,
        device,
        lr_actor=3e-4,      # 提高学习率，加速学习
        lr_critic=5e-4,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.02,   # 提高熵系数，增加探索
        value_loss_coef=0.5,
        max_grad_norm=1.0,   # 放宽梯度裁剪
        num_epochs=5,        # 减少epoch，避免过拟合
        batch_size=512,
        share_actor=True,  # 是否共享Actor参数
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.device = device
        
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.share_actor = share_actor
        
        # 创建网络
        actor_hidden = [256, 256, 128]
        critic_hidden = [512, 256, 128]
        
        if share_actor:
            # 共享Actor参数（同质智能体）
            self.actor = MAPPOActor(obs_dim, action_dim, hidden_dims=actor_hidden, init_std=0.5).to(device)
            self.actors = [self.actor] * num_agents
        else:
            # 独立Actor（异质智能体）
            self.actors = [MAPPOActor(obs_dim, action_dim, hidden_dims=actor_hidden, init_std=0.5).to(device) for _ in range(num_agents)]
        
        # 集中式Critic
        self.critic = MAPPOCritic(global_obs_dim, hidden_dims=critic_hidden).to(device)
        
        # 优化器
        if share_actor:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        else:
            actor_params = []
            for actor in self.actors:
                actor_params.extend(actor.parameters())
            self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def get_actions(self, obs, deterministic=False):
        """
        获取所有智能体的动作
        obs: [num_envs, num_agents, obs_dim]
        返回: actions [num_envs, num_agents, action_dim], log_probs [num_envs, num_agents]
        """
        num_envs = obs.shape[0]
        actions = torch.zeros((num_envs, self.num_agents, self.action_dim), device=self.device)
        log_probs = torch.zeros((num_envs, self.num_agents), device=self.device)
        
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i, :]  # [num_envs, obs_dim]
            act, lp = actor.get_action(agent_obs, deterministic=deterministic)
            actions[:, i, :] = act
            log_probs[:, i] = lp
        
        return actions, log_probs
    
    def get_value(self, global_obs):
        """获取全局状态价值"""
        return self.critic(global_obs)
    
    def update(self, buffer):
        """更新网络参数"""
        buffer.compute_gae(
            self.get_value(buffer.global_obs[-1]).detach(),
            self.gamma,
            self.lam
        )
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.num_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = batch['obs']              # [batch, agents, obs_dim]
                global_obs = batch['global_obs']  # [batch, global_obs_dim]
                actions = batch['actions']      # [batch, agents, action_dim]
                old_log_probs = batch['log_probs']  # [batch, agents]
                advantages = batch['advantages']  # [batch]
                returns = batch['returns']      # [batch]
                
                # ========== 更新Actor ==========
                new_log_probs = torch.zeros_like(old_log_probs)
                entropy = torch.zeros_like(old_log_probs)
                
                for i, actor in enumerate(self.actors):
                    agent_obs = obs[:, i, :]
                    agent_actions = actions[:, i, :]
                    lp, ent = actor.evaluate_actions(agent_obs, agent_actions)
                    new_log_probs[:, i] = lp
                    entropy[:, i] = ent
                
                # 计算比率（所有智能体的联合比率）
                ratio = torch.exp(new_log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))
                
                # PPO裁剪目标
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # Actor总损失
                total_actor_loss_batch = actor_loss + self.entropy_coef * entropy_loss
                
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                if self.share_actor:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                else:
                    for actor in self.actors:
                        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # ========== 更新Critic ==========
                values = self.critic(global_obs)
                critic_loss = self.value_loss_coef * ((values - returns) ** 2).mean()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def save(self, path):
        """保存模型"""
        state = {
            'critic': self.critic.state_dict(),
        }
        if self.share_actor:
            state['actor'] = self.actor.state_dict()
        else:
            for i, actor in enumerate(self.actors):
                state[f'actor_{i}'] = actor.state_dict()
        torch.save(state, path)
    
    def load(self, path):
        """加载模型"""
        state = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(state['critic'])
        if self.share_actor:
            self.actor.load_state_dict(state['actor'])
        else:
            for i, actor in enumerate(self.actors):
                actor.load_state_dict(state[f'actor_{i}'])
