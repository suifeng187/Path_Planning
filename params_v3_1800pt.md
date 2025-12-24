# 训练参数配置 - multi-drone-ppo-v3

**检查点**: model_1800.pt

## PPO超参数

### 算法参数

- **class_name**: PPO
- **clip_param**: 0.2
- **desired_kl**: 0.01
- **entropy_coef**: 0.015
- **gamma**: 0.99
- **lam**: 0.95
- **learning_rate**: 0.00015
- **max_grad_norm**: 0.5
- **num_learning_epochs**: 4
- **num_mini_batches**: 4
- **schedule**: adaptive
- **use_clipped_value_loss**: True
- **value_loss_coef**: 0.2

### 策略网络参数

- **activation**: elu
- **actor_hidden_dims**: [256, 256, 128]
- **critic_hidden_dims**: [256, 256, 128]
- **init_noise_std**: 0.5
- **class_name**: ActorCritic

### 训练配置

- **num_steps_per_env**: 100
- **save_interval**: 100
- **seed**: 1
- **max_iterations**: 1000


## 环境参数

- **num_drones**: 3
- **episode_length_s**: 35.0
- **at_target_threshold**: 0.5
- **obstacle_safe_distance**: 0.3
- **obstacle_collision_distance**: 0.1
- **drone_safe_distance**: 0.2
- **drone_collision_distance**: 0.1
- **obstacle_radius**: 0.1
- **obstacle_height**: 2.5

## 奖励参数

- **target**: 80.0
- **progress**: 35.0
- **alive**: 3.0
- **smooth**: -1e-06
- **crash**: -10.0
- **obstacle**: -7
- **separation**: -0.1
- **direction**: 30.0

## 观测配置

- **num_obs**: 69
- **num_obs_per_drone**: 23
### obs_scales

- **rel_pos**: 0.3333333333333333
- **lin_vel**: 0.3333333333333333
- **ang_vel**: 0.31831015504887655


