"""
多无人机避障路径规划环境 - PPO版本 (Local Perception Final)
1. 距离计算：依赖局部感知。
   - 障碍物：水平距离 + sensing_radius过滤。
   - 队友：3D距离 + sensing_radius过滤 + 每架独立计算。
2. 观测空间：
   - 队友：4维 [x, y, z, mask]，不可见时坐标置0且mask为0。
   - 障碍物：Top-2，不可见或不足填充 1.0。
"""
import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

class MultiDronePPOEnv:
    """多无人机PPO环境 - 局部感知版"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.num_drones = env_cfg.get("num_drones", 3)
        self.rendered_env_num = min(5, self.num_envs)
        
        self.num_obs_per_drone = obs_cfg["num_obs_per_drone"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions_per_drone = env_cfg["num_actions"]
        self.num_actions = self.num_actions_per_drone * self.num_drones
        self.device = gs.device

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])
        
        # 核心参数：感知半径和最近障碍物数量
        self.sensing_radius = env_cfg.get("sensing_radius", 3.0)
        self.num_nearest_obstacles = env_cfg.get("num_nearest_obstacles", 2)

        # ==================== 创建仿真场景 ====================
        if show_viewer:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                viewer_options=gs.options.ViewerOptions(
                    max_FPS=env_cfg["max_visualize_FPS"],
                    camera_pos=(5.0, 0.0, 5.0),
                    camera_lookat=(0.0, 0.0, 1.0),
                    camera_fov=50,
                ),
                vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                show_viewer=True,
            )
        else:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                show_viewer=False,
            )

        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加障碍物 ====================
        self.obstacles = []
        obstacle_positions = env_cfg.get("obstacle_positions", [])
        obstacle_radius = env_cfg.get("obstacle_radius", 0.12)
        obstacle_height = env_cfg.get("obstacle_height", 2.5)
        
        # 保存障碍物坐标的Tensor
        self.obstacle_pos_tensor = torch.tensor(obstacle_positions, device=self.device, dtype=gs.tc_float) if obstacle_positions else torch.empty((0,3), device=self.device)
        
        for pos in obstacle_positions:
            if show_viewer:
                obstacle = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        pos=pos,
                        radius=obstacle_radius,
                        height=obstacle_height,
                        fixed=True,
                        collision=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.3, 0.8)),
                    ),
                )
            else:
                obstacle = None
            self.obstacles.append({
                "entity": obstacle,
                "pos": torch.tensor(pos, device=gs.device),
                "radius": obstacle_radius
            })
        
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)
        self.drone_safe_distance = env_cfg.get("drone_safe_distance", 0.5)

        # ==================== 添加多架无人机 ====================
        self.drones = []
        self.drone_init_positions = env_cfg.get("drone_init_positions", [
            [-1.0, -2.5, 0.15], [0.0, -2.5, 0.15], [1.0, -2.5, 0.15],
        ])
        self.drone_goal_positions = env_cfg.get("drone_goal_positions", [
            [-1.0, 2.5, 0.15], [0.0, 2.5, 0.15], [1.0, 2.5, 0.15],
        ])
        self.drone_goal_pos_tensor = torch.tensor(self.drone_goal_positions, device=self.device, dtype=gs.tc_float)
        
        drone_colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        for i in range(self.num_drones):
            drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
            self.drones.append(drone)

        # 目标点可视化
        if env_cfg.get("visualize_target", False):
            for i, goal_pos in enumerate(self.drone_goal_positions):
                self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj", scale=0.08, pos=goal_pos,
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=drone_colors[i % len(drone_colors)]),
                    ),
                )

        if env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(5.0, 0.0, 5.0),
                lookat=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False,
            )
        else:
            self.cam = None

        self.scene.build(n_envs=num_envs)

        # ==================== 初始化奖励函数 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ==================== 初始化状态缓冲区 ====================
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        
        self.commands = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, self.num_drones, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        self.drone_ever_reached_target = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=torch.bool)
        
        # [修改] min_drone_dist 改为 (num_envs, num_drones)，每架独立
        self.min_obstacle_dist = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        self.min_drone_dist = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        for i, goal_pos in enumerate(self.drone_goal_positions):
            self.commands[envs_idx, i, 0] = goal_pos[0]
            self.commands[envs_idx, i, 1] = goal_pos[1]
            self.commands[envs_idx, i, 2] = goal_pos[2]

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        for i, drone in enumerate(self.drones):
            start_idx = i * self.num_actions_per_drone
            end_idx = start_idx + self.num_actions_per_drone
            drone_actions = self.actions[:, start_idx:end_idx]
            drone.set_propellels_rpm((1 + drone_actions * 0.4) * 14468.429183500699)

        self.scene.step()
        self.episode_length_buf += 1

        self.last_base_pos[:] = self.base_pos[:]
        for i, drone in enumerate(self.drones):
            self.base_pos[:, i, :] = drone.get_pos()
            self.base_quat[:, i, :] = drone.get_quat()
            
            self.base_euler[:, i, :] = quat_to_xyz(
                transform_quat_by_quat(
                    self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1),
                    self.base_quat[:, i, :],
                ),
                rpy=True, degrees=True,
            )
            
            inv_quat_i = inv_quat(self.base_quat[:, i, :])
            self.base_lin_vel[:, i, :] = transform_by_quat(drone.get_vel(), inv_quat_i)
            self.base_ang_vel[:, i, :] = transform_by_quat(drone.get_ang(), inv_quat_i)

        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.commands - self.base_pos

        # 计算距离 (Local Only)
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()

        # ==================== 终止条件 ====================
        crash_any = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        
        # 机间碰撞检测 (适配新形状 (B, N))
        drone_collision_per_drone = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        drone_collision_any = drone_collision_per_drone.any(dim=1)
        
        for i in range(self.num_drones):
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance) # 注意：如果是感知盲区，dist=100，这里不会触发crash
            )
            crash_any = crash_any | drone_crash
            
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            self.drone_ever_reached_target[:, i] = self.drone_ever_reached_target[:, i] | drone_success

        crash_any = crash_any | drone_collision_any

        success_all = torch.all(self.drone_ever_reached_target, dim=1)
        self.crash_condition = crash_any
        self.success_condition = success_all
        time_out = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = crash_any | success_all | time_out

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ==================== 构建观测 (Local + 4D Flag + Top-2) ====================
        obs_list = []
        has_obstacles = self.obstacle_pos_tensor.shape[0] > 0
        sensing_radius = self.env_cfg.get("sensing_radius", 3.0)
        
        for i in range(self.num_drones):
            # 1. 基础观测 (17维)
            base_obs = torch.cat([
                torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat[:, i, :],
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions[:, i*4:(i+1)*4],
            ], dim=-1)
            
            # 2. 队友感知 (4维: [x, y, z, mask])
            other_drones_rel_pos = []
            for j in range(self.num_drones):
                if i != j:
                    rel_vec = self.base_pos[:, j, :] - self.base_pos[:, i, :]
                    dist = torch.norm(rel_vec, dim=1, keepdim=True)
                    
                    # 4维感知 Mask
                    mask = (dist < sensing_radius).float()
                    
                    scaled_rel = torch.clip(rel_vec * self.obs_scales["rel_pos"], -1, 1)
                    masked_rel = scaled_rel * mask
                    
                    # [x, y, z, mask]
                    obs_with_flag = torch.cat([masked_rel, mask], dim=-1)
                    other_drones_rel_pos.append(obs_with_flag)
            
            # 3. 障碍物感知 (3维 + Top-2)
            # 默认填充值 1.0 (表示"安全/边界外")
            nearest_obs_vecs = torch.ones((self.num_envs, self.num_nearest_obstacles * 3), device=gs.device)
            
            if has_obstacles:
                curr_drone_pos = self.base_pos[:, i, :]
                
                # 计算水平距离
                obs_pos_expanded = self.obstacle_pos_tensor.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
                obs_pos_expanded[:, :, 2] = curr_drone_pos[:, 2].unsqueeze(1) 
                vecs = obs_pos_expanded - curr_drone_pos.unsqueeze(1) 
                dists = torch.norm(vecs, dim=-1) 
                
                # 局部感知筛选
                out_of_range_mask = dists > sensing_radius
                dists_masked = dists.clone()
                dists_masked[out_of_range_mask] = float('inf')
                
                # Top-K
                k = min(self.num_nearest_obstacles, self.obstacle_pos_tensor.shape[0])
                sorted_dists, indices = torch.topk(dists_masked, k, dim=1, largest=False)
                
                # 提取向量
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
                topk_vecs = torch.gather(vecs, 1, indices_expanded)
                topk_vecs = topk_vecs * self.obs_scales["obstacle"]
                topk_vecs = torch.clip(topk_vecs, -1, 1)
                
                # Mask有效性
                valid_mask = (sorted_dists < sensing_radius).unsqueeze(-1).float()
                
                # 融合
                final_vecs = topk_vecs * valid_mask + 1.0 * (1.0 - valid_mask)
                flat_obs = final_vecs.reshape(self.num_envs, -1)
                
                if k < self.num_nearest_obstacles:
                    nearest_obs_vecs[:, :k*3] = flat_obs
                else:
                    nearest_obs_vecs = flat_obs

            # 拼接
            drone_obs = torch.cat([base_obs] + other_drones_rel_pos + [nearest_obs_vecs], dim=-1)
            obs_list.append(drone_obs)
        
        self.obs_buf = torch.cat(obs_list, dim=-1)
        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_min_obstacle_distance(self):
        min_dist = torch.ones((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float) * 100.0
        
        if len(self.obstacles) == 0:
            return min_dist
            
        radius = self.sensing_radius
        
        for i in range(self.num_drones):
            drone_pos_xy = self.base_pos[:, i, :2]
            
            for obs in self.obstacles:
                obs_pos_xy = obs["pos"][:2].unsqueeze(0)
                
                dist_center = torch.norm(drone_pos_xy - obs_pos_xy, dim=1)
                dist_surface = dist_center - obs["radius"]
              
                is_visible = dist_surface < radius
                dist_filtered = torch.where(is_visible, dist_surface, torch.tensor(100.0, device=gs.device))
                
                min_dist[:, i] = torch.minimum(min_dist[:, i], dist_filtered)
                
        return min_dist

    def _get_min_drone_distance(self):
        min_dist = torch.ones((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float) * 100.0
        radius = self.sensing_radius

        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i == j:
                    continue
                
                dist_3d = torch.norm(self.base_pos[:, i, :] - self.base_pos[:, j, :], dim=1)
                
                is_visible = dist_3d < radius
                dist_filtered = torch.where(is_visible, dist_3d, torch.tensor(100.0, device=gs.device))
                
                min_dist[:, i] = torch.minimum(min_dist[:, i], dist_filtered)
                
        return min_dist

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        for i, drone in enumerate(self.drones):
            init_pos = torch.tensor(self.drone_init_positions[i], device=gs.device)
            self.base_pos[envs_idx, i, :] = init_pos
            self.last_base_pos[envs_idx, i, :] = init_pos
            self.base_quat[envs_idx, i, :] = self.base_init_quat
            
            drone.set_pos(self.base_pos[envs_idx, i, :], zero_velocity=True, envs_idx=envs_idx)
            drone.set_quat(self.base_quat[envs_idx, i, :], zero_velocity=True, envs_idx=envs_idx)
            drone.zero_all_dofs_velocity(envs_idx)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.drone_ever_reached_target[envs_idx] = False
        
        self.min_obstacle_dist[envs_idx] = 10.0
        self.min_drone_dist[envs_idx] = 10.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ==================== 奖励函数 ====================
    # 速度略快，降低引导；直线引导再增加；速度设置一下上限
    def _reward_target(self):
        target_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        for i in range(self.num_drones):
            
            #curr_dist_sq = torch.sum(torch.square(self.rel_pos[:, i, :]), dim=1)
            #last_dist_sq = torch.sum(torch.square(self.last_rel_pos[:, i, :]), dim=1)
            curr_dist = torch.norm(self.rel_pos[:, i, :], dim=1)
            last_dist = torch.norm(self.last_rel_pos[:, i, :], dim=1)
            
            dist_reduction = last_dist - curr_dist
            target_rew += dist_reduction 
        
            #target_rew += torch.clamp(5 - curr_dist, min=0) * 0.05
            
            drone_at_target = curr_dist < self.env_cfg["at_target_threshold"]
            target_rew[drone_at_target] += 5.0 
            target_rew[self.success_condition] += 15.0 
        
        return target_rew / self.num_drones

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, :, 2]  # (num_envs, num_drones)
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return torch.mean(yaw_rew, dim=1)  # 对所有无人机求平均，返回 (num_envs,)

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=2)  # (num_envs, num_drones)
        return torch.mean(angular_rew, dim=1)  # 对所有无人机求平均，返回 (num_envs,)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1.0
        return crash_rew

    def _reward_obstacle(self):
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew

        for i in range(self.num_drones):
            d = self.min_obstacle_dist[:, i]
            safe_dist = self.obstacle_safe_distance
            mask = d < safe_dist
            if mask.any():
                obstacle_rew[mask] += (safe_dist - d[mask]) / safe_dist

        return obstacle_rew / self.num_drones

    def _reward_separation(self):
        sep_rew_per_drone = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        danger_dist = self.drone_safe_distance
        
        close_mask = self.min_drone_dist < danger_dist
        if close_mask.any():
            # 计算归一化惩罚 (0.0 ~ 1.0)
            sep_rew_per_drone[close_mask] = (danger_dist - self.min_drone_dist[close_mask]) / danger_dist
        
        return torch.mean(sep_rew_per_drone, dim=1)

    def _reward_alive(self):
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0.0
        return alive_rew

    def _reward_progress(self):
        """
        Alignment Reward
        计算“当前速度方向”与“目标方向”的余弦相似度。
        范围：[-1.0, 1.0]
        """
        alignment_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        for i in range(self.num_drones):
            # self.rel_pos 已经是 (Goal - Pos)
            target_vec = self.rel_pos[:, i, :] 
            # 归一化 (添加极小值防止除以0)
            target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)

            vel_vec = self.base_lin_vel[:, i, :]
            speed = torch.norm(vel_vec, dim=1, keepdim=True)
            # 归一化
            vel_dir = vel_vec / (speed + 1e-6)
            dot_prod = torch.sum(target_dir * vel_dir, dim=1)
            
            # 如果速度非常小(接近悬停)，方向可能是噪声，可以将奖励置为0
            mask_static = speed.squeeze(-1) < 0.05
            dot_prod[mask_static] = 0.0

            alignment_rew += dot_prod

        return alignment_rew / self.num_drones

        