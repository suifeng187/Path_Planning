"""
多无人机避障路径规划环境 - MAPPO版本 (CTDE架构)
【修改说明】
1. 邻居观测移除四元数 (RelQuat)。
2. 奖励函数逻辑更新 (Yaw 设为 0)。
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

class MultiDroneMAPPOEnv:
    """多无人机MAPPO环境 - 集中训练分散执行"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_physical_envs = num_envs
        self.num_drones = env_cfg.get("num_drones", 3)
        self.num_envs = self.num_physical_envs * self.num_drones 
        self.rendered_env_num = min(5, self.num_physical_envs)
        
        # 观测维度配置
        self.num_obs_per_drone = obs_cfg["num_obs_per_drone"] # Actor 输入维度
        self.num_obs = self.num_obs_per_drone
        
        # 特权观测维度 (Critic 输入)
        self.num_privileged_obs = self.num_obs_per_drone * self.num_drones
        
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])
        
        # 团队奖励系数 (0.0: 完全自私, 1.0: 完全平均)
        self.team_spirit = env_cfg.get("team_spirit", 0.3) 
        
        self.sensing_radius = env_cfg.get("sensing_radius", 3.0)
        self.num_nearest_obstacles = env_cfg.get("num_nearest_obstacles", 2)

        # 轮次控制参数
        self.rounds_per_ep = env_cfg.get("rounds_per_episode", 5)
        self.obstacle_area_radius = env_cfg.get("obstacle_area_radius", 3.5)
        self.success_rounds = torch.zeros(self.num_physical_envs, device=self.device, dtype=torch.long)

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
        obstacle_radius = env_cfg.get("obstacle_radius", 0.1)
        obstacle_height = env_cfg.get("obstacle_height", 2.5)
        
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
            [-4.5, -1.0, 1.0], [-4.5, 0.0, 1.0], [-4.5, 1.0, 1.0],
        ])
        
        drone_colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        for i in range(self.num_drones):
            drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
            self.drones.append(drone)

        # 可视化目标点
        self.target_entities = []
        if env_cfg.get("visualize_target", False):
            for i in range(self.num_drones):
                entity = self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj", scale=0.08, pos=(0, 0, -10.0),
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=drone_colors[i % len(drone_colors)]),
                    ),
                )
                self.target_entities.append(entity)

        if env_cfg.get("visualize_camera", False):
            camera_height = 8.0
            lookat_height = 0.8
            tilt_angle_rad = math.radians(20) 
            horizontal_offset = (camera_height - lookat_height) * math.tan(tilt_angle_rad)
            self.cam = self.scene.add_camera(
                res=(1280, 720), 
                pos=(horizontal_offset, 0.0, camera_height), 
                lookat=(0.0, 0.0, lookat_height), 
                fov=50, 
                GUI=False,
            )
        else:
            self.cam = None

        self.scene.build(n_envs=self.num_physical_envs)

        # ==================== 初始化 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        
        self.commands = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        
        self.last_actions_phys = torch.zeros((self.num_physical_envs, self.num_drones, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.prev_actions_phys = torch.zeros_like(self.last_actions_phys)

        self.base_pos = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_physical_envs, self.num_drones, 4), device=gs.device, dtype=gs.tc_float)
        
        self.world_lin_vel = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        
        self.base_lin_vel = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float) # 机体系
        self.base_ang_vel = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float) # 机体系
        self.base_euler = torch.zeros((self.num_physical_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        self.drone_ever_reached_target = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device, dtype=torch.bool)
        self.min_obstacle_dist = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        self.min_drone_dist = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        self.phys_crash_cond = torch.zeros((self.num_physical_envs,), device=gs.device, dtype=torch.bool)
        self.phys_success_cond = torch.zeros((self.num_physical_envs,), device=gs.device, dtype=torch.bool)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _resample_commands(self, physical_envs_idx):
        num_resample = len(physical_envs_idx)
        if num_resample == 0: return

        has_obstacles = self.obstacle_pos_tensor.shape[0] > 0
        obstacle_radius = self.obstacles[0]["radius"] if has_obstacles else 0.1
        safe_threshold = obstacle_radius + 0.3 

        x = torch.zeros((num_resample, self.num_drones), device=self.device)
        y = torch.zeros((num_resample, self.num_drones), device=self.device)
        z = torch.ones((num_resample, self.num_drones), device=self.device) * 0.8 

        to_generate = torch.ones((num_resample, self.num_drones), dtype=torch.bool, device=self.device)
        
        for _ in range(20):
            if not to_generate.any(): break
            num_gen = to_generate.sum()
            r = torch.sqrt(torch.rand(num_gen, device=self.device)) * (self.obstacle_area_radius - 0.3)
            theta = torch.rand(num_gen, device=self.device) * 2 * math.pi
            x[to_generate] = r * torch.cos(theta)
            y[to_generate] = r * torch.sin(theta)

            if not has_obstacles: break

            curr_targets = torch.stack([x[to_generate], y[to_generate]], dim=-1)
            obs_xy = self.obstacle_pos_tensor[:, :2]
            dists = torch.cdist(curr_targets, obs_xy)
            min_dists, _ = dists.min(dim=1)
            collisions = min_dists < safe_threshold
            
            indices = torch.nonzero(to_generate, as_tuple=True)
            update_mask = torch.zeros_like(to_generate)
            update_mask[indices] = collisions
            to_generate = update_mask

        self.commands[physical_envs_idx, :, 0] = x
        self.commands[physical_envs_idx, :, 1] = y
        self.commands[physical_envs_idx, :, 2] = z

        current_pos = self.base_pos[physical_envs_idx]
        new_targets = self.commands[physical_envs_idx]
        new_rel_vec = new_targets - current_pos
        
        self.rel_pos[physical_envs_idx] = new_rel_vec
        self.last_rel_pos[physical_envs_idx] = new_rel_vec

        if self.env_cfg.get("visualize_target", False) and self.rendered_env_num > 0:
            for env_idx in physical_envs_idx:
                if env_idx < self.rendered_env_num:
                    for i in range(self.num_drones):
                        target_pos = self.commands[env_idx, i, :].cpu().numpy()
                        if i < len(self.target_entities):
                            self.target_entities[i].set_pos(target_pos, envs_idx=[env_idx.item()])

    def step(self, actions):
        self.extras = {}
        actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        phys_actions = actions.view(self.num_physical_envs, self.num_drones, self.num_actions)
        
        self.prev_actions_phys[:] = self.last_actions_phys[:]
        self.last_actions_phys[:] = phys_actions[:] 
        
        for i, drone in enumerate(self.drones):
            drone_actions = phys_actions[:, i, :]
            drone.set_propellels_rpm((1 + drone_actions * 0.4) * 14468.429183500699)

        self.scene.step()
        self.episode_length_buf += 1

        self.last_base_pos[:] = self.base_pos[:]
        for i, drone in enumerate(self.drones):
            self.base_pos[:, i, :] = drone.get_pos()
            self.base_quat[:, i, :] = drone.get_quat()
            self.base_euler[:, i, :] = quat_to_xyz(
                transform_quat_by_quat(
                    self.inv_base_init_quat.unsqueeze(0).expand(self.num_physical_envs, -1),
                    self.base_quat[:, i, :],
                ), rpy=True, degrees=True,
            )
            
            # [NEW] 记录世界系速度
            vel_world = drone.get_vel()
            self.world_lin_vel[:, i, :] = vel_world
            
            # 记录机体系速度
            inv_quat_i = inv_quat(self.base_quat[:, i, :])
            self.base_lin_vel[:, i, :] = transform_by_quat(vel_world, inv_quat_i)
            self.base_ang_vel[:, i, :] = transform_by_quat(drone.get_ang(), inv_quat_i)

        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.commands - self.base_pos
        
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()

        phys_crash_any = torch.zeros((self.num_physical_envs,), device=gs.device, dtype=torch.bool)
        drone_collision_per_drone = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        phys_collision_any = drone_collision_per_drone.any(dim=1)
        
        for i in range(self.num_drones):
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance)
            )
            phys_crash_any = phys_crash_any | drone_crash
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            self.drone_ever_reached_target[:, i] = self.drone_ever_reached_target[:, i] | drone_success

        phys_crash_any = phys_crash_any | phys_collision_any
        phys_success_all = torch.all(self.drone_ever_reached_target, dim=1)
        
        env_ids_success = phys_success_all.nonzero(as_tuple=False).flatten()
        if len(env_ids_success) > 0:
            self.success_rounds[env_ids_success] += 1
            is_episode_done = self.success_rounds[env_ids_success] >= self.rounds_per_ep
            env_ids_next_round = env_ids_success[~is_episode_done]
            
            if len(env_ids_next_round) > 0:
                self._resample_commands(env_ids_next_round)
                self.drone_ever_reached_target[env_ids_next_round] = False
                phys_success_all[env_ids_next_round] = False

        self.phys_crash_cond = phys_crash_any
        self.phys_success_cond = phys_success_all
        
        self.rew_buf[:] = 0.0
        raw_rewards = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device)
        
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            rew_shaped = rew.view(self.num_physical_envs, self.num_drones)
            raw_rewards += rew_shaped
            self.episode_sums[name] += rew

        if self.team_spirit > 0:
            team_mean_reward = raw_rewards.mean(dim=1, keepdim=True)
            final_rewards = (1.0 - self.team_spirit) * raw_rewards + self.team_spirit * team_mean_reward
            self.rew_buf = final_rewards.flatten()
        else:
            self.rew_buf = raw_rewards.flatten()

        phys_time_out = self.episode_length_buf.view(self.num_physical_envs, self.num_drones)[:, 0] >= self.max_episode_length
        phys_reset = phys_crash_any | phys_success_all | phys_time_out
        
        self.reset_buf = phys_reset.repeat_interleave(self.num_drones)
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if len(env_ids_to_reset) > 0:
            self.reset_idx(env_ids_to_reset)

        self._compute_observations()
        
        self.extras["privileged_obs"] = self.privileged_obs_buf
        self.extras["time_outs"] = phys_time_out 

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_observations(self):
        """
        全机体坐标系观测 + 仅观测最近1个邻居 + 引入相对速度 + 【修改】移除相对姿态
        """
        obs_list = []
        has_obstacles = self.obstacle_pos_tensor.shape[0] > 0
        sensing_radius = self.sensing_radius
        
        for i in range(self.num_drones):
            # 获取自身逆姿态 (World -> Body)
            inv_self_quat = inv_quat(self.base_quat[:, i, :])
            
            # --- [A] 自身状态 (13 dims) ---
            # 1. 目标位置转机体系
            rel_pos_target_body = transform_by_quat(self.rel_pos[:, i, :], inv_self_quat)
            
            # 2. 构建自身观测
            base_obs = torch.cat([
                torch.clip(rel_pos_target_body * self.obs_scales["rel_pos"], -1, 1), # (3)
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1), # (3)
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1), # (3)
                self.last_actions_phys[:, i, :], # (4)
            ], dim=-1)
            
            # --- [B] 友机感知 (最近的 1 个, 7 dims) ---
            # 1. 计算与所有其他无人机的距离
            pos_i = self.base_pos[:, i:i+1, :] # (B, 1, 3)
            all_dists = torch.norm(self.base_pos - pos_i, dim=-1) # (B, N)
            
            all_dists[:, i] = float('inf')
            
            # 找到最近邻居
            nearest_vals, nearest_idxs = torch.min(all_dists, dim=1) # (B,)
            
            def gather_neighbor(tensor, idxs):
                return tensor[torch.arange(self.num_physical_envs), idxs]

            neigh_pos = gather_neighbor(self.base_pos, nearest_idxs)       # World Pos
            neigh_vel = gather_neighbor(self.world_lin_vel, nearest_idxs) # World Vel
            # neigh_quat = gather_neighbor(self.base_quat, nearest_idxs)  # [Deleted] 不再需要邻居姿态
            
            # 2. 计算相对量并转机体系
            rel_pos_neigh_world = neigh_pos - self.base_pos[:, i, :]
            rel_pos_neigh_body = transform_by_quat(rel_pos_neigh_world, inv_self_quat)
            
            # 相对速度 (World -> Body)
            rel_vel_neigh_world = neigh_vel - self.world_lin_vel[:, i, :]
            rel_vel_neigh_body = transform_by_quat(rel_vel_neigh_world, inv_self_quat)
            
            # [Deleted] 相对姿态
            # rel_quat_neigh = transform_quat_by_quat(inv_self_quat, neigh_quat)
            
            # 3. Mask
            mask = (nearest_vals < sensing_radius).float().unsqueeze(-1)
            
            # 4. 拼接邻居观测
            # Pos(3) + Vel(3) + Mask(1) = 7 dims (原 11 dims)
            neighbor_obs = torch.cat([
                torch.clip(rel_pos_neigh_body * self.obs_scales["rel_pos"], -1, 1) * mask,
                torch.clip(rel_vel_neigh_body * self.obs_scales["lin_vel"], -1, 1) * mask,
                # rel_quat_neigh * mask, # [Deleted] 移除四元数
                mask
            ], dim=-1)

            # --- [C] 障碍物感知 (Top-K, 转机体系, 保持原样) ---
            nearest_obs_vecs = torch.ones((self.num_physical_envs, self.num_nearest_obstacles * 3), device=gs.device) * 2.0
            
            if has_obstacles:
                curr_drone_pos = self.base_pos[:, i, :]
                obs_pos_expanded = self.obstacle_pos_tensor.unsqueeze(0).expand(self.num_physical_envs, -1, -1).clone()
                obs_pos_expanded[:, :, 2] = curr_drone_pos[:, 2].unsqueeze(1) 
                
                # 世界系差值
                vecs_world = obs_pos_expanded - curr_drone_pos.unsqueeze(1) 
                dists = torch.norm(vecs_world, dim=-1) 
                
                out_of_range_mask = dists > sensing_radius
                dists_masked = dists.clone()
                dists_masked[out_of_range_mask] = float('inf')
                
                k = min(self.num_nearest_obstacles, self.obstacle_pos_tensor.shape[0])
                sorted_dists, indices = torch.topk(dists_masked, k, dim=1, largest=False)
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
                topk_vecs_world = torch.gather(vecs_world, 1, indices_expanded)
                
                # 将障碍物向量旋转至机体坐标系
                B, K, _ = topk_vecs_world.shape
                flat_vecs = topk_vecs_world.view(B * K, 3)
                flat_quat = inv_self_quat.unsqueeze(1).repeat(1, K, 1).view(B * K, 4)
                
                flat_vecs_body = transform_by_quat(flat_vecs, flat_quat)
                topk_vecs_body = flat_vecs_body.view(B, K, 3)
                
                # 裁剪与 Mask
                topk_vecs_body = torch.clip(topk_vecs_body * self.obs_scales["obstacle"], -1, 1)
                valid_mask = (sorted_dists < sensing_radius).unsqueeze(-1).float()
                
                final_vecs = topk_vecs_body * valid_mask + 2.0 * (1.0 - valid_mask)
                flat_obs = final_vecs.reshape(self.num_physical_envs, -1)
                if k < self.num_nearest_obstacles:
                    nearest_obs_vecs[:, :k*3] = flat_obs
                else:
                    nearest_obs_vecs = flat_obs

            # --- [D] 最终拼接 ---
            # Total = Base(13) + Neighbor(7) + Obstacles(6) = 26
            drone_obs = torch.cat([base_obs, neighbor_obs, nearest_obs_vecs], dim=-1)
            obs_list.append(drone_obs)
        
        # 整理 Buffer
        stacked_local_obs = torch.stack(obs_list, dim=0) 
        permuted_local_obs = stacked_local_obs.permute(1, 0, 2)
        self.obs_buf = permuted_local_obs.reshape(self.num_envs, self.num_obs)
        
        # Critic 观测
        global_state = stacked_local_obs.permute(1, 0, 2).reshape(self.num_physical_envs, -1)
        global_state_expanded = global_state.unsqueeze(1).expand(-1, self.num_drones, -1)
        self.privileged_obs_buf = global_state_expanded.reshape(self.num_envs, self.num_privileged_obs)

    def _get_min_obstacle_distance(self):
        min_dist = torch.ones((self.num_physical_envs, self.num_drones), device=gs.device, dtype=gs.tc_float) * 100.0
        if len(self.obstacles) == 0: return min_dist
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
        min_dist = torch.ones((self.num_physical_envs, self.num_drones), device=gs.device, dtype=gs.tc_float) * 100.0
        radius = self.sensing_radius
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i == j: continue
                dist_3d = torch.norm(self.base_pos[:, i, :] - self.base_pos[:, j, :], dim=1)
                is_visible = dist_3d < radius
                dist_filtered = torch.where(is_visible, dist_3d, torch.tensor(100.0, device=gs.device))
                min_dist[:, i] = torch.minimum(min_dist[:, i], dist_filtered)
        return min_dist

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0: return
        phys_env_ids = torch.unique(torch.div(env_ids, self.num_drones, rounding_mode='floor'))
        if len(phys_env_ids) == 0: return

        for i, drone in enumerate(self.drones):
            init_pos = torch.tensor(self.drone_init_positions[i], device=gs.device)
            self.base_pos[phys_env_ids, i, :] = init_pos
            self.last_base_pos[phys_env_ids, i, :] = init_pos
            self.base_quat[phys_env_ids, i, :] = self.base_init_quat
            
            drone.set_pos(self.base_pos[phys_env_ids, i, :], zero_velocity=True, envs_idx=phys_env_ids)
            drone.set_quat(self.base_quat[phys_env_ids, i, :], zero_velocity=True, envs_idx=phys_env_ids)
            drone.zero_all_dofs_velocity(phys_env_ids)

        self.base_lin_vel[phys_env_ids] = 0
        self.base_ang_vel[phys_env_ids] = 0
        self.world_lin_vel[phys_env_ids] = 0 # reset world vel
        self.last_actions_phys[phys_env_ids] = 0.0
        self.prev_actions_phys[phys_env_ids] = 0.0
        
        all_agent_indices = []
        for p_id in phys_env_ids:
            start = p_id * self.num_drones
            for d in range(self.num_drones):
                all_agent_indices.append(start + d)
        all_agent_indices = torch.tensor(all_agent_indices, device=gs.device)

        self.episode_length_buf[all_agent_indices] = 0
        self.reset_buf[all_agent_indices] = True
        self.drone_ever_reached_target[phys_env_ids] = False
        self.min_obstacle_dist[phys_env_ids] = 10.0
        self.min_drone_dist[phys_env_ids] = 10.0
        
        self.success_rounds[phys_env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][all_agent_indices]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][all_agent_indices] = 0.0

        self._resample_commands(phys_env_ids)
        self._compute_observations()

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, self.privileged_obs_buf

    # ==================== 奖励函数 ====================
    def _reward_target(self):
        curr_dist = torch.norm(self.rel_pos, dim=2)
        last_dist = torch.norm(self.last_rel_pos, dim=2)
        dist_reduction = last_dist - curr_dist
        rew = dist_reduction
        drone_at_target = curr_dist < self.env_cfg["at_target_threshold"]
        rew[drone_at_target] += 0.02
        return rew.flatten()

    def _reward_smooth(self):
        diff = self.last_actions_phys - self.prev_actions_phys
        smooth_penalty = torch.sum(torch.square(diff), dim=2)
        return -smooth_penalty.flatten()

    def _reward_yaw(self):
        """
        奖励机头对准速度方向 (相对稳定性)
        """
        # 1. 获取水平速度
        vel_xy = self.base_lin_vel[:, :, :2] 
        speed = torch.norm(vel_xy, dim=-1)
        
        # 2. 计算对齐度 (Vx / Speed)
        # 在机体坐标系下，Vx 就是机头方向的速度分量
        alignment = vel_xy[:, :, 0] / (speed + 1e-6)
        
        # 3. 只有在移动时才计算 (速度 > 0.1m/s)
        mask_moving = (speed > 0.1).float()
        
        # 4. 给正向对齐以奖励
        return (alignment * mask_moving).flatten()

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=2)
        return angular_rew.flatten()

    def _reward_crash(self):
        crash_expanded = self.phys_crash_cond.unsqueeze(1).expand(-1, self.num_drones)
        return crash_expanded.float().flatten()

    def _reward_obstacle(self):
        obstacle_rew = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device)
        if len(self.obstacles) > 0:
            d = self.min_obstacle_dist
            safe_dist = self.obstacle_safe_distance
            mask = d < safe_dist
            obstacle_rew[mask] = (safe_dist - d[mask]) / safe_dist
        return obstacle_rew.flatten()

    def _reward_separation(self):
        sep_rew = torch.zeros((self.num_physical_envs, self.num_drones), device=gs.device)
        danger_dist = self.drone_safe_distance
        mask = self.min_drone_dist < danger_dist
        sep_rew[mask] = (danger_dist - self.min_drone_dist[mask]) / danger_dist
        return sep_rew.flatten()

    def _reward_alive(self):
        alive_rew = torch.ones((self.num_physical_envs, self.num_drones), device=gs.device)
        crash_expanded = self.phys_crash_cond.unsqueeze(1).expand(-1, self.num_drones)
        alive_rew[crash_expanded] = 0.0
        return alive_rew.flatten()

    def _reward_progress(self):
        target_vec = self.rel_pos
        target_dir = target_vec / (torch.norm(target_vec, dim=2, keepdim=True) + 1e-6)
        vel_vec = self.base_lin_vel
        speed = torch.norm(vel_vec, dim=2, keepdim=True)
        vel_dir = vel_vec / (speed + 1e-6)
        dot_prod = torch.sum(target_dir * vel_dir, dim=2)
        mask_static = speed.squeeze(-1) < 0.05
        dot_prod[mask_static] = 0.0
        return dot_prod.flatten()

    def _reward_team_coordination(self):
        dists = torch.norm(self.rel_pos, dim=2)
        max_dist, _ = torch.max(dists, dim=1) 
        scale_factor = 6.0
        penalty = torch.tanh(max_dist / scale_factor)
        rew = -penalty
        return rew.unsqueeze(1).expand(-1, self.num_drones).flatten()