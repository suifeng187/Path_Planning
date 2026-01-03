"""
单无人机序列导航环境 - 悬停避障版
1. 任务：每回合需连续到达 10 个随机目标。
2. 逻辑：
   - 目标生成：在障碍物群组内部随机生成。
   - 到达判定：必须在目标半径内【连续悬停 0.5 秒】才算到达。
   - 到达后：奖励 +5，刷新下一个目标。
   - 完赛：到达第10个并悬停成功后，奖励 +5，重置。
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

class SingleDronePPOEnv:
    """单无人机序列导航PPO环境"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.num_drones = 1
        self.rendered_env_num = min(1, self.num_envs)
        
        self.max_goals_per_episode = 10
        
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
        
        self.sensing_radius = env_cfg.get("sensing_radius", 3.0)
        self.num_nearest_obstacles = env_cfg.get("num_nearest_obstacles", 2)

        # === 目标与悬停参数 ===
        self.arena_radius = env_cfg.get("arena_radius", 4.5) # 障碍物群半径
        self.goal_height = env_cfg.get("goal_height", 0.8)
        self.obstacle_radius_check = env_cfg.get("obstacle_radius", 0.1) + 0.3 
        
        # 悬停判定
        self.hover_duration = env_cfg.get("hover_duration_s", 0.5)
        self.hover_steps_needed = int(self.hover_duration / self.dt)
        # 初始化悬停计时器 (每个环境一个计数器)
        self.target_hold_timer = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        # ==================== 创建仿真场景 ====================
        if show_viewer:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                viewer_options=gs.options.ViewerOptions(
                    max_FPS=env_cfg["max_visualize_FPS"],
                    camera_pos=(5.0, -5.0, 5.0), # 视角拉远一点看大场景
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
        
        self.obstacle_pos_tensor = torch.tensor(obstacle_positions, device=self.device, dtype=gs.tc_float) if obstacle_positions else torch.empty((0,3), device=self.device)
        
        for pos in obstacle_positions:
            if show_viewer:
                self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        pos=pos, radius=obstacle_radius, height=obstacle_height,
                        fixed=True, collision=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.3, 0.8)),
                    ),
                )
            self.obstacles.append({
                "pos": torch.tensor(pos, device=gs.device),
                "radius": obstacle_radius
            })
        
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)

        # ==================== 添加无人机 ====================
        self.drones = []
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        self.drones.append(drone)

        # 目标点可视化
        self.target_visualizer = None
        if env_cfg.get("visualize_target", False):
            self.target_visualizer = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj", scale=0.1, pos=(0,0,-10),
                    fixed=True, collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.2, 0.2)),
                ),
            )

        if env_cfg.get("visualize_camera", False):
            # 从上方俯瞰视角，倾斜20°
            # 相机高度约8米，根据20°倾斜角计算水平偏移
            camera_height = 8.0
            lookat_height = 0.8
            tilt_angle_rad = math.radians(20)  # 20度倾斜
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

        self.scene.build(n_envs=num_envs)

        # ==================== 初始化缓冲区 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            if hasattr(self, "_reward_" + name):
                self.reward_functions[name] = getattr(self, "_reward_" + name)
                self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

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

        self.min_obstacle_dist = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        self.goals_reached_count = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _sample_valid_commands(self, env_ids, origin_pos):
        """
        在障碍物群组范围内(arena_radius)生成随机目标点。
        注意：不再完全基于 origin_pos 生成，而是全局生成 + 距离筛选。
        """
        num_targets = len(env_ids)
        if num_targets == 0:
            return
        
        # 1. 极坐标采样：在 arena_radius 内随机
        # 使用 sqrt(rand) 保证在圆内均匀分布
        r = torch.sqrt(torch.rand(num_targets, device=self.device)) * (self.arena_radius - 0.5) # 留0.5m边界
        theta = torch.rand(num_targets, device=self.device) * 2 * math.pi
        
        new_goals = torch.zeros((num_targets, 3), device=self.device)
        new_goals[:, 0] = r * torch.cos(theta)
        new_goals[:, 1] = r * torch.sin(theta)
        new_goals[:, 2] = self.goal_height
        
        # 2. 距离筛选：新目标点不能离当前位置太近 (例如 < 2.5m)，否则没有挑战
        # 如果太近，就强制把目标设在圆心的对面
        curr_xy = origin_pos[:, :2]
        goal_xy = new_goals[:, :2]
        dist_to_curr = torch.norm(goal_xy - curr_xy, dim=1)
        
        too_close_mask = dist_to_curr < 2.5
        if too_close_mask.any():
            # 简单的策略：如果太近，就翻转坐标 (x,y) -> (-x, -y)
            # 这样大概率会变远，且仍在圆内
            new_goals[too_close_mask, 0] *= -1
            new_goals[too_close_mask, 1] *= -1

        # 3. 障碍物防重叠检测
        if self.obstacle_pos_tensor.shape[0] > 0:
            diff = new_goals.unsqueeze(1) - self.obstacle_pos_tensor.unsqueeze(0)
            dist_to_obs = torch.norm(diff[:, :, :2], dim=-1)
            min_dist_to_obs, _ = torch.min(dist_to_obs, dim=1)
            
            invalid_mask = min_dist_to_obs < self.obstacle_radius_check
            if invalid_mask.any():
                # 如果撞到障碍物，简单策略：微调位置或者直接拉回原点上方(最安全)
                # 考虑到训练效率，这里简单地向圆心移动一点
                new_goals[invalid_mask, :2] *= 0.5 
        
        self.commands[env_ids, 0, :] = new_goals

        if self.target_visualizer is not None and self.rendered_env_num == 1 and 0 in env_ids:
            self.target_visualizer.set_pos(new_goals[0].cpu().numpy())

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
                transform_quat_by_quat(self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1), self.base_quat[:, i, :]),
                rpy=True, degrees=True,
            )
            inv_quat_i = inv_quat(self.base_quat[:, i, :])
            self.base_lin_vel[:, i, :] = transform_by_quat(drone.get_vel(), inv_quat_i)
            self.base_ang_vel[:, i, :] = transform_by_quat(drone.get_ang(), inv_quat_i)

        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.commands - self.base_pos

        self.min_obstacle_dist = self._get_min_obstacle_distance()
        
        # ==================== 悬停与到达逻辑 (核心修改) ====================
        curr_dist = torch.norm(self.rel_pos[:, 0, :], dim=1)
        in_target_zone = curr_dist < self.env_cfg["at_target_threshold"]
        
        # 1. 如果在区域内，计时器 +1
        self.target_hold_timer[in_target_zone] += 1
        # 2. 如果离开区域，计时器重置
        self.target_hold_timer[~in_target_zone] = 0
        
        # 3. 判定：是否满足悬停时间
        success_hold = self.target_hold_timer >= self.hover_steps_needed
        
        # just_reached 用于给奖励
        self.just_reached = success_hold 
        
        if success_hold.any():
            reached_ids = success_hold.nonzero(as_tuple=False).flatten()
            self.goals_reached_count[reached_ids] += 1
            
            # 重要：到达后重置该环境的计时器
            self.target_hold_timer[reached_ids] = 0
            
            continuing_mask = self.goals_reached_count[reached_ids] < self.max_goals_per_episode
            continuing_ids = reached_ids[continuing_mask]
            
            if len(continuing_ids) > 0:
                self._sample_valid_commands(continuing_ids, self.base_pos[continuing_ids, 0, :])
                new_rel_pos = self.commands[continuing_ids] - self.base_pos[continuing_ids]
                self.rel_pos[continuing_ids] = new_rel_pos
                self.last_rel_pos[continuing_ids] = new_rel_pos

        # ==================== 终止条件判定 ====================
        crash = (
            (torch.abs(self.base_euler[:, 0, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (self.base_pos[:, 0, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.min_obstacle_dist[:, 0] < self.obstacle_collision_distance)
        )
        
        success_all = self.goals_reached_count >= self.max_goals_per_episode
        time_out = self.episode_length_buf >= self.max_episode_length
        
        self.reset_buf = crash | success_all | time_out
        self.crash_condition = crash
        self.success_condition = success_all

        if self.reset_buf.any():
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励与观测 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        obs_list = []
        has_obstacles = self.obstacle_pos_tensor.shape[0] > 0
        sensing_radius = self.env_cfg.get("sensing_radius", 3.0)
        
        # Drone 0 Obs
        i = 0
        base_obs = torch.cat([
            torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),
            self.base_quat[:, i, :],
            torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),
            self.last_actions[:, i*4:(i+1)*4],
        ], dim=-1)
        
        nearest_obs_vecs = torch.ones((self.num_envs, self.num_nearest_obstacles * 3), device=gs.device)
        
        if has_obstacles:
            curr_drone_pos = self.base_pos[:, i, :]
            obs_pos_expanded = self.obstacle_pos_tensor.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
            obs_pos_expanded[:, :, 2] = curr_drone_pos[:, 2].unsqueeze(1) 
            vecs = obs_pos_expanded - curr_drone_pos.unsqueeze(1) 
            dists = torch.norm(vecs, dim=-1) 
            
            out_of_range_mask = dists > sensing_radius
            dists_masked = dists.clone()
            dists_masked[out_of_range_mask] = float('inf')
            
            k = min(self.num_nearest_obstacles, self.obstacle_pos_tensor.shape[0])
            sorted_dists, indices = torch.topk(dists_masked, k, dim=1, largest=False)
            
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
            topk_vecs = torch.gather(vecs, 1, indices_expanded)
            topk_vecs = topk_vecs * self.obs_scales["obstacle"]
            topk_vecs = torch.clip(topk_vecs, -1, 1)
            
            valid_mask = (sorted_dists < sensing_radius).unsqueeze(-1).float()
            final_vecs = topk_vecs * valid_mask + 1.0 * (1.0 - valid_mask)
            flat_obs = final_vecs.reshape(self.num_envs, -1)
            
            if k < self.num_nearest_obstacles:
                nearest_obs_vecs[:, :k*3] = flat_obs
            else:
                nearest_obs_vecs = flat_obs

        drone_obs = torch.cat([base_obs, nearest_obs_vecs], dim=-1)
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

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # 位置重置
        init_pos = torch.tensor([0.0, 0.0, 0.8], device=gs.device) 
        for i, drone in enumerate(self.drones):
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
        self.min_obstacle_dist[envs_idx] = 10.0

        # === 重置悬停计时器和计数器 ===
        self.target_hold_timer[envs_idx] = 0 
        
        self.extras["episode"] = {}
        self.extras["episode"]["goals_reached"] = torch.mean(self.goals_reached_count[envs_idx].float()).item()
        self.goals_reached_count[envs_idx] = 0 
        
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # === 采样第一个目标 ===
        self._sample_valid_commands(envs_idx, self.base_pos[envs_idx, 0, :])
        self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ==================== 奖励函数 ====================
    def _reward_target(self):
        curr_dist = torch.norm(self.rel_pos[:, 0, :], dim=1)
        last_dist = torch.norm(self.last_rel_pos[:, 0, :], dim=1)
        
        target_rew = last_dist - curr_dist
        
        # 只有当悬停成功(just_reached=True)时，才给大的奖励
        target_rew[self.just_reached] += 5.0 
        
        return target_rew

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, :, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return torch.mean(yaw_rew, dim=1)

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=2)
        return torch.mean(angular_rew, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1.0
        return crash_rew

    def _reward_obstacle(self):
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew

        d = self.min_obstacle_dist[:, 0]
        safe_dist = self.obstacle_safe_distance
        mask = d < safe_dist
        if mask.any():
            obstacle_rew[mask] += (safe_dist - d[mask]) / safe_dist

        return obstacle_rew

    def _reward_alive(self):
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0.0
        return alive_rew

    def _reward_progress(self):
        target_vec = self.rel_pos[:, 0, :] 
        target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)

        vel_vec = self.base_lin_vel[:, 0, :]
        speed = torch.norm(vel_vec, dim=1, keepdim=True)
        vel_dir = vel_vec / (speed + 1e-6)
        dot_prod = torch.sum(target_dir * vel_dir, dim=1)
        
        mask_static = speed.squeeze(-1) < 0.05
        dot_prod[mask_static] = 0.0

        return dot_prod