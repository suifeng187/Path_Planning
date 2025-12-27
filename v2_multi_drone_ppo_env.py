"""
多无人机避障路径规划环境 - PPO版本 (优化版)
修改说明：
1. Progress奖励：改为目标方向速度投影，解耦坐标系。
2. Obstacle奖励：改为指数型势场惩罚。
3. Direction奖励：避障时豁免侧向惩罚。
4. Target奖励：移除阶梯函数，保持平滑。
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
    """多无人机PPO环境 - 每架无人机可感知其他无人机 + 最近的障碍物"""

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
        
        # 保存障碍物坐标的Tensor，用于计算感知
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
        # [新增] 将目标点转为 Tensor，用于并行计算奖励
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
        
        # 预先分配内存给 min_obstacle_dist，避免 step 中重复分配
        self.min_obstacle_dist = torch.zeros((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float)
        self.min_drone_dist = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

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
            drone.set_propellels_rpm((1 + drone_actions * 0.8) * 14468.429183500699)

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

        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()

        # ==================== 终止条件 ====================
        crash_any = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        
        for i in range(self.num_drones):
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance)
            )
            crash_any = crash_any | drone_crash
            
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            self.drone_ever_reached_target[:, i] = self.drone_ever_reached_target[:, i] | drone_success

        drone_collision = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        crash_any = crash_any | drone_collision

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

        # ==================== 构建观测 (含障碍物感知) ====================
        obs_list = []
        has_obstacles = self.obstacle_pos_tensor.shape[0] > 0
        
        for i in range(self.num_drones):
            # 1. 基础观测
            base_obs = torch.cat([
                torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat[:, i, :],
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions[:, i*4:(i+1)*4],
            ], dim=-1)
            
            # 2. 队友感知
            other_drones_rel_pos = []
            for j in range(self.num_drones):
                if i != j:
                    rel_to_other = (self.base_pos[:, j, :] - self.base_pos[:, i, :]) * self.obs_scales["rel_pos"]
                    other_drones_rel_pos.append(torch.clip(rel_to_other, -1, 1))
            
            # 3. 障碍物感知
            nearest_obs_vecs = torch.zeros((self.num_envs, self.num_nearest_obstacles * 3), device=gs.device)
            if has_obstacles:
                curr_drone_pos = self.base_pos[:, i, :]
                vecs = self.obstacle_pos_tensor.unsqueeze(0) - curr_drone_pos.unsqueeze(1) # (B, M, 3)
                dists = torch.norm(vecs, dim=-1) # (B, M)
                
                k = min(self.num_nearest_obstacles, self.obstacle_pos_tensor.shape[0])
                _, indices = torch.topk(dists, k, dim=1, largest=False) # (B, k)
                
                indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
                topk_vecs = torch.gather(vecs, 1, indices_expanded) # (B, k, 3)
                
                topk_vecs = topk_vecs * self.obs_scales["obstacle"]
                topk_vecs = torch.clip(topk_vecs, -1, 1)
                
                flat_obs = topk_vecs.reshape(self.num_envs, -1)
                if k < self.num_nearest_obstacles:
                    nearest_obs_vecs[:, :k*3] = flat_obs
                else:
                    nearest_obs_vecs = flat_obs

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
        for i in range(self.num_drones):
            drone_pos_xy = self.base_pos[:, i, :2]
            for obs in self.obstacles:
                obs_pos_xy = obs["pos"][:2].unsqueeze(0)
                dist = torch.norm(drone_pos_xy - obs_pos_xy, dim=1) - obs["radius"]
                min_dist[:, i] = torch.minimum(min_dist[:, i], dist)
        return min_dist

    def _get_min_drone_distance(self):
        min_dist = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * 100.0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = torch.norm(self.base_pos[:, i, :] - self.base_pos[:, j, :], dim=1)
                min_dist = torch.minimum(min_dist, dist)
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
        
        # 填充初始观测的占位符
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

   # ==================== 优化的奖励函数====================

    def _reward_target(self):
        """
        [修改] 目标奖励 - 精简版
        保留项：
        1. 靠近target奖励 (通过 dist_reduction 实现)
        2. 惩罚乱飞 (方向一致性检查)
        3. 单机到达奖励 (Section 5)
        4. 全员到达奖励 (Section 6)
        5.
        """
        target_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        for i in range(self.num_drones):
            curr_dist = torch.norm(self.rel_pos[:, i, :], dim=1)
            last_dist = torch.norm(self.last_rel_pos[:, i, :], dim=1)
            
            # --- 1. 靠近target奖励 (核心信号) ---
            # 奖励距离缩短量，提供最基础的导航梯度
            dist_reduction = last_dist - curr_dist
            target_rew += dist_reduction * 10.0 
            # target_rew -= curr_dist * 0.2
            
            # --- 2. 惩罚乱飞 (方向一致性检查) ---
            # 计算速度方向与目标方向的一致性
            to_target = self.drone_goal_pos_tensor[i] - self.base_pos[:, i, :]
            to_target_norm = torch.norm(to_target, dim=1, keepdim=True) + 1e-6
            target_dir = to_target / to_target_norm
            
            vel = self.base_lin_vel[:, i, :]
            vel_norm = torch.norm(vel, dim=1, keepdim=True) + 1e-6
            vel_dir = vel / vel_norm
            
            # 点积计算夹角余弦值
            direction_alignment = torch.sum(vel_dir * target_dir, dim=1)
            
            # 避障豁免：只在远离障碍物时检查方向，避免避障时被误判为乱飞
            is_far_from_obstacle = self.min_obstacle_dist[:, i] > (self.obstacle_safe_distance * 2.0)
            
            # 如果方向严重偏离 (夹角 > 72度) 且周围安全，则惩罚
            bad_direction = direction_alignment < 0.4 
            direction_penalty = torch.where(
                is_far_from_obstacle & bad_direction,
                torch.ones_like(direction_alignment) * 5.0, 
                torch.zeros_like(direction_alignment)
            )
            target_rew -= direction_penalty
            
            # --- 3. 单机到达奖励 (原Section 5) ---
            drone_at_target = curr_dist < self.env_cfg["at_target_threshold"]
            target_rew[drone_at_target] += 50.0 
            
            # --- 4. 所有无人机都到达的额外奖励 (原Section 6) ---
            target_rew[self.success_condition] += 150.0 
        
        return target_rew / self.num_drones

    def _reward_smooth(self):
        """
        [保持] 动作平滑奖励
        """
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_crash(self):
        """
        [实现] 撞击障碍物或地面给予惩罚
        注意：step中已经计算了 crash_condition (包含撞地、撞柱、姿态失控)
        此处返回 1.0，配合 config 中的 crash scale (负值) 形成惩罚
        """
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1.0
        return crash_rew

    def _reward_obstacle(self):
        """
        [实现] 线性避障惩罚
        进入障碍物危险距离给予惩罚，越近惩罚越大 (线性)
        """
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew

        for i in range(self.num_drones):
            d = self.min_obstacle_dist[:, i]
            safe_dist = self.obstacle_safe_distance # e.g. 0.25
            
            # 筛选出进入危险区域的
            mask = d < safe_dist
            
            if mask.any():
                # 线性公式: (safe - current) / safe
                # d=safe时为0，d=0时为1。配合 scale (负值) 形成惩罚。
                obstacle_rew[mask] += (safe_dist - d[mask]) / safe_dist

        return obstacle_rew / self.num_drones

    def _reward_separation(self):
        """
        [实现] 无人机之间距离过近给予惩罚
        """
        sep_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        danger_dist = self.drone_safe_distance * 0.7
        
        close_mask = self.min_drone_dist < danger_dist
        if close_mask.any():
             # 线性惩罚
            sep_rew[close_mask] = (danger_dist - self.min_drone_dist[close_mask]) / danger_dist
            
        return sep_rew

    def _reward_alive(self):
        """
        [保持] 存活奖励
        """
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0.0
        return alive_rew

    def _reward_progress(self):
        """
        [优化后] 仅保留核心动力学奖励：
        1. 推进奖励：速度在目标方向的投影 (vel_proj)
        2. 侧向约束：惩罚侧向漂移，但在避障时豁免
        3. 姿态约束：高度和倾角限制
        """
        progress_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        for i in range(self.num_drones):
            # --- 1. 计算目标方向向量 ---
            to_target = self.drone_goal_pos_tensor[i] - self.base_pos[:, i, :]
            dist = torch.norm(to_target, dim=1, keepdim=True) + 1e-6
            target_dir = to_target / dist
            
            # --- 2. 分解当前速度 ---
            vel = self.base_lin_vel[:, i, :]
            
            # [核心] 速度在目标方向上的投影 (Parallel Component)
            # 这是一个标量：正值代表朝目标飞，负值代表背离
            vel_proj = torch.sum(vel * target_dir, dim=1)
            
            # [核心] 计算侧向速度向量 (Lateral Component)
            # 原始速度 - 平行分量 = 垂直分量 (漂移速度)
            vel_lateral_vec = vel - vel_proj.unsqueeze(-1) * target_dir
            vel_lateral_norm = torch.norm(vel_lateral_vec, dim=1)
            
            # --- 3. 检测障碍物距离（用于动态调整权重）---
            obstacle_dist = self.min_obstacle_dist[:, i]
            safe_dist = self.obstacle_safe_distance
            
            # 定义两个阈值，形成缓冲区，避免硬切
            is_far_from_obstacle = obstacle_dist > (safe_dist * 2.5)  # 绝对安全区
            
            # --- 4. 推进奖励 (Velocity Projection) ---
            # 逻辑：只要速度朝向目标，就给奖励。
            # 改进：加入 clamp 防止奖励爆炸 (例如限制最大奖励对应 3m/s)
            base_progress = torch.clamp(vel_proj, max=3.0) * 0.8
            enhanced_progress = torch.clamp(vel_proj, max=3.0) * 0.8
            
            # 在远离障碍物时，给予更高的速度权重，鼓励全速前进
            progress_rew += torch.where(is_far_from_obstacle, enhanced_progress, base_progress)
            
            # --- 5. 智能侧向惩罚 (Smart Lateral Penalty) ---
            # 逻辑：平时严禁横移，但避障时允许横移
            lateral_penalty_base = vel_lateral_norm * 0.5
            
            # 计算过渡系数 (0.0 ~ 1.0)
            # 距离: <0.5倍安全距离 -> 系数0 (完全豁免)
            # 距离: >1.5倍安全距离 -> 系数1 (完全惩罚)
            transition_factor = torch.clamp(
                (obstacle_dist - safe_dist * 0.5) / (safe_dist * 1.0), 
                0.0, 1.0
            )
            
            # 应用过渡系数
            lateral_penalty = lateral_penalty_base * transition_factor
            progress_rew -= lateral_penalty
            
            # --- 6. 姿态与高度约束 (Constraints) ---
            # 逻辑：只惩罚违规行为 (负约束)
            height = self.base_pos[:, i, 2]
            
            # 高度违规惩罚 (太低或太高)
            height_bad = (height < 0.4) | (height > 1.2)
            progress_rew -= torch.where(height_bad, torch.ones_like(height) * 3, torch.zeros_like(height))
        
            # 姿态违规惩罚 (翻滚或俯仰角过大)
            roll = torch.abs(self.base_euler[:, i, 0])
            pitch = torch.abs(self.base_euler[:, i, 1])
            unstable = (roll > 60) | (pitch > 60)
            progress_rew -= torch.where(unstable, torch.ones_like(roll) * 1.0, torch.zeros_like(roll))
            
        return progress_rew / self.num_drones
    
    