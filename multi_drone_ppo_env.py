"""
多无人机避障路径规划环境 - PPO版本
与MAPPO不同，PPO将所有无人机视为一个整体：
- 观测：拼接所有无人机的状态 + 每架无人机能感知其他无人机的相对位置
- 动作：所有无人机的电机控制拼接在一起
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


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MultiDronePPOEnv:
    """多无人机PPO环境 - 每架无人机可感知其他无人机"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.num_drones = env_cfg.get("num_drones", 3)
        self.rendered_env_num = min(5, self.num_envs)
        
        # PPO关键：总观测和总动作维度
        self.num_obs_per_drone = obs_cfg["num_obs_per_drone"]  # 23维（含其他无人机信息）
        self.num_obs = obs_cfg["num_obs"]  # 69维 = 23 * 3
        self.num_privileged_obs = None
        self.num_actions_per_drone = env_cfg["num_actions"]  # 4
        self.num_actions = self.num_actions_per_drone * self.num_drones  # 12
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])


        # ==================== 创建仿真场景 ====================
        # 根据是否需要可视化来配置场景
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
            # 无头模式：禁用所有可视化，避免 EGL/OpenGL 错误
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
        
        drone_colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        for i in range(self.num_drones):
            drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
            self.drones.append(drone)

        # 目标点可视化
        self.targets = []
        if env_cfg.get("visualize_target", False):
            for i, goal_pos in enumerate(self.drone_goal_positions):
                target = self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj", scale=0.08, pos=goal_pos,
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=drone_colors[i % len(drone_colors)]),
                    ),
                )
                self.targets.append(target)

        # ==================== 添加录制相机（必须在 build 之前）====================
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

        # 每架无人机的状态
        self.base_pos = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, self.num_drones, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        """设置各无人机的目标位置"""
        for i, goal_pos in enumerate(self.drone_goal_positions):
            self.commands[envs_idx, i, 0] = goal_pos[0]
            self.commands[envs_idx, i, 1] = goal_pos[1]
            self.commands[envs_idx, i, 2] = goal_pos[2]

    def step(self, actions):
        """执行一步仿真"""
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # 为每架无人机设置动作
        for i, drone in enumerate(self.drones):
            start_idx = i * self.num_actions_per_drone
            end_idx = start_idx + self.num_actions_per_drone
            drone_actions = self.actions[:, start_idx:end_idx]
            drone.set_propellels_rpm((1 + drone_actions * 0.8) * 14468.429183500699)

        self.scene.step()
        self.episode_length_buf += 1

        # 更新每架无人机的状态
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

        # 计算距离
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()


        # ==================== 终止条件 ====================
        crash_any = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        success_all = torch.ones((self.num_envs,), device=gs.device, dtype=torch.bool)
        
        for i in range(self.num_drones):
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance)
            )
            crash_any = crash_any | drone_crash
            
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            success_all = success_all & drone_success

        drone_collision = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        crash_any = crash_any | drone_collision

        self.crash_condition = crash_any
        self.success_condition = success_all
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash_any | success_all

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ==================== 构建观测（PPO关键：包含其他无人机信息）====================
        obs_list = []
        for i in range(self.num_drones):
            # 基础观测：17维
            base_obs = torch.cat([
                torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),  # 3
                self.base_quat[:, i, :],  # 4
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),  # 3
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),  # 3
                self.last_actions[:, i*4:(i+1)*4],  # 4
            ], dim=-1)
            
            # 其他无人机的相对位置：(num_drones-1) * 3 = 6维
            other_drones_rel_pos = []
            for j in range(self.num_drones):
                if i != j:
                    rel_to_other = (self.base_pos[:, j, :] - self.base_pos[:, i, :]) * self.obs_scales["rel_pos"]
                    other_drones_rel_pos.append(torch.clip(rel_to_other, -1, 1))
            
            # 拼接：17 + 6 = 23维
            drone_obs = torch.cat([base_obs] + other_drones_rel_pos, dim=-1)
            obs_list.append(drone_obs)
        
        # 所有无人机观测拼接：23 * 3 = 69维
        self.obs_buf = torch.cat(obs_list, dim=-1)
        self.last_actions[:] = self.actions[:]
        
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_min_obstacle_distance(self):
        """计算每架无人机与障碍物的最小距离"""
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
        """计算无人机之间的最小距离"""
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
        """重置指定环境"""
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
    def _reward_target(self):
        """目标奖励"""
        target_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        for i in range(self.num_drones):
            curr_dist = torch.norm(self.rel_pos[:, i, :], dim=1)
            last_dist = torch.norm(self.last_rel_pos[:, i, :], dim=1)
            dist_reduction = last_dist - curr_dist
            target_rew += dist_reduction * 10.0
            target_rew -= curr_dist * 0.1
            target_rew += torch.where(curr_dist < 2.0, torch.ones_like(curr_dist) * 2.0, torch.zeros_like(curr_dist))
            target_rew += torch.where(curr_dist < 1.0, torch.ones_like(curr_dist) * 5.0, torch.zeros_like(curr_dist))
            drone_at_target = curr_dist < self.env_cfg["at_target_threshold"]
            target_rew[drone_at_target] += 80.0  # v1:50
        target_rew[self.success_condition] += 500.0  # v1:300 从 300.0 提高到 500.0（更重视所有无人机都到达终点）
        return target_rew / self.num_drones

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_obstacle(self):
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew
        for i in range(self.num_drones):
            danger_dist = self.obstacle_safe_distance * 0.6
            close_mask = self.min_obstacle_dist[:, i] < danger_dist
            obstacle_rew[close_mask] -= (danger_dist - self.min_obstacle_dist[close_mask, i]) / danger_dist
        return obstacle_rew / self.num_drones

    def _reward_separation(self):
        sep_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        danger_dist = self.drone_safe_distance * 0.7
        close_mask = self.min_drone_dist < danger_dist
        sep_rew[close_mask] = -(danger_dist - self.min_drone_dist[close_mask]) / danger_dist
        return sep_rew

    def _reward_progress(self):
        progress_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        for i in range(self.num_drones):
            y_progress = self.base_pos[:, i, 1] - self.last_base_pos[:, i, 1]
            progress_rew += y_progress * 20.0
            height = self.base_pos[:, i, 2]
            height_good = (height > 0.4) & (height < 1.5)
            progress_rew += torch.where(height_good, torch.ones_like(height) * 1.0, -torch.ones_like(height) * 0.5)
            roll = torch.abs(self.base_euler[:, i, 0])
            pitch = torch.abs(self.base_euler[:, i, 1])
            stable = (roll < 30) & (pitch < 30)
            progress_rew += torch.where(stable, torch.ones_like(roll) * 0.5, -torch.ones_like(roll) * 1.0)
        return progress_rew / self.num_drones
    
    def _reward_alive(self):
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0
        return alive_rew

    def _reward_direction(self):
        """
        朝向目标方向飞行的奖励：
        - 鼓励沿“当前 → 目标”的直线方向前进（路径更短）
        - 惩罚垂直于目标方向的“横向偏移”（减少绕远路）
        实际上是在 xy 平面上做投影，适合从起点到终点之间“穿柱子”。
        """
        # 只在水平面 (x, y) 上考虑方向
        dir_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        for i in range(self.num_drones):
            # 指向目标的向量（当前 → 目标）
            rel_xy = self.rel_pos[:, i, :2]  # (B, 2)
            dist = torch.norm(rel_xy, dim=1) + 1e-6
            unit_dir = rel_xy / dist.unsqueeze(-1)  # 单位方向

            # 本步真实位移（世界坐标）
            step_xy = (self.base_pos[:, i, :2] - self.last_base_pos[:, i, :2])

            # 沿目标方向的分量（投影）——越大越好
            proj = torch.sum(step_xy * unit_dir, dim=1)  # (B,)

            # 垂直于目标方向的“横向分量”——越小越好
            lateral = step_xy - proj.unsqueeze(-1) * unit_dir
            lateral_norm = torch.norm(lateral, dim=1)

            # 组合奖励：
            #   正向投影 * 系数  -  横向偏移 * 系数
            #   并进行轻微截断，避免异常大步长带来数值不稳定
            dir_rew += torch.clamp(proj * 8.0, min=0.0, max=0.1)  # 只奖励“朝向目标”的前进
            dir_rew -= torch.clamp(lateral_norm * 4.0, min=0.0, max=0.1)

        # 对多架无人机取平均
        return dir_rew / self.num_drones