"""
多无人机避障路径规划环境
支持多架无人机同时在障碍物场景中进行路径规划训练
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
    """在指定范围内生成随机浮点数张量"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MultiDroneEnv:
    """多无人机避障路径规划环境"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.num_drones = env_cfg.get("num_drones", 3)  # 每个环境中的无人机数量
        self.rendered_env_num = min(5, self.num_envs)
        self.num_obs_per_drone = obs_cfg["num_obs_per_drone"]  # 单架无人机观测维度
        self.num_obs = self.num_obs_per_drone * self.num_drones  # 总观测维度
        self.num_privileged_obs = None
        self.num_actions_per_drone = env_cfg["num_actions"]  # 单架无人机动作维度
        self.num_actions = self.num_actions_per_drone * self.num_drones  # 总动作维度
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # 保存配置
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # ==================== 创建仿真场景 ====================
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
            show_viewer=show_viewer,
        )

        # 添加地面
        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加障碍物 ====================
        self.obstacles = []
        obstacle_positions = env_cfg.get("obstacle_positions", [])
        obstacle_radius = env_cfg.get("obstacle_radius", 0.12)
        obstacle_height = env_cfg.get("obstacle_height", 2.5)
        
        for pos in obstacle_positions:
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
            self.obstacles.append({
                "entity": obstacle,
                "pos": torch.tensor(pos, device=gs.device),
                "radius": obstacle_radius
            })
        
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)
        self.drone_safe_distance = env_cfg.get("drone_safe_distance", 0.5)  # 无人机间安全距离

        # ==================== 添加多架无人机 ====================
        self.drones = []
        self.drone_init_positions = env_cfg.get("drone_init_positions", [
            [-1.0, -2.5, 0.15],
            [0.0, -2.5, 0.15],
            [1.0, -2.5, 0.15],
        ])
        self.drone_goal_positions = env_cfg.get("drone_goal_positions", [
            [-1.0, 2.5, 0.15],
            [0.0, 2.5, 0.15],
            [1.0, 2.5, 0.15],
        ])
        
        # 无人机颜色（用于区分）
        drone_colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
        
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        for i in range(self.num_drones):
            drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
            self.drones.append(drone)

        # 添加目标点可视化
        self.targets = []
        if env_cfg.get("visualize_target", False):
            for i, goal_pos in enumerate(self.drone_goal_positions):
                target = self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj",
                        scale=0.08,
                        pos=goal_pos,
                        fixed=True,
                        collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=drone_colors[i % len(drone_colors)]),
                    ),
                )
                self.targets.append(target)

        # 添加录制相机
        if env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(6.0, 0.0, 4.0),
                lookat=(0, 0, 0.5),
                fov=50,
                GUI=True,
            )

        # 构建场景
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
        
        # 每架无人机的目标位置
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

        # MAPPO专用：分离的局部观测和全局观测
        self.local_obs = torch.zeros((self.num_envs, self.num_drones, self.num_obs_per_drone), 
                                      device=gs.device, dtype=gs.tc_float)
        self.global_obs_dim = self.num_obs + len(self.obstacles) * 2  # 全局观测包含所有无人机状态+障碍物信息
        self.global_obs = torch.zeros((self.num_envs, self.global_obs_dim), device=gs.device, dtype=gs.tc_float)

    def _resample_commands(self, envs_idx):
        """为指定环境设置各无人机的目标位置"""
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
            
            # 计算欧拉角
            self.base_euler[:, i, :] = quat_to_xyz(
                transform_quat_by_quat(
                    self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1),
                    self.base_quat[:, i, :],
                ),
                rpy=True,
                degrees=True,
            )
            
            # 计算速度
            inv_quat_i = inv_quat(self.base_quat[:, i, :])
            self.base_lin_vel[:, i, :] = transform_by_quat(drone.get_vel(), inv_quat_i)
            self.base_ang_vel[:, i, :] = transform_by_quat(drone.get_ang(), inv_quat_i)

        # 计算相对位置
        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.commands - self.base_pos

        # ==================== 计算距离 ====================
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()

        # ==================== 终止条件 ====================
        crash_any = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        success_all = torch.ones((self.num_envs,), device=gs.device, dtype=torch.bool)
        
        for i in range(self.num_drones):
            # 单架无人机的crash条件
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance)
            )
            crash_any = crash_any | drone_crash
            
            # 检测是否到达目标
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            success_all = success_all & drone_success

        # 无人机间碰撞
        drone_collision = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        crash_any = crash_any | drone_collision

        self.crash_condition = crash_any
        self.success_condition = success_all
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash_any | success_all

        # 重置
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ==================== 构建观测 ====================
        obs_list = []
        for i in range(self.num_drones):
            drone_obs = torch.cat([
                torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat[:, i, :],
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions[:, i*4:(i+1)*4],
            ], dim=-1)
            self.local_obs[:, i, :] = drone_obs  # MAPPO: 存储局部观测
            obs_list.append(drone_obs)
        
        self.obs_buf = torch.cat(obs_list, dim=-1)
        self.last_actions[:] = self.actions[:]
        
        # MAPPO: 构建全局观测（所有无人机状态 + 障碍物位置）
        obstacle_info = []
        for obs in self.obstacles:
            obstacle_info.append(obs["pos"][:2].unsqueeze(0).expand(self.num_envs, -1))
        if obstacle_info:
            obstacle_tensor = torch.cat(obstacle_info, dim=-1)
            self.global_obs = torch.cat([self.obs_buf, obstacle_tensor], dim=-1)
        else:
            self.global_obs = self.obs_buf
        
        self.extras["observations"]["critic"] = self.global_obs
        self.extras["observations"]["local"] = self.local_obs

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
        """返回观测，支持MAPPO的局部/全局观测分离"""
        self.extras["observations"]["critic"] = self.global_obs
        self.extras["observations"]["local"] = self.local_obs
        return self.obs_buf, self.extras
    
    def get_local_obs(self):
        """MAPPO: 获取每个智能体的局部观测 [num_envs, num_agents, obs_dim]"""
        return self.local_obs
    
    def get_global_obs(self):
        """MAPPO: 获取全局观测 [num_envs, global_obs_dim]"""
        return self.global_obs

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
        """目标奖励：密集引导信号版"""
        target_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        for i in range(self.num_drones):
            # 当前距离和上一步距离
            curr_dist = torch.norm(self.rel_pos[:, i, :], dim=1)
            last_dist = torch.norm(self.last_rel_pos[:, i, :], dim=1)
            
            # 1. 距离减小奖励（核心密集信号）
            dist_reduction = last_dist - curr_dist
            target_rew += dist_reduction * 10.0  # 放大距离减小奖励
            
            # 2. 距离惩罚（越远惩罚越大，提供持续梯度）
            target_rew -= curr_dist * 0.1
            
            # 3. 接近目标的阶段性奖励
            target_rew += torch.where(curr_dist < 4.0, torch.ones_like(curr_dist) * 0.5, torch.zeros_like(curr_dist))
            target_rew += torch.where(curr_dist < 3.0, torch.ones_like(curr_dist) * 1.0, torch.zeros_like(curr_dist))
            target_rew += torch.where(curr_dist < 2.0, torch.ones_like(curr_dist) * 2.0, torch.zeros_like(curr_dist))
            target_rew += torch.where(curr_dist < 1.0, torch.ones_like(curr_dist) * 5.0, torch.zeros_like(curr_dist))
            
            # 4. 到达目标大奖励
            drone_at_target = curr_dist < self.env_cfg["at_target_threshold"]
            target_rew[drone_at_target] += 50.0
        
        # 5. 全部到达目标额外奖励
        target_rew[self.success_condition] += 300.0
        
        return target_rew / self.num_drones

    def _reward_smooth(self):
        """动作平滑奖励"""
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_crash(self):
        """坠机惩罚"""
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_obstacle(self):
        """避障奖励（优化：只在非常接近时惩罚）"""
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew
        
        for i in range(self.num_drones):
            # 只在接近碰撞距离时才惩罚
            danger_dist = self.obstacle_safe_distance * 0.6
            close_mask = self.min_obstacle_dist[:, i] < danger_dist
            obstacle_rew[close_mask] -= (danger_dist - self.min_obstacle_dist[close_mask, i]) / danger_dist
        
        return obstacle_rew / self.num_drones

    def _reward_separation(self):
        """无人机间距奖励：保持安全距离"""
        sep_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        danger_dist = self.drone_safe_distance * 0.7
        close_mask = self.min_drone_dist < danger_dist
        sep_rew[close_mask] = -(danger_dist - self.min_drone_dist[close_mask]) / danger_dist
        return sep_rew

    def _reward_progress(self):
        """前进奖励：强化Y轴前进和高度保持"""
        progress_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        for i in range(self.num_drones):
            # 1. Y轴前进奖励（核心）- 放大奖励
            y_progress = self.base_pos[:, i, 1] - self.last_base_pos[:, i, 1]
            progress_rew += y_progress * 20.0  # 大幅增加前进奖励
            
            # 2. Y轴位置奖励（越往前越好）
            y_pos = self.base_pos[:, i, 1]
            progress_rew += (y_pos + 2.5) * 0.2  # 从-2.5开始，越大越好
            
            # 3. 高度保持奖励（必须飞起来）
            height = self.base_pos[:, i, 2]
            # 高度在0.5-1.2m之间给奖励
            height_good = (height > 0.4) & (height < 1.5)
            progress_rew += torch.where(height_good, torch.ones_like(height) * 1.0, -torch.ones_like(height) * 0.5)
            
            # 4. 速度方向奖励（鼓励向Y正方向飞）
            vel_y = self.base_lin_vel[:, i, 1]
            progress_rew += torch.clamp(vel_y, -0.5, 2.0) * 2.0
            
            # 5. 姿态稳定奖励（防止翻滚）
            roll = torch.abs(self.base_euler[:, i, 0])
            pitch = torch.abs(self.base_euler[:, i, 1])
            stable = (roll < 30) & (pitch < 30)
            progress_rew += torch.where(stable, torch.ones_like(roll) * 0.5, -torch.ones_like(roll) * 1.0)
        
        return progress_rew / self.num_drones
    
    def _reward_alive(self):
        """存活奖励：鼓励无人机保持飞行状态"""
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        # 如果坠机则无奖励
        alive_rew[self.crash_condition] = 0
        return alive_rew
