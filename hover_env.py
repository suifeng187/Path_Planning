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


'''
    1. 初始化基础配置
    2. 搭建环境的地面、障碍物
    3. 添加无人机

'''
class HoverEnv:
    """无人机悬停任务的强化学习环境"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs  # 并行环境数量
        self.rendered_env_num = min(10, self.num_envs)  # 可视化环境数量上限
        self.num_obs = obs_cfg["num_obs"]  # 观测空间维度
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 动作空间维度（4个螺旋桨）
        self.num_commands = command_cfg["num_commands"]  # 目标指令维度（xyz坐标）
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # 是否模拟动作延迟
        self.dt = 0.01  # 仿真时间步长，100Hz，模拟真实世界的时间流逝
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)  # 最大回合步数

        # 保存配置
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]  # 观测值缩放系数
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])  # 奖励缩放系数

        # ==================== 创建仿真场景 ====================
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),  # 每步2个子步骤
            # 配置相机来录制视频
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),  # 相机位置
                camera_lookat=(0.0, 0.0, 1.0),  # 相机朝向
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,  # 牛顿约束求解器
                enable_collision=True,  # 启用碰撞检测
                enable_joint_limit=True,  # 启用关节限制
            ),
            show_viewer=show_viewer,
        )

        # 添加地面
        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加柱子障碍物 ====================
        self.obstacles = []
        obstacle_positions = env_cfg.get("obstacle_positions", [])
        obstacle_radius = env_cfg.get("obstacle_radius", 0.1)
        obstacle_height = env_cfg.get("obstacle_height", 2.0)
        
        for i, pos in enumerate(obstacle_positions):
            # 使用圆柱体作为障碍物
            obstacle = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=pos,
                    radius=obstacle_radius,
                    height=obstacle_height,
                    fixed=True,  # 固定障碍物
                    collision=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.3, 0.8)),  # 蓝色柱子
                ),
            )
            self.obstacles.append({"entity": obstacle, "pos": torch.tensor(pos, device=gs.device), "radius": obstacle_radius})
        
        # 保存障碍物配置
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.5)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.2)

        # 添加目标点可视化（红色小球）
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,  # 目标点不参与碰撞
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target = None

        # 添加相机
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(8.0, -3.0, 6.0),
                lookat=(0, 1.0, 0.5),
                fov=50,
                GUI=True,
            )

        # 添加无人机
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)  # 初始位置
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)  # 初始姿态四元数
        self.inv_base_init_quat = inv_quat(self.base_init_quat)  # 初始姿态的逆四元数
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf")) # 引入Genesis库中自带的无人机

        # 多个并行环境
        self.scene.build(n_envs=num_envs)

        # ==================== 初始化奖励函数 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt  # 奖励乘以时间步长，保证时间无关性
            self.reward_functions[name] = getattr(self, "_reward_" + name)  # 动态绑定奖励函数
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ==================== 初始化状态缓冲区 ====================
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)  # 观测
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # 奖励
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # 重置标志
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # 回合步数
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)  # 目标位置

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)  # 当前动作
        self.last_actions = torch.zeros_like(self.actions)  # 上一步动作

        # 无人机状态
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 位置
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)  # 姿态四元数
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 线速度（机体坐标系）
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 角速度（机体坐标系）
        self.last_base_pos = torch.zeros_like(self.base_pos)  # 上一步位置

        self.extras = dict()  # 额外日志信息
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        """为指定环境设置固定目标位置"""
        # 使用固定终点位置
        goal_pos = self.env_cfg.get("goal_position", [0.0, 2.0, 0.15])
        self.commands[envs_idx, 0] = goal_pos[0]
        self.commands[envs_idx, 1] = goal_pos[1]
        self.commands[envs_idx, 2] = goal_pos[2]

    def _at_target(self):
        """检测哪些环境的无人机已到达目标点"""
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        """执行一步仿真"""
        # 裁剪动作到有效范围
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # ==================== 设置螺旋桨转速 ====================
        # 14468.43 是悬停基准转速，动作值[-1,1]映射到转速的±80%调整
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)

        # 更新目标点位置（可视化用）
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # ==================== 更新状态缓冲区 ====================
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()  # 获取当前位置
        self.rel_pos = self.commands - self.base_pos  # 相对目标的位置差
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()  # 获取当前姿态

        # 将四元数转换为欧拉角（roll, pitch, yaw）
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        # 将速度从世界坐标系转换到机体坐标系
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # 检测到达目标（用于成功奖励，不重采样）
        self.reached_target = self._at_target()

        # ==================== 终止条件检测 ====================
        # 计算与障碍物的最小距离
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])  # 俯仰角过大
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])  # 横滚角过大
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])  # 太接近地面（非目标区域）
            | (self.min_obstacle_dist < self.obstacle_collision_distance)  # 碰撞障碍物
        )
        
        # 成功到达目标也触发重置（但不是crash）
        self.success_condition = torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition | self.success_condition

        # 记录超时信息
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # 重置需要重置的环境
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ==================== 构建观测向量 ====================
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),  # 相对位置
                self.base_quat,  # 姿态四元数
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),  # 线速度
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),  # 角速度
                self.last_actions,  # 上一步动作
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """获取当前观测"""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """获取特权观测（本环境未使用）"""
        return None

    def reset_idx(self, envs_idx):
        """重置指定索引的环境"""
        if len(envs_idx) == 0:
            return

        # 重置无人机位置和姿态
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)  # 清零所有速度

        # 重置缓冲区
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 记录回合统计信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # 为重置的环境采样新目标
        self._resample_commands(envs_idx)

    def reset(self):
        """重置所有环境"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ==================== 奖励函数 ====================
    def _reward_target(self):
        """目标奖励：鼓励无人机靠近目标点"""
        # 距离减小奖励
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        # 到达目标额外奖励
        target_rew[self.success_condition] += 50.0
        return target_rew

    def _reward_smooth(self):
        """平滑奖励：惩罚动作剧烈变化"""
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        """偏航奖励：惩罚偏航角偏离"""
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # 转换为弧度
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        """角速度奖励：惩罚过大的角速度"""
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        """坠机惩罚：触发终止条件时给予惩罚"""
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_obstacle(self):
        """避障奖励：距离障碍物越近惩罚越大"""
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        if len(self.obstacles) == 0:
            return obstacle_rew
        
        # 接近障碍物时的渐进惩罚
        close_mask = self.min_obstacle_dist < self.obstacle_safe_distance
        obstacle_rew[close_mask] = -(self.obstacle_safe_distance - self.min_obstacle_dist[close_mask]) / self.obstacle_safe_distance
        
        # 碰撞惩罚
        collision_mask = self.min_obstacle_dist < self.obstacle_collision_distance
        obstacle_rew[collision_mask] = -1.0
        
        return obstacle_rew

    def _get_min_obstacle_distance(self):
        """计算无人机与所有障碍物的最小距离（只考虑XY平面）"""
        if len(self.obstacles) == 0:
            return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * 100.0
        
        min_dist = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * 100.0
        drone_pos_xy = self.base_pos[:, :2]  # [num_envs, 2]
        
        for obs in self.obstacles:
            obs_pos_xy = obs["pos"][:2].unsqueeze(0)  # [1, 2]
            dist = torch.norm(drone_pos_xy - obs_pos_xy, dim=1) - obs["radius"]
            min_dist = torch.minimum(min_dist, dist)
        
        return min_dist
