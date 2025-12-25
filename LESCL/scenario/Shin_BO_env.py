import gymnasium as gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import OrderedDict
from typing import Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import random
from scipy.special import j0  # Bessel function of the first kind, 0th order
from scipy.integrate import quad, dblquad
import math

UWOC_DEFAULT_C_ATTENUATION = 0.056    # 衰减系数 (吸收+散射)
UWOC_DEFAULT_P_TX_OP = 1
class BOEnv(gym.Env):
    """
    角度统一使用角度制表示
    状态：
    ideal_pointing_angle -- 理想的指向角（3维向量表示）,
    theta_divergence -- 当前光束的发散角，
    theta_azimuth -- 当前指向角的朝向角,
    theta_elevation -- 当前指向角的仰角,
    theta_azimuth_error -- 理想的指向角分解出来的朝向角与当前指向角的朝向角的误差,
    theta_elevation_error -- -- 理想的指向角分解出来的仰角与当前指向角的仰角的误差,
    snr -- 信噪比
    """

    def __init__(self,
                 max_orientation_angle: float = 180.0,  # 最大指向角 (单位：度)，包括方位角和仰角
                 max_position: float = 100.0,  # 最大位置距离 (单位：米)
                 delta_orientation_angle: float = 1,  # 指向角度变化步长 (单位：度)
                 max_steps: int = 256,  # 每个回合的最大步数
                 render_mode: str = "human",  # 渲染模式，默认 "human"
                 P_tr_op=1.0,
                 seed=None,
                 dataPath=None):
        super().__init__()

        self.global_seed = seed
        if self.global_seed:
            random.seed(self.global_seed)
        else:
            random.seed()

        # 初始化各个参数
        self.max_orientation_angle = max_orientation_angle  # 最大指向角度
        self.max_position = max_position  # 最大位置
        self.delta_orientation_angle = delta_orientation_angle  # 指向角度变化步长

        self.max_steps = max_steps  # 最大步数
        self.render_mode = render_mode  # 渲染模式
        self.P_tr_op = P_tr_op  # 发射功率

        # 初始化当前步数
        self.current_step = 0
        self.time_step = 0

        # 创建 UWOC 对象
        self.uwoc = UWOC(P_tx_op=UWOC_DEFAULT_P_TX_OP,c=UWOC_DEFAULT_C_ATTENUATION)
        self.sum_snr = None
        self.c_snr =None
        self.c_ber = None
        self.sum_ber = None
        # 观测空间，表示环境中每个维度的状态范围
        # 低值（low）和高值（high）表示各维度的最小值和最大值
        # d 0~5 a 0~360 e 0~90
        # [theta_e, ca, ce, ia, ie]
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0],
                dtype=np.float32),
            high=np.array(
                [max_orientation_angle, max_orientation_angle, max_orientation_angle / 2, max_orientation_angle,
                 max_orientation_angle / 2],
                dtype=np.float32),
            dtype=np.float32)

        # 动作空间，表示每次可采取的动作：方位角和仰角的变化量
        self.action_space = spaces.Discrete(9)
        # print(self.action_space)
        # 接收位置，固定为 [0, 0, 0]，表示挂在UAV上悬停在水中的多模节点，但会受到海流扰动
        self.receive_position_init = [0, 0, -1]
        self.receive_position = np.zeros(3, dtype=np.float32)

        # 发射端的移动参数
        self.transmit_movalbe = False  # 发射端是否可以移动
        self.transmit_position = None  # 发射端位置
        self.transmit_speed = 1  # 初始速度
        self.transmit_direction = np.random.uniform(-1, 1, size=3)  # 初始方向
        self.transmit_direction /= np.linalg.norm(self.transmit_direction)  # 归一化方向
        self.alpha = 0.8  # 平滑参数

        self.dt = 1  # 时间步长 一步表示1秒

        # 相对位置
        self.relative_position = None
        # 理想指向角
        self.ideal_pointing_angle = None
        # 误差角
        self.theta_error_deg = None
        self.sum_error_deg = None
        self.link_time =None
        # 状态初始化
        self.state = None
        self.count = None  # 计数器 记录误差角小于1度的连续时间长度
        # 动作粒度
        self.action_multiple_1 = 1
        self.stream_data = None

        def read_ocean_data(file_path):
            """
            从文件中读取海流数据。

            参数：
            - file_path: str, 数据文件路径。
            返回：
            - ocean_data: list, 读取后的海流数据，每行为 [深度, 流速(m/s), 流向(度)]。
            """
            ocean_data = []
            with open(file_path, 'r') as file:
                for line in file:
                    depth, speed, direction = map(float, line.split())
                    ocean_data.append([depth, speed, direction])
            return ocean_data

        if dataPath is not None:
            self.stream_data = read_ocean_data(dataPath)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """
        Reset the environment to a random state, and return the initial observation.
        """
        if self.global_seed is not None:
            current_seed = random.randint(0, 1e4)
            # print(f"current seed= {current_seed}")
            np.random.seed(current_seed)
            # print(f"jiyvseed {np.random.rand(3)}")
        else:
            np.random.seed()  # 不传种子时，会基于系统时间初始化种子
            # print(f"jiyushijian{np.random.rand(3)}")

        # Randomize the target position (receiver location) todo 更改为初始位置固定，但会根据海流噪声扰动产生位置变化
        self.receive_position = [0, 0, -5]

        self.transmit_position = np.array([
            np.random.uniform(-self.max_position / 5, self.max_position / 5),  # x in [-20, 20]
            np.random.uniform(-self.max_position / 5, self.max_position / 5),  # y in [-20, 20]
            np.random.uniform(-86, -84)  # z in [-86, -84]
        ], dtype=np.float32)

        self.transmit_movalbe = True
        if self.transmit_movalbe:
            # 重置发射端速度和方向
            self.transmit_speed = 0.2
            self.transmit_direction = np.random.uniform(-1, 1, size=3)
            self.transmit_direction /= np.linalg.norm(self.transmit_direction)
        else:
            pass

        self.relative_position = self.receive_position - self.transmit_position
        ideal_pointing_angle = self.relative_position / np.linalg.norm(self.relative_position)
        self.ideal_pointing_angle = ideal_pointing_angle  # 三维向量表示 ideal指向角
        ia, ie = np.degrees(np.arctan2(ideal_pointing_angle[1], ideal_pointing_angle[0])), np.degrees(
            np.arcsin(ideal_pointing_angle[2]))

        # 随机初始化 方位角、仰角
        azimuth_angle = np.random.uniform(0, 360)  # 方位角 [0, 360)
        elevation_angle = np.random.uniform(0, 90)  # 方位角 [0, 360)
        #elevation_angle = 90
        theta_error_deg = self.max_orientation_angle
        self.theta_error_deg = theta_error_deg
        # 初始化状态
        self.state = np.concatenate((
            np.array([theta_error_deg, azimuth_angle, elevation_angle, ia, ie], dtype=np.float32),  # ideal指向角
        ))
        self.sum_snr = 0
        self.c_snr = 0
        self.sum_ber = 0
        self.c_ber = 0
        self.sum_error_deg = 0
        self.count = 0
        self.link_time = 0
        # 重置当前步数
        self.current_step = 0

        # 返回状态和信息字典
        return self.state, {}

    def switch_action(self, action):
        up, keep, dowm = 1, 0, -1
        if action == 0:
            return up, up
        elif action == 1:
            return up, keep
        elif action == 2:
            return up, dowm
        elif action == 3:
            return keep, up
        elif action == 4:
            return keep, keep
        elif action == 5:
            return keep, dowm
        elif action == 6:
            return dowm, up
        elif action == 7:
            return dowm, keep
        elif action == 8:
            return dowm, dowm

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """

        """

        def cal_node_pos(now_pos, data):
            _, speed, direction = data
            direction_rad = np.deg2rad(direction)  # 将流向从度转换为弧度
            delta_x = speed * np.cos(direction_rad)
            delta_y = speed * np.sin(direction_rad)
            delta_z = 0  # 假设深度固定为 -1
            return [now_pos[0] + delta_x, now_pos[1] + delta_y, now_pos[2] + delta_z]

        # 更新接收端位置（海流扰动）
        if self.stream_data is not None:
            self.receive_position = cal_node_pos(now_pos=self.receive_position_init,
                                                 data=self.stream_data[self.time_step % len(self.stream_data)])
        # 更新发射端位置（随机移动）
        if self.transmit_movalbe:
            self._update_transmit_position()
            self.relative_position = self.receive_position - self.transmit_position
            self.ideal_pointing_angle = self.relative_position / np.linalg.norm(self.relative_position)

        theta_error_deg, ca, ce, ia, ie = self.state  # 指向角误差 当前朝向角 当前仰角 相对朝向角 相对仰角

        # 动作解析：分别处理发散角、水平角和俯仰角

        azimuth_action, elevation_action = self.switch_action(action=action)

        # print(f"act:{azimuth_action}__{elevation_action}")

        def clip_e(angle):
            if angle <= 0:
                angle = 0
            elif angle > 90:
                angle = 180 - angle
            return angle

        # 根据动作更新水平角
        ca += azimuth_action * self.delta_orientation_angle
        ca = ca % 360
        # 根据动作更新俯仰角
        ce += elevation_action * self.delta_orientation_angle
        ce = clip_e(ce)

        # 计算指向误差角 相对方位角 相对仰角
        theta_error_deg = self.cal_angle_between_vectors(ideal_vector=self.relative_position, current_azimuth=ca,
                                                        current_elevation=ce)
        self.theta_error_deg = theta_error_deg

        ia, ie = np.degrees(np.arctan2(self.ideal_pointing_angle[1], self.ideal_pointing_angle[0])), np.degrees(
            np.arcsin(self.ideal_pointing_angle[2]))

        # 更新状态
        self.state = np.concatenate((
            np.array([theta_error_deg, ca, ce, ia, ie], dtype=np.float32),  # 理想指向角
        ))

        self.state = np.around(self.state, decimals=3)  # 小数点精度

        # 计算奖励：基于光束与目标对准程度
        reward = self._compute_reward()
        theta_d = 1.5
        if theta_error_deg < theta_d:
            self.link_time += 1
            distance = np.linalg.norm(self.relative_position)

            # print(delta)
            self.c_snr = self.uwoc.cal_SNR_dB(theta_pe=theta_error_deg,theta_d=theta_d,d=distance)
            self.c_ber = self.uwoc.cal_BER(snr=self.c_snr)

            self.sum_ber += self.c_ber
            self.sum_snr += self.c_snr
            self.sum_error_deg += self.theta_error_deg

        # 终止条件：连续10个时间步误差角小于1.5度
        if theta_error_deg <= 1.5:
            self.count += 1
        else:
            self.count = 0

        done = (self.current_step >= self.max_steps) or self.count >= 5

        # 增加当前步数
        self.current_step += 1
        self.time_step += 1

        # 返回新状态、奖励、终止状态、截断状态和附加信息
        return self.state, reward, done, False, {}

    def cal_angle_between_vectors(self, ideal_vector, current_azimuth, current_elevation):

        # 归一化理想向量
        ideal_vector = np.array(ideal_vector, dtype=np.float32)
        ideal_unit_vector = ideal_vector / np.linalg.norm(ideal_vector)

        # 根据方位角和仰角计算当前向量
        current_pointing_vector = np.array([
            np.cos(np.radians(current_elevation)) * np.cos(np.radians(current_azimuth)),
            np.cos(np.radians(current_elevation)) * np.sin(np.radians(current_azimuth)),
            np.sin(np.radians(current_elevation))
        ])
        # 计算点积
        dot_product = np.dot(current_pointing_vector, ideal_unit_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 限制范围避免精度误差
        # 计算夹角（弧度）
        angle_rad = np.arccos(dot_product)
        # 转换为角度
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _compute_reward(self) -> float:
        """
        Compute the reward based on the alignment of the beam with the target.
        A reward of 1 is given if the beam points directly at the receiver.
        """

        theta_error_deg, ca, ce, ia, ie = self.state
        d_r = 0
        # 计算误差（方位角和仰角分别处理）
        ae = abs(ia - ca)
        ee = abs(ie - ce)
        if ae > 180:
            ae = 360 - ae  # 转换到[0, 180]范围
        ae /= 90
        ee /= 30
        if ae < 3 and ee < 3:
            d_r = 1
        # 总奖励
        reward = - ae - ee + d_r
        # print(f"t={self.time_step} ae={-ae} ee={-ee}")
        return reward

    def _update_transmit_position(self):
        """
        更新发射端的位置。
        """
        # 更新速度和方向
        self.transmit_speed = self.alpha * self.transmit_speed + (1 - self.alpha) * np.random.normal(0.01, 0.02)
        new_direction = self._generate_new_direction(self.transmit_direction)
        self.transmit_direction = new_direction

        # 更新位置
        self.transmit_position += self.transmit_direction * self.transmit_speed * self.dt

        # 确保位置在边界范围内
        self.transmit_position = np.clip(self.transmit_position, -self.max_position, self.max_position)

    def _generate_new_direction(self, current_direction: np.ndarray) -> np.ndarray:
        """
        生成新的方向向量。
        """
        noise = np.random.normal(0, 0.1, size=3)  # 高斯噪声
        new_direction = current_direction + noise
        return new_direction / np.linalg.norm(new_direction)  # 归一化

    def update_render_state(self, azimuth, elevation):
        """更新状态并计算发送向量"""

        # 计算理想发射向量
        relative_position = self.receive_position - self.transmit_position
        ideal_vector = relative_position / np.linalg.norm(relative_position)

        # 计算由朝向角与俯仰角组成的发送向量
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        send_vector = np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad)
        ])

        return ideal_vector, send_vector

    def render(self) -> None:
        """
        Render the environment state (e.g., print the state).
        """
        if self.render_mode == "human":
            theta_error_deg, ca, ce, ia, ie = self.state

            # 启用交互模式
            plt.ion()

            # 检查是否已有图形窗口
            if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.set_xlim([-self.max_position / 2, self.max_position / 2])
                self.ax.set_ylim([-self.max_position / 2, self.max_position / 2])
                self.ax.set_zlim([-self.max_position, 0])
                self.ax.set_xlabel('X-axis', color='r')
                self.ax.set_ylabel('Y-axis', color='g')
                self.ax.set_zlabel('Z-axis', color='b')

            # 清除之前的绘图
            self.ax.cla()

            # 设置坐标轴范围
            self.ax.set_xlim([-self.max_position / 2, self.max_position / 2])
            self.ax.set_ylim([-self.max_position / 2, self.max_position / 2])
            self.ax.set_zlim([-self.max_position, 0])
            self.ax.set_xlabel('X-axis', color='r')
            self.ax.set_ylabel('Y-axis', color='g')
            self.ax.set_zlabel('Z-axis', color='b')

            # 绘制发射端和接收端位置
            self.ax.scatter(self.transmit_position[0], self.transmit_position[1], self.transmit_position[2],
                            color='r', label='Transmit')
            self.ax.scatter(self.receive_position[0], self.receive_position[1], self.receive_position[2],
                            color='b', label='Receive')

            # 计算理想发射向量和发送向量
            ideal_vector, send_vector = self.update_render_state(azimuth=ca, elevation=ce)

            # 绘制理想发射向量
            scale_factor = np.linalg.norm(self.receive_position - self.transmit_position)  # 放大倍数
            self.ax.quiver(self.transmit_position[0], self.transmit_position[1], self.transmit_position[2],
                           ideal_vector[0] * scale_factor, ideal_vector[1] * scale_factor,
                           ideal_vector[2] * scale_factor,
                           color='g', label='Ideal Beam Vector')

            # 绘制由朝向角与俯仰角组成的发送向量
            self.ax.quiver(self.transmit_position[0], self.transmit_position[1], self.transmit_position[2],
                           send_vector[0] * scale_factor, send_vector[1] * scale_factor, send_vector[2] * scale_factor,
                           color='y', label='Send Beam Vector')

            self.ax.legend(loc="upper left")
            plt.draw()
            plt.pause(0.01)  # 暂停一会，确保更新
            plt.ioff()

    def close(self) -> None:
        pass

class UWOC:
    # theta_pe  指向角误差   deg
    # theta_d   光束发散角   deg
    # d 为发射机与接收机的距离 m
    # c 衰减因此 m^-1
    # P_tx_op 发射端功率     w

    def __init__(self,
                 c: float,  # 衰减系数 c = a + b, 单位 m^-1
                 P_tx_op: float,  # 发射光功率 P_tx, 单位 W
                 Ar: float = 0.01,  # 接收面积 Ar, 单位 m^2
                 # APD 参数
                 M: float = 10,  # APD 增益
                 R: float = 0.75,  # APD 响应度 A/W
                 Nf: float = 0.5,  # APD 噪声系数
                 B: float = 5e9,  # 带宽 Hz
                 Id: float = 15e-9,  # 暗电流 A
                 T: float = 298,  # 温度 K
                 RL: float = 100  # 负载电阻 Ω
                 ):
        # 信道与发射参数
        self.c = c
        self.P_tx_op = P_tx_op
        self.Ar = Ar

        # APD 参数
        self.M = M
        self.R = R
        self.Nf = Nf
        self.B = B
        self.Id = Id
        self.Kb = 1.38e-23  # 波尔兹曼常数
        self.T = T
        self.RL = RL

        # 湍流混合分布参数
        self.omega = 0.4589
        self.lambda_ = 0.3449
        self.a = 1.0421
        self.b = 1.5768
        self.c_gg = 35.9424

    #  计算路径衰减
    def cal_Lp(self,d,theta_pe):
        """路径衰减 Lp = exp(-c d)"""
        if theta_pe >= 90:
            return 0
        Lp=np.exp(-self.c * d)
        return Lp

    #  计算几何衰减
    def cal_Lg(self,theta_pe,theta_d,d):
        """几何衰减 Lg，当 θ_pe≤θ_d 时计算，否则为 0"""
        if theta_pe > theta_d:
            Lg= 0.0
        θpe = math.radians(theta_pe)
        θd = math.radians(theta_d)
        Lg = (self.Ar * math.cos(θpe) / (2 * math.pi * d ** 2 * (1 - math.cos(θd))))
        return Lg

    #  计算湍流衰减
    def cal_Lt(self):
        """
        计算湍流衰减系数 h_t，使用指数-广义伽马混合分布 (EGG)

        返回：
        - h_t：湍流衰减因子（每次可变）
        """
        # 混合分布参数
        omega = 0.4589  # 混合权重
        lambda_ = 0.3449  # 指数分布参数
        a = 1.0421  # Gamma 分布 shape 参数
        b = 1.5768  # Gamma 分布 scale 参数
        c = 35.9424  # Generalized Gamma 的指数参数

        # 按照概率选择分布来源
        if np.random.rand() < omega:
            # 从指数分布采样
            Lt = np.random.exponential(scale=lambda_)
        else:
            # 从广义 Gamma 分布采样：我们使用 gamma 分布的幂变换近似
            # generalized_gamma ≈ (gamma_sample)^(1/c)
            gamma_sample = np.random.gamma(shape=a, scale=b)
            Lt = gamma_sample ** (1 / c)

        return Lt

    def cal_P_rx_op(self,theta_pe,theta_d,d):
        Lp = self.cal_Lp(d=d,theta_pe=theta_pe)
        Lg = self.cal_Lg(theta_pe=theta_pe,theta_d=theta_d,d=d)
        Lt = self.cal_Lt()
        P_rx_op = self.P_tx_op * Lp * Lg * Lt
        return P_rx_op

    def cal_SNR_linear(self, theta_pe:float,theta_d:float,d: float) -> float:
        """
        线性 SNR 计算：
        SNR = (M R P_rx)^2 / (I_S^2 + I_D^2 + I_T^2)
        """
        P_rx = self.cal_P_rx_op(theta_pe=theta_pe,theta_d=theta_d,d = d)
        Ip   = self.M * self.R * P_rx

        q    = 1.6e-19  # 电子电荷
        Is2  = 2 * q * Ip * self.M**2 * self.Nf * self.B
        Id2  = 2 * q * self.Id * self.M**2 * self.Nf * self.B
        It2  = 4 * self.Kb * self.T * self.B / self.RL
        noise2 = Is2 + Id2 + It2

        return (Ip**2 / noise2) if noise2 > 0 else 0.0

    def cal_SNR_dB(self, theta_pe:float,theta_d:float,d: float) -> float:
        """将线性 SNR 转换为 dB"""
        snr_lin = self.cal_SNR_linear(theta_pe= theta_pe,theta_d=theta_d,d=d)
        return 10 * math.log10(snr_lin) if snr_lin > 0 else -math.inf

    def cal_BER(self, snr: float) -> float:
        """BER = Q(√SNR_lin) = 0.5*erfc(√SNR_lin/√2)"""
        if snr <= 0:
            return 0.5
        return 0.5 * math.erfc(math.sqrt(snr) / math.sqrt(2))
