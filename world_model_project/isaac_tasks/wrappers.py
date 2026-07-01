# 启用前向类型注解，支持在类内部使用自身类名做类型标注
from __future__ import annotations

# 导入数据类装饰器，自动生成__init__、__repr__、__eq__等基础方法，用于存储配置数据
from dataclasses import dataclass

# 导入NumPy数值计算库，用于数组运算、随机采样和数学相关计算
import numpy as np


@dataclass
class TaskSpec:
    """
    数据类，用于存储单个机器人任务环境的全部静态超参数与维度配置。
    将所有固定任务参数统一封装，方便传递与读取。
    """
    # 任务唯一标识字符串，例如 "pick_place"、"push_box"
    name: str
    # 观测向量总维度，env.reset()与env.step()返回的状态长度
    obs_dim: int
    # 动作向量维度，策略输出并传入环境step函数的控制指令维度
    action_dim: int
    # 目标向量维度，用于稀疏/稠密奖励的距离计算
    goal_dim: int
    # 单个回合允许的最大步数，达到该数值回合自动终止
    max_steps: int
    # 判断任务成功的欧氏距离阈值：当前状态与目标距离小于该值则判定成功
    success_threshold: float


class DummyIsaacTask:
    """
    简易仿真环境，模拟Isaac Sim机器人任务，用于离线强化学习算法测试。
    实现标准Gym环境接口：reset()、step(action)，额外提供专家脚本动作函数用于演示。
    模拟连续控制任务：智能体调整自身状态，逼近随机生成的目标点。
    """
    def __init__(self, spec: TaskSpec):
        """
        初始化虚拟仿真任务，传入预定义的任务配置参数。
        参数：
            spec: TaskSpec实例，包含任务全部维度、步数、成功阈值等超参
        """
        # 将任务配置绑定到环境实例
        self.spec = spec
        # 当前回合步数计数器，每次reset会清零
        self.step_count = 0
        # 智能体完整观测状态向量，初始化为全0，精度float32
        self.state = np.zeros(self.spec.obs_dim, dtype=np.float32)
        # 目标向量，每个回合重置后固定不变，智能体需要逼近该值
        self.goal = np.zeros(self.spec.goal_dim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        重置环境，开启全新回合：清零步数、随机初始化智能体状态与目标。
        目标对应状态维度会叠加高斯噪声，模拟初始时与目标存在偏移。
        返回：
            np.ndarray：新初始化的完整观测副本，形状[obs_dim]，数据类型float32
        """
        # 回合步数计数器重置为0，开启新轨迹
        self.step_count = 0
        # 在[-0.5, 0.5]均匀分布内随机生成完整观测状态
        self.state = np.random.uniform(-0.5, 0.5, size=self.spec.obs_dim).astype(np.float32)
        # 在更小范围[-0.3, 0.3]内随机生成目标向量
        self.goal = np.random.uniform(-0.3, 0.3, size=self.spec.goal_dim).astype(np.float32)
        # 在目标对应的状态维度叠加高斯噪声，制造智能体与目标的初始偏移
        self.state[: self.spec.goal_dim] = self.goal + np.random.normal(0.0, 0.1, size=self.spec.goal_dim)
        # 返回状态副本，防止外部代码原地修改环境内部真实状态
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        执行单步环境动力学，输入动作，计算奖励、终止标志与辅助信息。
        动力学规则：动作仅修改前action_dim维状态，带固定缩放系数更新位置。
        奖励为负欧氏距离（离目标越近，奖励值越高）。
        回合终止条件：到达目标 或 步数达到上限。
        参数：
            action: np.ndarray，策略输出的动作向量，形状[action_dim]，float32
        返回元组：
            1. np.ndarray：执行动作后的新观测状态
            2. float：当前单步稠密奖励
            3. bool：回合是否结束的终止标志
            4. dict：辅助信息字典，包含是否成功、目标向量、当前距离
        """
        # 步数计数器自增，代表执行一步交互
        self.step_count += 1
        # 将动作裁剪至[-1.0, 1.0]区间，限制控制信号范围
        clipped = np.clip(action, -1.0, 1.0)
        # 更新可控制维度的状态：缩放后的动作增量叠加到原有状态
        self.state[: self.spec.action_dim] = self.state[: self.spec.action_dim] + 0.05 * clipped
        # 计算状态目标维度与真实目标的L2欧氏距离
        distance = np.linalg.norm(self.state[: self.spec.goal_dim] - self.goal)
        # 稠密奖励：负距离，距离越小总回报越高
        reward = -float(distance)
        # 判断任务是否完成：距离小于设定成功阈值
        success = distance < self.spec.success_threshold
        # 回合终止：任务成功 或 达到最大步数限制
        done = success or self.step_count >= self.spec.max_steps
        # 打包诊断信息，用于日志打印、算法评估、回放缓存存储
        info = {"success": success, "goal": self.goal.copy(), "distance": distance}
        # 返回状态副本，隔离环境内部状态，避免外部篡改
        return self.state.copy(), reward, done, info

    def scripted_action(self, obs: np.ndarray) -> np.ndarray:
        """
        手写专家策略函数，生成近似最优的启发式动作。
        计算当前观测与目标的差值，放大后裁剪到合法动作区间输出。
        可用于采集专家演示数据，用于模仿学习、离线强化学习。
        参数：
            obs: np.ndarray，环境reset/step返回的完整观测向量
        返回：
            np.ndarray：专家启发动作，形状[action_dim]，值域[-1,1]，float32
        """
        # 初始化全零动作向量，维度与精度和任务配置匹配
        action = np.zeros(self.spec.action_dim, dtype=np.float32)
        # 计算差值：目标坐标 - 当前观测里的目标对应状态
        delta = self.goal - obs[: self.spec.goal_dim]
        # 差值放大4倍并裁剪到[-1,1]，赋值给动作前若干维度
        action[: self.spec.goal_dim] = np.clip(delta * 4.0, -1.0, 1.0)
        return action


def make_task(config: dict) -> DummyIsaacTask:
    """
    环境工厂构造函数：解析配置字典，生成TaskSpec并实例化虚拟Isaac环境。
    标准入口，从yaml/json读取配置后创建任务环境。
    参数：
        config: 嵌套字典，顶层key为"task"，存放全部任务超参
    返回：
        DummyIsaacTask：完整初始化完成的虚拟环境，可直接调用reset/step
    """
    # 从配置字典提取参数，实例化任务配置数据类
    spec = TaskSpec(
        name=config["task"]["name"],
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        goal_dim=config["task"]["goal_dim"],
        max_steps=config["task"]["max_steps"],
        success_threshold=config["task"]["success_threshold"],
    )
    # 用配置实例创建并返回虚拟仿真环境
    return DummyIsaacTask(spec)