from __future__ import annotations

"""
基于交叉熵方法（CEM）的模型预测控制（MPC）工具类。

该模块主要实现两个功能：
1. 可被 ``scripts/run_mpc.py`` 导入，提供 ``CEMPlanner`` 核心规划器类
2. 也可直接作为独立调试入口执行：
   ``python wm/mpc.py --config ... --checkpoint ...``

实现采用简洁的状态式设计，便于理解、调试，并可扩展到后续的真实Isaac Lab环境中。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


# 当该文件被直接执行时，Python仅会将 ``wm/`` 添加到 ``sys.path``。
# 我们手动添加项目根目录，使得同级包（如 ``isaac_tasks`` 和 ``scripts``）无需设置PYTHONPATH即可导入
def _add_project_root_to_sys_path() -> Path:
    """
    定位项目根目录并将其插入Python模块搜索路径。
    这样可以避免用户在运行脚本前手动配置PYTHONPATH环境变量。

    返回:
        Path: 指向顶级项目根文件夹的绝对路径
    """
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


# 全局常量，存储项目根目录绝对路径，供跨模块引用
PROJECT_ROOT = _add_project_root_to_sys_path()

# 路径注入后导入项目内部模块
from isaac_tasks.wrappers import make_task
from scripts.common import load_config, set_seed
from wm.dynamics import WorldModel


class CEMPlanner:
    """
    基于交叉熵方法（CEM）的规划器，用于基于模型的滚动时域模型预测控制（MPC）。

    该规划器通过迭代随机优化来求解多步未来动作序列：
    1. 从高斯分布中采样大量候选完整长度动作序列
    2. 使用预训练的学习型世界模型前向推演每个候选轨迹
    3. 选择累积奖励分数最高的"精英"轨迹
    4. 使用精英样本重新拟合高斯分布的均值和标准差，用于下一轮迭代

    MPC滚动时域规则：优化完完整时域动作序列后，仅执行第一个动作。
    在下一个环境时间步，基于最新的真实观测重新运行完整规划，以修正模型预测偏差。
    目标应用场景：基于状态的Franka机械臂到达/拾取/推动等连续机器人控制任务。
    所有计算在torch.no_grad()下运行，禁用梯度计算以加速推理。
    """

    def __init__(
            self,
            model: WorldModel,
            action_dim: int,
            horizon: int,
            num_samples: int,
            num_elites: int,
            iterations: int,
            action_std: float,
            device: str = "cpu",
            success_bonus_weight: float = 0.1,
    ) -> None:
        """
        初始化CEM规划器实例，存储所有优化超参数和世界模型引用。

        参数:
            model: 预训练的基于状态的WorldModel实例，必须实现encode()和step_latent()接口
            action_dim: 单步连续机器人控制动作向量的维度
            horizon: MPC预测时域；单次轨迹推演中模拟的未来时间步总数
            num_samples: 单次CEM优化迭代中采样的候选动作序列总数
            num_elites: 用于更新高斯采样分布的高分精英轨迹数量
            iterations: 每次规划调用执行的采样-精英-重拟合优化总轮数
            action_std: 高斯动作采样分布的初始标准差
            device: 所有张量模拟和优化操作的Torch计算设备（"cpu"或"cuda"）
            success_bonus_weight: 轨迹评分中预测任务成功概率奖励的缩放系数

        抛出异常:
            ValueError: 若精英轨迹数量超过总采样轨迹数（无效配置）
        """
        if num_elites > num_samples:
            raise ValueError("精英轨迹数量（num_elites）必须小于或等于采样轨迹总数（num_samples）")

        # 绑定预训练的世界动力学模型，用于轨迹前向模拟
        self.model = model
        # 机器人控制动作向量的维度
        self.action_dim = action_dim
        # MPC前瞻规划窗口长度
        self.horizon = horizon
        # 每轮CEM迭代生成的随机动作序列总数
        self.num_samples = num_samples
        # 保留的高奖励精英轨迹数量，用于重新拟合采样高斯分布
        self.num_elites = num_elites
        # 每个规划请求执行的完整优化迭代轮数
        self.iterations = iterations
        # 动作高斯采样的初始探索噪声标准差
        self.action_std = action_std
        # 所有张量计算的硬件设备
        self.device = device
        # 轨迹评分中用于放大预测任务完成概率的权重因子
        self.success_bonus_weight = success_bonus_weight

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """
        核心公共规划入口：基于当前环境观测计算最优单步动作。
        完全禁用自动求导计算图，以节省内存并加速推理速度。

        参数:
            obs: 原始环境状态观测张量，支持两种有效形状：
                1维: [obs_dim]（单条非批处理状态向量）
                2维: [1, obs_dim]（单批次维度包装的状态）

        返回:
            torch.Tensor: 优化后的单步机器人控制动作张量，形状为[action_dim]，转移到CPU内存
        """
        # 将观测张量形状标准化为批处理格式[1, obs_dim]并移至计算设备
        obs = self._normalize_obs_shape(obs).to(self.device)
        # 将原始状态编码为隐空间表示（对于纯基于状态的世界模型，直接使用原始状态）
        latent = self.model.encode(obs)

        # 初始化完整时域动作序列的高斯分布均值
        # 张量形状: [horizon, action_dim]，初始猜测为全零动作序列
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        # 使用固定的初始探索噪声值初始化高斯分布标准差
        std = torch.full_like(mean, fill_value=self.action_std)

        # CEM迭代优化主循环：采样轨迹 → 评分 → 选择精英 → 更新分布
        for _ in range(self.iterations):
            # 从当前高斯分布生成一批候选动作序列
            # 广播均值/标准差以匹配样本数量维度，并添加高斯噪声
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=self.device,
            )
            # 计算每个采样轨迹的奖励+成功奖励组合分数
            scores = self._score_action_sequences(latent, samples)
            # 获取得分最高的K个精英轨迹索引
            elite_indices = torch.topk(scores, self.num_elites).indices
            # 从完整样本批次中分离出精英动作序列
            elites = samples[elite_indices]
            # 通过所有精英轨迹的逐元素平均更新分布均值
            mean = elites.mean(dim=0)
            # 从精英轨迹统计量更新分布标准差
            # 将最小标准差限制为1e-3，防止探索噪声消失（局部最优停滞）
            std = elites.std(dim=0).clamp_min(1e-3)

        # MPC滚动时域规则：仅返回优化后的完整轨迹的第一个动作
        # 将张量从计算图中分离，并转移到CPU后返回
        return mean[0].detach().cpu()

    def _normalize_obs_shape(self, obs: torch.Tensor) -> torch.Tensor:
        """
        内部辅助函数，将观测张量形状统一为标准化的批处理格式[1, obs_dim]。
        在保持灵活的公共API输入的同时，满足WorldModel encode()的固定输入形状要求。

        参数:
            obs: 原始输入观测张量，可为1维单状态或2维单批次状态

        返回:
            torch.Tensor: 标准化的批处理观测张量，形状为[1, obs_dim]

        抛出异常:
            ValueError: 若输入张量维度或批次大小不符合支持的格式
        """
        if obs.ndim == 1:
            # 将1维非批处理观测扩展为2维单批次格式
            return obs.unsqueeze(0)
        if obs.ndim == 2 and obs.shape[0] == 1:
            # 已匹配所需的批处理形状，直接返回
            return obs
        # 拒绝不支持的张量形状
        raise ValueError(f"期望的obs形状为[obs_dim]或[1, obs_dim]，实际得到 {tuple(obs.shape)}")

    @torch.no_grad()
    def evaluate_action_chunk(self, obs: torch.Tensor, action_chunk: torch.Tensor) -> dict[str, float]:
        """
        独立的轨迹评估工具，用于评估外部生成的动作序列（如ACT、Diffusion Policy）。
        使用世界模型模拟固定的预定义多步动作序列，并返回聚合评分指标。
        适用于比较外部策略轨迹与CEM优化轨迹，无需完整的CEM采样循环。

        参数:
            obs: 当前原始环境观测张量（支持[obs_dim]或[1, obs_dim]格式）
            action_chunk: 固定动作序列张量，形状为[T, action_dim]，T为轨迹长度

        返回:
            dict[str, float]: 轨迹评估聚合指标字典：
                predicted_return: 轨迹上所有稠密步奖励的总和
                predicted_success_score: 每步预测任务成功概率的平均值
                combined_score: 与CEM轨迹评分函数匹配的加权总目标值

        抛出异常:
            ValueError: 若action_chunk张量维度或动作维度与规划器配置不匹配
        """
        # 标准化观测形状并转移到计算设备
        obs = self._normalize_obs_shape(obs).to(self.device)
        latent = self.model.encode(obs)
        actions = action_chunk.to(self.device)

        # 验证动作序列张量形状一致性
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"期望的action_chunk形状为[T, {self.action_dim}]，实际得到 {tuple(actions.shape)}"
            )

        total_reward = 0.0
        total_success = 0.0
        # 通过世界模型逐步前向推演固定动作序列
        for step in range(actions.shape[0]):
            wm_out = self.model.step_latent(latent, actions[step: step + 1])
            latent = wm_out.next_latent
            # 累积稠密奖励总和，将张量值移至CPU并转换为浮点数
            total_reward += float(wm_out.reward.squeeze().cpu())
            # 将成功对数几率转换为概率并累积总成功分数
            total_success += float(torch.sigmoid(wm_out.success_logit).squeeze().cpu())

        # 将聚合评估指标打包到输出字典
        return {
            "predicted_return": total_reward,
            "predicted_success_score": total_success / max(actions.shape[0], 1),
            "combined_score": total_reward + self.success_bonus_weight * total_success,
        }

    def _score_action_sequences(self, latent: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        """
        CEM采样循环的内部评分核心：通过世界模型前向模拟评估每个候选动作轨迹。
        轨迹目标分数 = 累积预测稠密奖励 + 加权累积预测任务成功概率。
        该目标函数设计简洁可解释，适用于轻量级基于状态的机器人学项目框架，易于修改和扩展。

        参数:
            latent: 所有轨迹推演的固定初始隐状态张量，形状为[1, obs_dim]
            action_sequences: 完整批次的采样候选轨迹，形状为[num_samples, horizon, action_dim]

        返回:
            torch.Tensor: 1维张量，存储每个采样轨迹的总目标分数，形状为[num_samples]
        """
        # 初始化零缓冲区，存储每个采样动作轨迹的总分
        scores = torch.zeros(action_sequences.shape[0], device=self.device)

        # 遍历完整样本批次中的每个候选轨迹
        for idx in range(action_sequences.shape[0]):
            # 克隆初始隐状态，避免修改所有轨迹共享的起始状态
            sim_latent = latent.clone()
            total_reward = 0.0
            success_bonus = 0.0

            # 单条候选轨迹的完整时域前向推演
            for step in range(self.horizon):
                action = action_sequences[idx, step: step + 1]
                # 调用世界模型单步动力学预测
                wm_out = self.model.step_latent(sim_latent, action)
                # 覆盖模拟隐状态，用于下一时间步预测
                sim_latent = wm_out.next_latent
                # 累积所有规划步骤的稠密奖励
                total_reward = total_reward + wm_out.reward.squeeze()
                # 累积经过sigmoid转换的成功概率奖励
                success_bonus = success_bonus + torch.sigmoid(wm_out.success_logit).squeeze()

            # 合并累积奖励和加权成功奖励作为最终轨迹分数
            scores[idx] = total_reward + self.success_bonus_weight * success_bonus

        return scores


def build_planner_from_checkpoint(config: dict[str, Any], checkpoint_path: str, device: str = "cpu") -> CEMPlanner:
    """
    统一的工厂辅助函数，从YAML配置和保存的世界模型检查点构建完全初始化的CEMPlanner。
    集中模型加载逻辑，消除独立的mpc.py和run_mpc.py脚本之间的重复代码。
    加载网络权重，将模型设置为评估推理模式，并使用配置超参数实例化CEM规划器。

    参数:
        config: 加载的YAML配置字典，包含任务、模型和MPC超参数
        checkpoint_path: 预训练WorldModel检查点文件（.pt）的绝对/相对路径
        device: 模型张量分配的目标Torch计算设备

    返回:
        CEMPlanner: 完全初始化、加载权重、可直接使用的MPC CEM规划器实例
    """
    # 加载序列化检查点文件，并将张量存储映射到目标计算设备
    payload = torch.load(checkpoint_path, map_location=device)

    # 使用配置中的维度超参数初始化空的WorldModel网络
    model = WorldModel(
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)
    # 从检查点载荷恢复训练好的网络权重
    model.load_state_dict(payload["model_state"])
    # 将模型切换到评估模式（禁用dropout、批归一化的训练行为）
    model.eval()

    # 使用从配置中提取的MPC超参数构建CEM规划器实例
    planner = CEMPlanner(
        model=model,
        action_dim=config["task"]["action_dim"],
        horizon=config["mpc"]["horizon"],
        num_samples=config["mpc"]["num_samples"],
        num_elites=config["mpc"]["num_elites"],
        iterations=config["mpc"]["iterations"],
        action_std=config["mpc"]["action_std"],
        device=device,
    )
    return planner


@torch.no_grad()
def run_mpc_episode(env: Any, planner: CEMPlanner) -> dict[str, float]:
    """
    执行由CEM-MPC规划器完全控制的完整环境回合。
    标准推演循环：重置环境 → 迭代规划并执行动作直到回合终止。
    收集核心回合统计信息，用于事后评估分析。

    参数:
        env: 通过make_task()生成的任务环境实例，必须实现reset()和step(action)接口
        planner: 完全配置、加载权重的CEMPlanner MPC控制器

    返回:
        dict[str, float]: 单回合评估聚合统计信息：
            episode_return: 整个回合收集的所有稠密奖励总和
            episode_length: 回合终止前执行的总时间步数
            success: 二进制浮点标志（1.0=任务成功完成，0.0=失败）
    """
    # 重置环境状态并随机化目标，初始化新回合
    obs = env.reset()
    done = False
    episode_return = 0.0
    final_info = {"success": False}
    episode_length = 0

    # 回合主循环：重复规划动作并与环境交互
    while not done:
        # 将numpy环境状态转换为torch张量，作为规划器输入
        action = planner.plan(torch.tensor(obs, dtype=torch.float32)).numpy()
        # 在真实环境中执行优化后的动作，并接收转移数据
        obs, reward, done, final_info = env.step(action)
        # 累积回合总奖励
        episode_return += float(reward)
        episode_length += 1

    # 返回打包的回合指标摘要
    return {
        "episode_return": episode_return,
        "episode_length": episode_length,
        "success": float(final_info.get("success", False)),
    }


def main() -> None:
    """
    独立命令行CLI入口，用于将mpc.py直接作为调试脚本执行。
    镜像scripts/run_mpc.py的完整执行流程，支持无需外部启动脚本的独立模块测试。

    示例执行命令：
        python wm/mpc.py --config configs/franka_reach_state.yaml \
            --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt

    执行流程：
        1. 解析命令行输入参数（配置路径、检查点路径、回合数、计算设备）
        2. 加载YAML配置文件，并设置全局随机种子以确保可复现性
        3. 通过工厂包装器make_task()实例化Isaac Lab Franka任务环境
        4. 构建并加载预训练世界模型 + 初始化CEM MPC规划器
        5. 运行指定数量的MPC控制评估回合
        6. 计算聚合统计摘要并打印最终评估结果
    """
    parser = argparse.ArgumentParser(description="使用训练好的世界模型运行MPC。")
    parser.add_argument("--config", required=True, help="YAML配置文件路径。")
    parser.add_argument("--checkpoint", required=True, help="训练好的世界模型检查点路径。")
    parser.add_argument("--episodes", type=int, default=20, help="要运行的评估回合数。")
    parser.add_argument("--device", default="cuda", help="使用的Torch设备，如cpu或cuda。")
    args = parser.parse_args()

    # 从YAML文件加载完整的任务/模型/MPC超参数配置
    config = load_config(args.config)
    # 固定torch/numpy的随机种子，确保确定性评估结果
    set_seed(config["seed"])
    # 根据配置规范创建Franka机械臂任务环境实例
    env = make_task(config)
    # 构建MPC规划器，并加载预训练的世界模型检查点
    planner = build_planner_from_checkpoint(config, args.checkpoint, device=args.device)

    # 初始化评估回合的统计累加器
    successes = 0
    returns = []
    lengths = []

    # 遍历所有评估回合
    for _ in range(args.episodes):
        episode_stats = run_mpc_episode(env, planner)
        successes += int(episode_stats["success"])
        returns.append(episode_stats["episode_return"])
        lengths.append(episode_stats["episode_length"])

    # 计算聚合评估摘要指标
    summary = {
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(args.episodes, 1),
        "mean_return": sum(returns) / max(len(returns), 1),
        "mean_episode_length": sum(lengths) / max(len(lengths), 1),
    }
    # 将最终聚合评估统计信息打印到控制台
    print(summary)


# 程序入口触发器：仅当该文件作为脚本直接运行时，执行main() CLI逻辑
if __name__ == "__main__":
    main()