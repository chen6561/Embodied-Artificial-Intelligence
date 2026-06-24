from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import Tensor


class RT2Tokenizer:
    """
    教学版 RT-2 统一 tokenizer。

    设计目标：
    1. 同一套词表同时容纳“普通文本 token”和“机器人动作 token”。
    2. 文本回答和动作输出都走同一个自回归生成接口。
    3. 保留 RT-2 的核心思想：把动作也写成 token，交给同一个 VLM 风格模型生成。

    这里是一个固定词表、按空格分词的简化实现。
    真正的 RT-2 使用的是更强的大模型 tokenizer 和更复杂的训练管线。
    """

    def __init__(self, num_action_dims: int = 8, action_bins: int = 64) -> None:
        self.num_action_dims = num_action_dims
        self.action_bins = action_bins

        # 一组最基础的特殊 token。
        self.special_tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
        ]

        # 教学版里会用到的文本词汇。
        # 这些词足以覆盖我们构造的 VQA 问题、机器人指令和可选的规划文本。
        self.text_tokens = sorted(
            {
                "answer",
                "action",
                "and",
                "above",
                "below",
                "blue",
                "color",
                "done",
                "find",
                "grasp",
                "green",
                "is",
                "left",
                "leftmost",
                "move",
                "no",
                "object",
                "pick",
                "plan",
                "question",
                "red",
                "right",
                "rightmost",
                "target",
                "task",
                "then",
                "the",
                "to",
                "what",
                "yes",
            }
        )

        # 为每个动作维度、每个离散 bin 构造一个唯一 token。
        # 例如：
        #   <act_0_12> 表示第 0 个动作维度取第 12 个离散桶
        self.action_tokens = [
            f"<act_{dim}_{bin_id}>"
            for dim in range(self.num_action_dims)
            for bin_id in range(self.action_bins)
        ]

        self.id_to_token = self.special_tokens + self.text_tokens + self.action_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.unk_token_id = self.token_to_id["<unk>"]

        # 预先把每个动作维度允许生成的 token id 收集起来。
        # 生成机器人动作时，我们会用这些列表做“受约束解码”。
        self.action_token_ids_by_dim: list[list[int]] = []
        for dim in range(self.num_action_dims):
            ids = [
                self.token_to_id[f"<act_{dim}_{bin_id}>"]
                for bin_id in range(self.action_bins)
            ]
            self.action_token_ids_by_dim.append(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _normalize(self, text: str) -> list[str]:
        """
        把输入文本做一个非常轻量的标准化。

        这里故意不做复杂分词，只保留教学版所需的最小功能：
        - 全部转小写
        - 去掉常见标点
        - 按空格切分
        """

        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9_<>\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        if not lowered:
            return []
        return lowered.split(" ")

    def encode_text(self, text: str, add_eos: bool = False) -> list[int]:
        """
        把普通文本编码成 token id 序列。
        """

        token_ids: list[int] = []
        for token in self._normalize(text):
            token_ids.append(self.token_to_id.get(token, self.unk_token_id))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode_text(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """
        把 token id 序列解码回文本。

        注意：
        动作 token 也在统一词表里，所以如果把动作 id 传进来，
        这里会直接返回 `<act_x_y>` 这样的字符串。
        """

        tokens: list[str] = []
        for idx in token_ids:
            if idx < 0 or idx >= len(self.id_to_token):
                continue
            token = self.id_to_token[idx]
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def discretize_action_values(self, actions: Tensor) -> Tensor:
        """
        把 [-1, 1] 范围的连续动作离散成 [0, action_bins - 1] 的整数。

        这一步对应 RT-1 / RT-2 的动作 token 化思路：
        连续控制不直接回归，而是先离散化，再做分类式生成。
        """

        clipped = actions.clamp(-1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * (self.action_bins - 1)
        return scaled.round().long()

    def undiscretize_action_values(self, discrete_actions: Tensor) -> Tensor:
        """
        把离散动作 id 近似还原成连续值，方便打印和调试。
        """

        scaled = discrete_actions.float() / max(self.action_bins - 1, 1)
        return scaled * 2.0 - 1.0

    def encode_continuous_action(self, actions: Tensor) -> list[int]:
        """
        把连续动作向量编码成动作 token id 列表。

        输入:
          actions: [num_action_dims]

        输出:
          [id(<act_0_bin>), id(<act_1_bin>), ...]
        """

        discrete = self.discretize_action_values(actions)
        token_ids: list[int] = []
        for dim, bin_id in enumerate(discrete.tolist()):
            token = f"<act_{dim}_{bin_id}>"
            token_ids.append(self.token_to_id[token])
        return token_ids

    def decode_action_token_ids(self, token_ids: Iterable[int]) -> Tensor:
        """
        把动作 token id 序列解析回近似连续动作。

        这是一个教学辅助函数，便于把模型输出的 token 重新看成动作值。
        """

        bins: list[int] = []
        pattern = re.compile(r"<act_(\d+)_(\d+)>")
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                continue
            token = self.id_to_token[token_id]
            matched = pattern.fullmatch(token)
            if matched is None:
                continue
            _, bin_id = matched.groups()
            bins.append(int(bin_id))

        if not bins:
            return torch.empty(0)

        discrete = torch.tensor(bins, dtype=torch.long)
        return self.undiscretize_action_values(discrete)

    def build_robot_target(
        self,
        actions: Tensor,
        plan_text: str | None = None,
    ) -> list[int]:
        """
        构造机器人任务的目标 token 序列。

        两种模式：
        1. 纯动作输出：
           <act_0_x> <act_1_y> ... <eos>
        2. 规划 + 动作输出：
           plan find red then grasp <act_0_x> ... <eos>

        这对应 RT-2 论文里提到的 Action Chain-of-Thought 风格扩展。
        """

        target_ids: list[int] = []
        if plan_text is not None:
            target_ids.extend(self.encode_text(plan_text, add_eos=False))
        target_ids.extend(self.encode_continuous_action(actions))
        target_ids.append(self.eos_token_id)
        return target_ids

    def robot_action_constraints(self, include_eos: bool = True) -> list[list[int]]:
        """
        返回机器人动作解码时每一步允许生成的 token id 集合。

        例如：
        第 0 步只能生成第 0 个动作维度对应的离散 token，
        第 1 步只能生成第 1 个动作维度对应的离散 token，
        ...
        最后一步允许生成 <eos>。
        """

        constraints = [list(ids) for ids in self.action_token_ids_by_dim]
        if include_eos:
            constraints.append([self.eos_token_id])
        return constraints
