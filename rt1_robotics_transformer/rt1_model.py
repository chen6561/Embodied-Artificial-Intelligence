import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RT1Config:
    """
    教学版 RT-1 模型的统一超参数配置。

    原始 RT-1 论文的配置更大，也更偏工程化。
    这里仅保留理解整体流程所需的关键参数。
    """

    # 输入图像边长。为了简化实现，这里默认输入是方形图像。
    image_size: int = 128
    # 策略网络一次看到的历史帧数。
    sequence_length: int = 6
    # 每个动作维度离散化后的类别数。
    vocab_size: int = 256
    # 要预测的动作维度数。
    # 一个贴近 RT-1 风格的玩具设定可以理解为：
    # 机械臂位移/旋转 + 夹爪 + 底盘运动。
    num_action_dims: int = 11
    # 文本指令嵌入向量的维度。
    text_dim: int = 128
    # 视觉编码器、TokenLearner 和 Transformer 共用的隐藏维度。
    embed_dim: int = 256
    # 每一帧经过压缩后保留的视觉 token 数。
    tokens_per_frame: int = 8
    # 视觉编码器在压缩前输出的 token 数。
    visual_tokens_before_pool: int = 64
    # Transformer 层数。
    transformer_layers: int = 6
    # 每层 Transformer 的注意力头数。
    transformer_heads: int = 8
    # Transformer 内部的 dropout 比例。
    transformer_dropout: float = 0.1


class SinusoidalPositionEncoding(nn.Module):
    """
    标准 Transformer 正弦位置编码。

    为什么需要它：
    Transformer 看到的只是一个扁平 token 序列，如果不显式加入位置信息，
    那么第 1 帧和第 6 帧在模型看来只是“同一种 token 的不同排列”，
    时间顺序就丢掉了。
    """

    def __init__(self, dim: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, dim]
        return x + self.pe[:, : x.size(1)]


class FiLMBlock(nn.Module):
    """
    FiLM = Feature-wise Linear Modulation，按通道调制特征。

    这里把文本指令映射成每个通道的缩放和偏移，让视觉网络能根据任务
    动态强调不同的图像信息。这对应 RT-1 中“语言条件化视觉感知”的思想。
    """

    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(text_dim, channels * 2)

    def forward(self, x: Tensor, text_embedding: Tensor) -> Tensor:
        # x: [batch, channels, height, width]
        # text_embedding: [batch, text_dim]
        scale, shift = self.to_scale_shift(text_embedding).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class VisualEncoder(nn.Module):
    """
    一个带 FiLM 条件调制的小型 CNN。

    它在功能上对应 RT-1 中的 EfficientNet 图像 tokenizer，
    但这里故意保持简单，便于学习和二次实现。

    输入：
      images: [batch, 3, H, W]
      text_embedding: [batch, text_dim]

    输出：
      visual tokens: [batch, n_visual_tokens, embed_dim]

    阅读提示：
    这个模块不是输出一个全局 pooled 向量，而是输出一组局部视觉 token。
    原因是 RT-1 在 token 压缩之前，仍然需要保留一定的空间结构信息。
    """

    def __init__(self, config: RT1Config) -> None:
        super().__init__()
        hidden = config.embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, config.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
        )
        self.film = FiLMBlock(config.embed_dim, config.text_dim)
        side = int(math.sqrt(config.visual_tokens_before_pool))
        self.pool = nn.AdaptiveAvgPool2d((side, side))

    def forward(self, images: Tensor, text_embedding: Tensor) -> Tensor:
        # 第 1 步：用 CNN 提取局部视觉特征。
        x = self.stem(images)
        # 第 2 步：用 FiLM 把语言任务信息注入视觉特征。
        x = self.film(x, text_embedding)
        # 第 3 步：把空间网格池化到固定大小，保证 token 数恒定。
        x = self.pool(x)
        batch, channels, height, width = x.shape
        # 第 4 步：把二维空间网格展开成 token 序列。
        # 输出形状：[batch, height * width, channels]
        return x.flatten(2).transpose(1, 2).reshape(batch, height * width, channels)


class TokenLearner(nn.Module):
    """
    学习一组软注意力掩码，把大量视觉 token 压缩成少量 token。

    在 RT-1 里，这一步很关键，因为机器人控制要求推理足够快。
    如果把每一帧的所有视觉 token 都送入 Transformer，开销会很大；
    TokenLearner 用加权汇聚的方式保留关键信息，同时显著减少 token 数量。
    """

    def __init__(self, embed_dim: int, num_output_tokens: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_output_tokens),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [batch, n_tokens, embed_dim]
        #
        # self.score 会为每个输入 token、每个输出槽位打一个分数。
        # 转置后：
        #   attn_logits: [batch, num_output_tokens, n_tokens]
        attn_logits = self.score(tokens).transpose(1, 2)
        # 在原 token 维度上做 softmax，得到每个输出 token 对输入 token 的权重。
        attn = attn_logits.softmax(dim=-1)
        # 矩阵乘法实现加权池化：
        #   [batch, num_output_tokens, n_tokens] @ [batch, n_tokens, embed_dim]
        # -> [batch, num_output_tokens, embed_dim]
        return attn @ tokens


class RT1ActionHead(nn.Module):
    """
    最终动作分类头。

    它为每个动作维度预测一个离散词表分布。
    RT-1 的核心做法之一，就是把连续控制量离散化成 token 分类问题，
    而不是直接做连续值回归。
    """

    def __init__(self, config: RT1Config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.num_action_dims * config.vocab_size)
        self.num_action_dims = config.num_action_dims
        self.vocab_size = config.vocab_size

    def forward(self, hidden: Tensor) -> Tensor:
        # hidden: [batch, embed_dim]
        logits = self.proj(self.norm(hidden))
        # 返回形状：
        # [batch, num_action_dims, vocab_size]
        return logits.view(hidden.size(0), self.num_action_dims, self.vocab_size)


class RT1Model(nn.Module):
    """
    教学版 RT-1 风格模型。

    输入：
      images: [batch, time, 3, H, W]
      text_embedding: [batch, text_dim]

    输出：
      action_logits: [batch, action_dims, vocab_size]

    整体流程：
      1. 把每一帧图像编码成一组视觉 token。
      2. 用 TokenLearner 压缩每帧的 token。
      3. 把所有帧的 token 拼成一个时间序列。
      4. 用 Transformer 建模时间上下文。
      5. 取最后一个隐藏状态预测下一步动作。
    """

    def __init__(self, config: RT1Config) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = VisualEncoder(config)
        self.token_learner = TokenLearner(config.embed_dim, config.tokens_per_frame)
        self.frame_embedding = nn.Parameter(torch.randn(1, config.sequence_length, config.embed_dim))
        self.positional_encoding = SinusoidalPositionEncoding(
            config.embed_dim, max_length=config.sequence_length * config.tokens_per_frame
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.transformer_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.action_head = RT1ActionHead(config)

    def encode_frames(self, images: Tensor, text_embedding: Tensor) -> Tensor:
        """
        把一批图像序列编码成压缩后的 Transformer token。

        参数：
          images: [batch, time, 3, H, W]
          text_embedding: [batch, text_dim]

        返回：
          tokens: [batch, time * tokens_per_frame, embed_dim]
        """

        batch, time, channels, height, width = images.shape

        # 合并 batch 和 time 维度，让 CNN 能把每一帧当成独立图像处理。
        images = images.view(batch * time, channels, height, width)

        # 同一个轨迹里的所有帧共用同一条文本指令。
        # 把 [batch, text_dim] 扩展成 [batch * time, text_dim]。
        repeated_text = text_embedding.unsqueeze(1).expand(batch, time, -1).reshape(batch * time, -1)

        # 每一帧在压缩前都会得到一组较密的视觉 token。
        visual_tokens = self.visual_encoder(images, repeated_text)

        # 用 TokenLearner 把大量视觉 token 压缩成每帧固定的少量 token。
        learned_tokens = self.token_learner(visual_tokens)

        # 把时间维度恢复回来：
        # [batch * time, tokens_per_frame, embed_dim]
        # -> [batch, time, tokens_per_frame, embed_dim]
        learned_tokens = learned_tokens.view(batch, time, self.config.tokens_per_frame, self.config.embed_dim)

        # 加入可学习的帧嵌入，让不同时间步的 token 更容易区分。
        learned_tokens = learned_tokens + self.frame_embedding[:, :time].unsqueeze(2)

        # 再次展平成单个序列，送入 Transformer。
        return learned_tokens.view(batch, time * self.config.tokens_per_frame, self.config.embed_dim)

    def forward(self, images: Tensor, text_embedding: Tensor) -> Tensor:
        # 先把整段图像序列编码成压缩后的 token 序列。
        tokens = self.encode_frames(images, text_embedding)
        # 在自注意力前加入位置编码。
        tokens = self.positional_encoding(tokens)
        # 对整段时间上下文做建模。
        hidden = self.transformer(tokens)
        # 教学版简化：
        # 直接取最后一个 token 的隐藏状态作为动作预测摘要。
        last_state = hidden[:, -1]
        return self.action_head(last_state)


def discretize_action(actions: Tensor, vocab_size: int = 256) -> Tensor:
    """
    把 [-1, 1] 范围内的连续动作映射为整数 bin。

    例如：
      -1.0 -> 0
       0.0 -> 大约 vocab_size / 2
       1.0 -> vocab_size - 1
    """

    clipped = actions.clamp(-1.0, 1.0)
    scaled = (clipped + 1.0) * 0.5 * (vocab_size - 1)
    return scaled.round().long()


def undiscretize_action(tokens: Tensor, vocab_size: int = 256) -> Tensor:
    """
    `discretize_action` 的近似逆变换，用于把预测结果还原成更直观的连续值，
    方便调试和打印。
    """

    scaled = tokens.float() / max(vocab_size - 1, 1)
    return scaled * 2.0 - 1.0
