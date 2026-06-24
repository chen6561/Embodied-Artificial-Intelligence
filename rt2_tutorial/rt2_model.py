from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RT2Config:
    """
    教学版 RT-2 模型超参数。

    这里刻意做了两层简化：
    1. 把真实的大型 VLM 换成一个小型多模态自回归 Transformer；
    2. 保留 RT-2 最关键的算法思想，而不追求规模和工程一致性。
    """

    image_size: int = 32
    num_frames: int = 3
    vocab_size: int = 512
    embed_dim: int = 256
    vision_hidden_dim: int = 128
    vision_pool_side: int = 4
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    max_text_tokens: int = 64


class RT2VisionEncoder(nn.Module):
    """
    把多帧图像压成视觉 token 序列。

    真实 RT-2 的底座来自大型视觉语言模型；
    这里我们用一个小型 CNN 来扮演“视觉 tokenizer”的角色。
    """

    def __init__(self, config: RT2Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Conv2d(3, config.vision_hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(config.vision_hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.vision_hidden_dim, config.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
            nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((config.vision_pool_side, config.vision_pool_side))
        self.frame_embedding = nn.Parameter(
            torch.randn(1, config.num_frames, config.embed_dim) * 0.02
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        输入:
          images: [batch, time, 3, H, W]

        输出:
          image_tokens: [batch, time * pooled_tokens_per_frame, embed_dim]
        """

        batch, time, channels, height, width = images.shape
        x = images.reshape(batch * time, channels, height, width)
        x = self.backbone(x)
        x = self.pool(x)

        # 把空间网格展开成 token 序列。
        x = x.flatten(2).transpose(1, 2)
        tokens_per_frame = x.size(1)

        # 恢复时间维度，并加入帧嵌入。
        x = x.reshape(batch, time, tokens_per_frame, self.config.embed_dim)
        x = x + self.frame_embedding[:, :time].unsqueeze(2)

        # 最后再展平成一条视觉 token 序列。
        return x.reshape(batch, time * tokens_per_frame, self.config.embed_dim)


class RT2Model(nn.Module):
    """
    教学版 RT-2。

    核心思想：
    - 图像是视觉前缀
    - 文本 prompt 和输出 target 共享同一语言 token 空间
    - 机器人动作也编码成 token，因此模型既能“回答问题”，也能“输出动作”
    """

    def __init__(self, config: RT2Config) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = RT2VisionEncoder(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # 学习式位置编码和模态编码。
        # 这不是 RT-2 论文里的唯一选择，但非常适合教学版实现。
        max_image_tokens = config.num_frames * (config.vision_pool_side ** 2)
        self.max_total_tokens = max_image_tokens + config.max_text_tokens
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_total_tokens, config.embed_dim) * 0.02
        )
        self.image_modality_embedding = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.text_modality_embedding = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.transformer_layers,
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    def _build_prefix_lm_mask(
        self,
        num_image_tokens: int,
        num_text_tokens: int,
        device: torch.device,
    ) -> Tensor:
        """
        构造 RT-2 风格“视觉前缀 + 文本自回归”注意力掩码。

        设计原则：
        1. 图像 token 之间彼此可见；
        2. 图像 token 不需要看文本 token；
        3. 文本 token 可以看到所有图像 token 和它之前的文本 token；
        4. 文本 token 不能看到未来文本 token。

        这样就得到一个 prefix-LM 结构：
        图像是前缀，上面的语言/动作 token 是自回归生成目标。
        """

        total = num_image_tokens + num_text_tokens
        mask = torch.full((total, total), float("-inf"), device=device)

        # 图像 token 区域互相可见。
        mask[:num_image_tokens, :num_image_tokens] = 0.0

        # 文本 token 可以看所有图像 token。
        mask[num_image_tokens:, :num_image_tokens] = 0.0

        # 文本 token 之间走标准因果 mask。
        text_causal_mask = torch.triu(
            torch.full((num_text_tokens, num_text_tokens), float("-inf"), device=device),
            diagonal=1,
        )
        mask[num_image_tokens:, num_image_tokens:] = text_causal_mask
        mask[num_image_tokens:, num_image_tokens:] += torch.tril(
            torch.zeros((num_text_tokens, num_text_tokens), device=device),
            diagonal=0,
        )
        return mask

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        前向传播。

        输入：
          images: [batch, time, 3, H, W]
          input_ids: [batch, text_seq_len]
          labels: [batch, text_seq_len]，仅 target 部分不是 -100

        输出：
          {
            "logits": [batch, text_seq_len, vocab_size],
            "loss": 标量（如果 labels 不为空）
          }
        """

        device = images.device
        image_tokens = self.vision_encoder(images)
        text_tokens = self.token_embedding(input_ids)

        num_image_tokens = image_tokens.size(1)
        num_text_tokens = text_tokens.size(1)
        total_tokens = num_image_tokens + num_text_tokens

        if total_tokens > self.max_total_tokens:
            raise ValueError(
                f"总 token 数 {total_tokens} 超过模型上限 {self.max_total_tokens}，"
                "请减小图像 token 数或文本长度。"
            )

        multimodal_tokens = torch.cat(
            [
                image_tokens + self.image_modality_embedding,
                text_tokens + self.text_modality_embedding,
            ],
            dim=1,
        )
        multimodal_tokens = multimodal_tokens + self.position_embedding[:, :total_tokens]

        attention_mask = self._build_prefix_lm_mask(
            num_image_tokens=num_image_tokens,
            num_text_tokens=num_text_tokens,
            device=device,
        )

        hidden_states = self.transformer(multimodal_tokens, mask=attention_mask)
        hidden_states = self.final_norm(hidden_states)

        # 只对文本部分做词表预测。
        text_hidden_states = hidden_states[:, num_image_tokens:]
        logits = self.lm_head(text_hidden_states)

        outputs: dict[str, Tensor] = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        prompt_ids: list[int],
        max_new_tokens: int,
        constrained_token_ids: list[list[int]] | None = None,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """
        自回归生成接口。

        对 RT-2 教学版来说，这个函数有两种典型用法：
        1. VQA：不加约束，自由生成普通文本答案
        2. Robot：逐步约束到某个动作维度允许的 token 集合，体现“动作也是 token”
        """

        self.eval()
        device = images.device

        generated = [*prompt_ids]
        produced: list[int] = []

        for step in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            outputs = self.forward(images=images, input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :]

            if constrained_token_ids is not None and step < len(constrained_token_ids):
                allowed = constrained_token_ids[step]
                constrained_logits = torch.full_like(next_token_logits, float("-inf"))
                constrained_logits[:, allowed] = next_token_logits[:, allowed]
                next_token_logits = constrained_logits

            next_token_id = int(next_token_logits.argmax(dim=-1).item())
            generated.append(next_token_id)
            produced.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

        return produced
