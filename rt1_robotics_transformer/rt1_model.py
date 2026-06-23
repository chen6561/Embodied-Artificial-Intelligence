import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RT1Config:
    image_size: int = 128
    sequence_length: int = 6
    vocab_size: int = 256
    num_action_dims: int = 11
    text_dim: int = 128
    embed_dim: int = 256
    tokens_per_frame: int = 8
    visual_tokens_before_pool: int = 64
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_dropout: float = 0.1


class SinusoidalPositionEncoding(nn.Module):
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
        return x + self.pe[:, : x.size(1)]


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(text_dim, channels * 2)

    def forward(self, x: Tensor, text_embedding: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(text_embedding).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class VisualEncoder(nn.Module):
    """
    A small FiLM-conditioned CNN that plays the same role as the EfficientNet
    image tokenizer in RT-1, but stays simple enough for study and reimplementation.
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
        x = self.stem(images)
        x = self.film(x, text_embedding)
        x = self.pool(x)
        batch, channels, height, width = x.shape
        return x.flatten(2).transpose(1, 2).reshape(batch, height * width, channels)


class TokenLearner(nn.Module):
    """
    Learns a small set of attention masks that compress many visual tokens into a few.
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
        attn_logits = self.score(tokens).transpose(1, 2)
        attn = attn_logits.softmax(dim=-1)
        return attn @ tokens


class RT1ActionHead(nn.Module):
    def __init__(self, config: RT1Config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.num_action_dims * config.vocab_size)
        self.num_action_dims = config.num_action_dims
        self.vocab_size = config.vocab_size

    def forward(self, hidden: Tensor) -> Tensor:
        logits = self.proj(self.norm(hidden))
        return logits.view(hidden.size(0), self.num_action_dims, self.vocab_size)


class RT1Model(nn.Module):
    """
    Educational RT-1 style model.

    Inputs:
      images: [batch, time, 3, H, W]
      text_embedding: [batch, text_dim]

    Outputs:
      action_logits: [batch, action_dims, vocab_size]
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
        batch, time, channels, height, width = images.shape
        images = images.view(batch * time, channels, height, width)
        repeated_text = text_embedding.unsqueeze(1).expand(batch, time, -1).reshape(batch * time, -1)
        visual_tokens = self.visual_encoder(images, repeated_text)
        learned_tokens = self.token_learner(visual_tokens)
        learned_tokens = learned_tokens.view(batch, time, self.config.tokens_per_frame, self.config.embed_dim)
        learned_tokens = learned_tokens + self.frame_embedding[:, :time].unsqueeze(2)
        return learned_tokens.view(batch, time * self.config.tokens_per_frame, self.config.embed_dim)

    def forward(self, images: Tensor, text_embedding: Tensor) -> Tensor:
        tokens = self.encode_frames(images, text_embedding)
        tokens = self.positional_encoding(tokens)
        hidden = self.transformer(tokens)
        last_state = hidden[:, -1]
        return self.action_head(last_state)


def discretize_action(actions: Tensor, vocab_size: int = 256) -> Tensor:
    """
    Maps continuous actions in [-1, 1] to integer bins like RT-1's tokenized actions.
    """

    clipped = actions.clamp(-1.0, 1.0)
    scaled = (clipped + 1.0) * 0.5 * (vocab_size - 1)
    return scaled.round().long()


def undiscretize_action(tokens: Tensor, vocab_size: int = 256) -> Tensor:
    scaled = tokens.float() / max(vocab_size - 1, 1)
    return scaled * 2.0 - 1.0
