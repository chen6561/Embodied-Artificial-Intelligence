from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoders import MLP, StateEncoder


@dataclass
class WorldModelOutput:
    latent: torch.Tensor
    next_latent: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor
    success_logit: torch.Tensor


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = StateEncoder(obs_dim, hidden_dim, latent_dim)
        self.transition = MLP(latent_dim + action_dim, hidden_dim, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dim, obs_dim)
        self.reward_head = MLP(latent_dim + action_dim, hidden_dim, 1)
        self.done_head = MLP(latent_dim + action_dim, hidden_dim, 1)
        self.success_head = MLP(latent_dim, hidden_dim, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def step_latent(self, latent: torch.Tensor, action: torch.Tensor) -> WorldModelOutput:
        inputs = torch.cat([latent, action], dim=-1)
        next_latent = self.transition(inputs)
        next_obs = self.decoder(next_latent)
        reward = self.reward_head(inputs)
        done_logit = self.done_head(inputs)
        success_logit = self.success_head(next_latent)
        return WorldModelOutput(
            latent=latent,
            next_latent=next_latent,
            next_obs=next_obs,
            reward=reward,
            done_logit=done_logit,
            success_logit=success_logit,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> WorldModelOutput:
        latent = self.encode(obs)
        return self.step_latent(latent, action)
