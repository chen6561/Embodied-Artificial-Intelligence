from __future__ import annotations

import torch
import torch.nn.functional as F


def world_model_loss(predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    obs_loss = F.mse_loss(predictions["next_obs"], batch["next_obs"])
    reward_loss = F.mse_loss(predictions["reward"], batch["reward"])
    done_loss = F.binary_cross_entropy_with_logits(predictions["done_logit"], batch["done"])
    success_loss = F.binary_cross_entropy_with_logits(predictions["success_logit"], batch["success"])
    total = obs_loss + reward_loss + done_loss + success_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "obs_loss": float(obs_loss.detach().cpu()),
        "reward_loss": float(reward_loss.detach().cpu()),
        "done_loss": float(done_loss.detach().cpu()),
        "success_loss": float(success_loss.detach().cpu()),
    }
    return total, metrics
