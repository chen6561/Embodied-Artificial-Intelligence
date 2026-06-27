from __future__ import annotations

import torch


def rollout_mse(pred_rollout: torch.Tensor, gt_rollout: torch.Tensor) -> float:
    return float(torch.mean((pred_rollout - gt_rollout) ** 2).detach().cpu())


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return float((preds == targets).float().mean().detach().cpu())


def episode_return(rewards: list[float]) -> float:
    return float(sum(rewards))
