from __future__ import annotations

import torch


class ACTPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def action_chunk(self, obs: torch.Tensor, horizon: int) -> torch.Tensor:
        actions = []
        for _ in range(horizon):
            actions.append(self.policy.act(obs))
        return torch.stack(actions, dim=0)
