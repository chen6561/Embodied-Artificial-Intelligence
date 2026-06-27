from __future__ import annotations

import torch


class DiffusionPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def action_chunk(self, obs: torch.Tensor, horizon: int) -> torch.Tensor:
        if hasattr(self.policy, "sample_action_chunk"):
            return self.policy.sample_action_chunk(obs, horizon)
        actions = []
        for _ in range(horizon):
            actions.append(self.policy.act(obs))
        return torch.stack(actions, dim=0)
