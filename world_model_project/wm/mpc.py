from __future__ import annotations

import torch


class CEMPlanner:
    def __init__(
        self,
        model,
        action_dim: int,
        horizon: int,
        num_samples: int,
        num_elites: int,
        iterations: int,
        action_std: float,
        device: str = "cpu",
    ):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.action_std = action_std
        self.device = device

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.model.encode(obs.to(self.device))
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones_like(mean) * self.action_std

        for _ in range(self.iterations):
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.num_samples, self.horizon, self.action_dim, device=self.device
            )
            scores = self._score_action_sequences(latent, samples)
            elite_indices = torch.topk(scores, self.num_elites).indices
            elites = samples[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(1e-3)

        return mean[0].detach().cpu()

    def _score_action_sequences(self, latent: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(action_sequences.shape[0], device=self.device)
        for idx in range(action_sequences.shape[0]):
            sim_latent = latent.clone()
            total_reward = 0.0
            success_bonus = 0.0
            for step in range(self.horizon):
                wm_out = self.model.step_latent(sim_latent, action_sequences[idx, step : step + 1])
                sim_latent = wm_out.next_latent
                total_reward = total_reward + wm_out.reward.squeeze()
                success_bonus = success_bonus + torch.sigmoid(wm_out.success_logit).squeeze()
            scores[idx] = total_reward + 0.1 * success_bonus
        return scores
