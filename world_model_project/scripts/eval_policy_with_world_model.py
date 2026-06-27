from __future__ import annotations

import argparse

import torch

from isaac_tasks.wrappers import make_task
from policies.bc_policy import BehaviorCloningPolicy
from policies.diffusion_policy_adapter import DiffusionPolicyAdapter
from scripts.common import load_config, set_seed
from wm.dynamics import WorldModel


def score_action_chunk(model: WorldModel, obs: torch.Tensor, action_chunk: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        latent = model.encode(obs.unsqueeze(0))
        total_reward = 0.0
        total_success = 0.0
        for action in action_chunk:
            wm_out = model.step_latent(latent, action.unsqueeze(0))
            latent = wm_out.next_latent
            total_reward += float(wm_out.reward.squeeze().cpu())
            total_success += float(torch.sigmoid(wm_out.success_logit).squeeze().cpu())
    return {
        "predicted_return": total_reward,
        "predicted_success_score": total_success / max(len(action_chunk), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    env = make_task(config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = WorldModel(
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    base_policy = BehaviorCloningPolicy(config["task"]["obs_dim"], config["task"]["action_dim"])
    adapter = DiffusionPolicyAdapter(base_policy)

    obs = torch.tensor(env.reset(), dtype=torch.float32)
    action_chunk = adapter.action_chunk(obs.unsqueeze(0), args.horizon).squeeze(1)
    score = score_action_chunk(model, obs, action_chunk)
    print(score)


if __name__ == "__main__":
    main()
