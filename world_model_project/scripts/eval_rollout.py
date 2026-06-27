from __future__ import annotations

from bootstrap import add_project_root_to_path

add_project_root_to_path()

import argparse

import torch

from scripts.common import load_config
from wm.datasets import TransitionDataset
from wm.dynamics import WorldModel
from wm.metrics import binary_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = TransitionDataset(args.dataset_dir)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = WorldModel(
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    obs_errors, reward_errors, done_accs = [], [], []
    for idx in range(min(512, len(dataset))):
        batch = dataset[idx]
        with torch.no_grad():
            wm_out = model(batch["obs"].unsqueeze(0), batch["action"].unsqueeze(0))
        obs_errors.append(torch.mean((wm_out.next_obs.squeeze(0) - batch["next_obs"]) ** 2).item())
        reward_errors.append(torch.mean((wm_out.reward.squeeze(0) - batch["reward"]) ** 2).item())
        done_accs.append(binary_accuracy(wm_out.done_logit, batch["done"].unsqueeze(0)))

    print(
        {
            "obs_mse": sum(obs_errors) / len(obs_errors),
            "reward_mse": sum(reward_errors) / len(reward_errors),
            "done_acc": sum(done_accs) / len(done_accs),
        }
    )


if __name__ == "__main__":
    main()


