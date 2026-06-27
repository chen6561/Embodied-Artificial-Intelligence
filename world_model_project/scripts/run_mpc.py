from __future__ import annotations

from bootstrap import add_project_root_to_path

add_project_root_to_path()

import argparse

import torch

from isaac_tasks.wrappers import make_task
from scripts.common import load_config, set_seed
from wm.dynamics import WorldModel
from wm.mpc import CEMPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
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

    planner = CEMPlanner(
        model=model,
        action_dim=config["task"]["action_dim"],
        horizon=config["mpc"]["horizon"],
        num_samples=config["mpc"]["num_samples"],
        num_elites=config["mpc"]["num_elites"],
        iterations=config["mpc"]["iterations"],
        action_std=config["mpc"]["action_std"],
    )

    successes = 0
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        info = {"success": False}
        while not done:
            action = planner.plan(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()
            obs, _, done, info = env.step(action)
        successes += int(info["success"])

    print({"episodes": args.episodes, "successes": successes, "success_rate": successes / max(args.episodes, 1)})


if __name__ == "__main__":
    main()


