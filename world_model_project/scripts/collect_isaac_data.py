from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from isaac_tasks.wrappers import make_task
from scripts.common import ensure_dir, load_config, save_json, set_seed


def collect_episode(env, policy_mode: str) -> dict[str, np.ndarray]:
    obs = env.reset()
    obs_buffer, action_buffer, next_obs_buffer = [], [], []
    reward_buffer, done_buffer, success_buffer = [], [], []

    done = False
    while not done:
        if policy_mode == "random":
            action = np.random.uniform(-1.0, 1.0, size=env.spec.action_dim).astype(np.float32)
        elif policy_mode == "scripted":
            action = env.scripted_action(obs)
        elif policy_mode == "noisy_scripted":
            action = env.scripted_action(obs) + np.random.normal(0.0, 0.15, size=env.spec.action_dim)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"Unknown policy_mode={policy_mode}")

        next_obs, reward, done, info = env.step(action)
        obs_buffer.append(obs)
        action_buffer.append(action)
        next_obs_buffer.append(next_obs)
        reward_buffer.append([reward])
        done_buffer.append([float(done)])
        success_buffer.append([float(info["success"])])
        obs = next_obs

    return {
        "obs": np.asarray(obs_buffer, dtype=np.float32),
        "action": np.asarray(action_buffer, dtype=np.float32),
        "next_obs": np.asarray(next_obs_buffer, dtype=np.float32),
        "reward": np.asarray(reward_buffer, dtype=np.float32),
        "done": np.asarray(done_buffer, dtype=np.float32),
        "success": np.asarray(success_buffer, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy_mode", default="random", choices=["random", "scripted", "noisy_scripted"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    env = make_task(config)
    output_dir = ensure_dir(Path(args.config).resolve().parent.parent / config["data"]["raw_dir"] / args.policy_mode)

    for episode_idx in range(config["data"]["num_episodes"]):
        episode = collect_episode(env, args.policy_mode)
        episode_path = output_dir / f"episode_{episode_idx:06d}.hdf5"
        with h5py.File(episode_path, "w") as handle:
            for key, value in episode.items():
                handle.create_dataset(key, data=value)

    save_json(output_dir / "meta.json", {"policy_mode": args.policy_mode, "num_episodes": config["data"]["num_episodes"]})
    print(f"Saved episodes to {output_dir}")


if __name__ == "__main__":
    main()
