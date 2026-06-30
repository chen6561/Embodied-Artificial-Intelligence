from __future__ import annotations

# Import utility to add project root directory to Python path
from bootstrap import add_project_root_to_path

# Execute the path setup to ensure project modules are accessible
add_project_root_to_path()

# Standard library imports
import argparse
from pathlib import Path

# Third-party library imports
import h5py
import numpy as np

# Project-specific imports
from isaac_tasks.wrappers import make_task
from scripts.common import ensure_dir, load_config, save_json, set_seed


def collect_episode(env, policy_mode: str) -> dict[str, np.ndarray]:
    """
    Collect a single episode of interaction data with the environment using specified policy mode.

    Args:
        env: The Isaac Gym environment instance
        policy_mode: Policy type to generate actions ("random", "scripted", "noisy_scripted")

    Returns:
        Dictionary containing episode data with following keys:
            - obs: Observation sequence (shape: [timesteps, obs_dim])
            - action: Action sequence (shape: [timesteps, action_dim])
            - next_obs: Next observation sequence (shape: [timesteps, obs_dim])
            - reward: Reward sequence (shape: [timesteps, 1])
            - done: Termination flag sequence (shape: [timesteps, 1])
            - success: Task success flag sequence (shape: [timesteps, 1])
    """
    # Reset environment to start new episode and get initial observation
    obs = env.reset()

    # Initialize buffers to store episode data
    obs_buffer, action_buffer, next_obs_buffer = [], [], []
    reward_buffer, done_buffer, success_buffer = [], [], []

    # Episode termination flag
    done = False

    # Run episode loop until termination
    while not done:
        # Generate action based on selected policy mode
        if policy_mode == "random":
            # Random uniform action in [-1.0, 1.0] range (matches action space bounds)
            action = np.random.uniform(-1.0, 1.0, size=env.spec.action_dim).astype(np.float32)

        elif policy_mode == "scripted":
            # Use environment's built-in scripted policy (deterministic)
            action = env.scripted_action(obs)

        elif policy_mode == "noisy_scripted":
            # Add Gaussian noise to scripted action for exploration
            action = env.scripted_action(obs) + np.random.normal(0.0, 0.15, size=env.spec.action_dim)
            # Clip action to valid range [-1.0, 1.0] and ensure float32 type
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

        else:
            # Raise error for unknown policy modes
            raise ValueError(f"Unknown policy_mode={policy_mode}")

        # Step environment with generated action
        next_obs, reward, done, info = env.step(action)

        # Store transition data in buffers
        obs_buffer.append(obs)  # Current observation
        action_buffer.append(action)  # Taken action
        next_obs_buffer.append(next_obs)  # Next observation
        reward_buffer.append([reward])  # Received reward (wrapped in list for consistent shape)
        done_buffer.append([float(done)])  # Termination flag (float conversion for numerical consistency)
        success_buffer.append([float(info["success"])])  # Task success flag

        # Update current observation for next timestep
        obs = next_obs

    # Convert buffers to numpy arrays (float32 for memory efficiency and GPU compatibility)
    return {
        "obs": np.asarray(obs_buffer, dtype=np.float32),
        "action": np.asarray(action_buffer, dtype=np.float32),
        "next_obs": np.asarray(next_obs_buffer, dtype=np.float32),
        "reward": np.asarray(reward_buffer, dtype=np.float32),
        "done": np.asarray(done_buffer, dtype=np.float32),
        "success": np.asarray(success_buffer, dtype=np.float32),
    }


def main() -> None:
    """
    Main function to collect multiple episodes of environment data:
    1. Parse command line arguments
    2. Load configuration and setup environment
    3. Collect specified number of episodes
    4. Save episode data to HDF5 files and metadata to JSON
    """
    # Initialize argument parser for command line configuration
    parser = argparse.ArgumentParser(
        description="Collect interaction data from Isaac Gym environments using different policies")

    # Command line arguments
    parser.add_argument("--config", required=True, help="Path to configuration file (JSON/YAML)")
    parser.add_argument(
        "--policy_mode",
        default="random",
        choices=["random", "scripted", "noisy_scripted"],
        help="Policy type for action generation: random (uniform random), scripted (environment-provided), noisy_scripted (scripted + Gaussian noise)"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Load environment and data collection configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_seed(config["seed"])

    # Create Isaac Gym environment from configuration
    env = make_task(config)

    # Create output directory structure: <config_dir>/../raw_dir/<policy_mode>
    # ensure_dir creates directory if it doesn't exist
    output_dir = ensure_dir(Path(args.config).resolve().parent.parent / config["data"]["raw_dir"] / args.policy_mode)

    # Collect specified number of episodes
    for episode_idx in range(config["data"]["num_episodes"]):
        # Collect single episode data
        episode = collect_episode(env, args.policy_mode)

        # Define path for episode HDF5 file (zero-padded episode index)
        episode_path = output_dir / f"episode_{episode_idx:06d}.hdf5"

        # Save episode data to HDF5 file (efficient for numerical arrays)
        with h5py.File(episode_path, "w") as handle:
            for key, value in episode.items():
                # Create dataset for each data type in HDF5 file
                handle.create_dataset(key, data=value)

    # Save metadata about collected data (policy type and number of episodes)
    metadata = {
        "policy_mode": args.policy_mode,
        "num_episodes": config["data"]["num_episodes"],
        "seed": config["seed"],  # Add seed to metadata for reproducibility
        "output_directory": str(output_dir)
    }
    save_json(output_dir / "meta.json", metadata)

    # Print completion message with output directory
    print(f"Successfully saved {config['data']['num_episodes']} episodes to {output_dir}")


# Entry point for script execution
if __name__ == "__main__":
    main()