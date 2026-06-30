from __future__ import annotations

# Add project root directory to Python path to enable module imports from root
from bootstrap import add_project_root_to_path

add_project_root_to_path()

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import torch
from torch.utils.data import DataLoader

# Local project imports
from scripts.common import ensure_dir, load_config, save_json, set_seed
from wm.datasets import TransitionDataset
from wm.dynamics import WorldModel
from wm.losses import world_model_loss


def main() -> None:
    """
    Main function to train the World Model.
    This function handles:
    1. Parsing command line arguments
    2. Loading configuration and setting random seed
    3. Preparing dataset and data loader
    4. Initializing model, optimizer and training components
    5. Running training loop with loss calculation and backpropagation
    6. Saving trained model checkpoint and training history
    """
    # Initialize argument parser for command line inputs
    parser = argparse.ArgumentParser(description="Train World Model for sequential decision making tasks")

    # Add required command line arguments
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the JSON configuration file containing training/model/dataset parameters"
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="Directory path containing the transition dataset (observations, actions, rewards, etc.)"
    )

    # Parse and store command line arguments
    args = parser.parse_args()

    # Load configuration from JSON file (contains hyperparameters, task specs, etc.)
    config = load_config(args.config)

    # Set random seed for reproducibility across training runs
    set_seed(config["seed"])

    # Set computation device (CPU/GPU) based on configuration
    device = torch.device(config["train"]["device"])

    # Initialize transition dataset (loads observation-action-reward transitions)
    dataset = TransitionDataset(args.dataset_dir)

    # Create DataLoader for batching and shuffling dataset
    # - batch_size: Number of samples per batch (from config)
    # - shuffle: Randomly shuffle data each epoch for better training
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True
    )

    # Initialize World Model with task/model-specific dimensions from config
    # - obs_dim: Dimension of observation space
    # - action_dim: Dimension of action space
    # - hidden_dim: Dimension of hidden layers in neural network
    # - latent_dim: Dimension of latent space in world model
    # - to(device): Move model to specified computation device
    model = WorldModel(
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)

    # Initialize AdamW optimizer for model parameter updates
    # - lr: Learning rate (from config)
    # - weight_decay: L2 regularization strength (from config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # Create output directory for saving model checkpoints (ensure directory exists)
    # Path structure: <config_parent_dir>/../outputs/checkpoints/
    output_root = ensure_dir(Path(args.config).resolve().parent.parent / "outputs" / "checkpoints")

    # Initialize list to store training metrics history across epochs
    history = []

    # Main training loop over specified number of epochs
    for epoch in range(config["train"]["epochs"]):
        # Initialize metrics dictionary to accumulate losses for current epoch
        # Tracks total loss and component losses (observation, reward, done, success)
        epoch_metrics = {
            "loss": 0.0,  # Total combined loss
            "obs_loss": 0.0,  # Observation prediction loss
            "reward_loss": 0.0,  # Reward prediction loss
            "done_loss": 0.0,  # Done flag classification loss
            "success_loss": 0.0  # Success flag classification loss
        }

        # Iterate over batches in the DataLoader
        for batch in loader:
            # Move all batch tensors to the specified computation device (CPU/GPU)
            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass through the World Model
            # Input: current observation + action
            # Output: predicted next observation, reward, done flag, success flag
            wm_out = model(batch["obs"], batch["action"])

            # Organize model predictions into a dictionary for loss calculation
            predictions = {
                "next_obs": wm_out.next_obs,  # Predicted next observation
                "reward": wm_out.reward,  # Predicted reward
                "done_logit": wm_out.done_logit,  # Logit for done flag (before sigmoid)
                "success_logit": wm_out.success_logit  # Logit for success flag (before sigmoid)
            }

            # Calculate total loss and individual loss components
            # world_model_loss returns: (total_loss, metrics_dict)
            loss, metrics = world_model_loss(predictions, batch)

            # Reset optimizer gradients to zero (prevents accumulation across batches)
            optimizer.zero_grad()

            # Backward pass: compute gradients of loss with respect to model parameters
            loss.backward()

            # Update model parameters using computed gradients
            optimizer.step()

            # Accumulate metrics for current batch into epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

        # Calculate average metrics per batch for the epoch
        # Use max(len(loader), 1) to avoid division by zero (empty loader edge case)
        num_batches = max(len(loader), 1)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        # Add epoch number to metrics for tracking
        epoch_metrics["epoch"] = epoch

        # Append current epoch metrics to training history
        history.append(epoch_metrics)

        # Print epoch metrics to console (training progress monitoring)
        print(epoch_metrics)

    # Define path for saving model checkpoint
    ckpt_path = output_root / f"{config['task']['name']}_world_model.pt"

    # Save model checkpoint: includes model state dict and configuration
    # - model_state: Trained weights/parameters of the model
    # - config: Training configuration (for reproducibility/loading)
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)

    # Save training history as JSON file (for post-training analysis/visualization)
    save_json(output_root / f"{config['task']['name']}_train_history.json", {"history": history})

    # Print confirmation message with checkpoint path
    print(f"Saved checkpoint to {ckpt_path}")


# Entry point for script execution
if __name__ == "__main__":
    main()