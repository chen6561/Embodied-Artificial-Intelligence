from __future__ import annotations

from bootstrap import add_project_root_to_path

add_project_root_to_path()

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.common import ensure_dir, load_config, save_json, set_seed
from wm.datasets import TransitionDataset
from wm.dynamics import WorldModel
from wm.losses import world_model_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device(config["train"]["device"])
    dataset = TransitionDataset(args.dataset_dir)
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)

    model = WorldModel(
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    output_root = ensure_dir(Path(args.config).resolve().parent.parent / "outputs" / "checkpoints")
    history = []

    for epoch in range(config["train"]["epochs"]):
        epoch_metrics = {"loss": 0.0, "obs_loss": 0.0, "reward_loss": 0.0, "done_loss": 0.0, "success_loss": 0.0}
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            wm_out = model(batch["obs"], batch["action"])
            predictions = {
                "next_obs": wm_out.next_obs,
                "reward": wm_out.reward,
                "done_logit": wm_out.done_logit,
                "success_logit": wm_out.success_logit,
            }
            loss, metrics = world_model_loss(predictions, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

        num_batches = max(len(loader), 1)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        print(epoch_metrics)

    ckpt_path = output_root / f"{config['task']['name']}_world_model.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)
    save_json(output_root / f"{config['task']['name']}_train_history.json", {"history": history})
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()


