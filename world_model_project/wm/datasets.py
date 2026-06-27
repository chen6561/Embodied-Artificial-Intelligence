from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("episode_*.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"No HDF5 episodes found in {self.root}")
        self.index = []
        self._build_index()

    def _build_index(self) -> None:
        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as handle:
                length = handle["obs"].shape[0]
            for step_idx in range(length):
                self.index.append((file_idx, step_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, step_idx = self.index[idx]
        path = self.files[file_idx]
        with h5py.File(path, "r") as handle:
            batch = {
                "obs": handle["obs"][step_idx],
                "action": handle["action"][step_idx],
                "next_obs": handle["next_obs"][step_idx],
                "reward": handle["reward"][step_idx],
                "done": handle["done"][step_idx],
                "success": handle["success"][step_idx],
            }
        return {key: torch.tensor(np.asarray(value), dtype=torch.float32) for key, value in batch.items()}
