# Enable forward reference annotations for type hints (supports Python versions < 3.10)
from __future__ import annotations

# Import Path class for cross-platform file and directory path manipulation
from pathlib import Path

# Library for reading/writing HDF5 binary data files (used to store trajectory episode data)
import h5py
# Numerical computation library for array storage and conversion before tensor casting
import numpy as np
# Core PyTorch library for tensor operations and deep learning framework
import torch
# Abstract Dataset base class from PyTorch data loading module, custom datasets inherit this
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """
    Custom PyTorch Dataset class to load single-step environment transition data stored in HDF5 episode files.
    Each data sample represents one environment transition tuple: (obs, action, next_obs, reward, done, success).
    All episode files follow the naming pattern "episode_*.hdf5" under the specified root directory.
    """
    def __init__(self, root: str | Path):
        """
        Initialize the transition dataset instance, scan all episode HDF5 files and build global sample index.

        Args:
            root: str or Path object pointing to the root directory containing all episode_*.hdf5 files.

        Raises:
            FileNotFoundError: If no HDF5 episode files matching the naming pattern are found under root.
        """
        # Convert input root path to Path object for unified cross-platform path operations
        self.root = Path(root)
        # Glob search all HDF5 episode files and sort them lexicographically for consistent loading order
        self.files = sorted(self.root.glob("episode_*.hdf5"))

        # Check if any episode file was located; raise error if file list is empty
        if not self.files:
            raise FileNotFoundError(f"No HDF5 episodes found in {self.root}")

        # Global index list storing tuples (file_index, step_index) for every single transition sample
        # Each entry maps one dataset sample to its corresponding file and in-file time step
        self.index = []
        # Build the global sample index by scanning length of each episode file
        self._build_index()

    def _build_index(self) -> None:
        """
        Private helper method to construct the global sample index self.index.
        Iterate over every episode file, read its total time step count, then record a (file_idx, step_idx)
        entry for each individual time step inside the file. This allows random access to any transition sample.
        """
        # Iterate over each episode file with its positional index in self.files list
        for file_idx, path in enumerate(self.files):
            # Open current HDF5 file in read-only mode to avoid write locks and data corruption
            with h5py.File(path, "r") as handle:
                # Read the first dimension of "obs" dataset to get total time steps of this episode
                # obs shape format: [num_steps, obs_dim], shape[0] = total transitions in this file
                length = handle["obs"].shape[0]
            # Traverse every single time step within the current episode file
            for step_idx in range(length):
                # Append mapping tuple to global index: which file, which time step inside that file
                self.index.append((file_idx, step_idx))

    def __len__(self) -> int:
        """
        Override base Dataset __len__ method to return total number of transition samples in the dataset.
        This value equals the total length of pre-built global index list.

        Returns:
            int: Total count of all single-step transition samples across all episode HDF5 files.
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Override base Dataset __getitem__ method to load a single transition sample by integer index.
        Perform file lookup via precomputed global index, read raw HDF5 arrays, convert to PyTorch float32 tensors.

        Args:
            idx: Integer index of the target transition sample (range: 0 ~ len(self.index) - 1)

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor-formatted transition data with keys:
                "obs": Environment observation tensor of current state
                "action": Action tensor executed at current state
                "next_obs": Environment observation tensor of the next state after action execution
                "reward": Scalar reward tensor received after taking the action
                "done": Binary flag tensor indicating whether the episode terminates after this step
                "success": Binary flag tensor indicating task completion success at this transition
        """
        # Unpack global index tuple to get target file index and in-file time step index
        file_idx, step_idx = self.index[idx]
        # Get absolute file path of the target episode HDF5 file
        path = self.files[file_idx]

        # Open target HDF5 episode file in read-only mode
        with h5py.File(path, "r") as handle:
            # Extract raw numpy arrays for all transition components at specified time step
            batch = {
                "obs": handle["obs"][step_idx],
                "action": handle["action"][step_idx],
                "next_obs": handle["next_obs"][step_idx],
                "reward": handle["reward"][step_idx],
                "done": handle["done"][step_idx],
                "success": handle["success"][step_idx],
            }

        # Convert each raw HDF5 array to standard np.ndarray, then cast to float32 PyTorch Tensor
        # Uniform dtype float32 ensures consistent computation precision for RL training pipelines
        return {key: torch.tensor(np.asarray(value), dtype=torch.float32) for key, value in batch.items()}