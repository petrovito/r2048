from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .trajectory import Trajectory


class GameDataset(Dataset):
    """Dataset for loading and preprocessing 2048 game trajectories for policy gradient training."""
    
    def __init__(
        self,
        data_path: Path,
        transform: Optional[callable] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the data file containing trajectories
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load trajectory data
        self.data = np.load(data_path)
        self.states = self.data["states"]  # Shape: (num_trajectories, max_steps, 4, 4)
        self.actions = self.data["actions"]  # Shape: (num_trajectories, max_steps)
        self.trajectory_lengths = self.data["trajectory_lengths"]  # Shape: (num_trajectories,)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Get a trajectory sample.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Dictionary containing:
                - states: Tensor of shape (trajectory_length, 4, 4)
                - actions: Tensor of shape (trajectory_length,)
                - length: Scalar tensor with trajectory length
        """
        length = self.trajectory_lengths[idx]
        
        states = torch.from_numpy(self.states[idx, :length]).float()
        actions = torch.from_numpy(self.actions[idx, :length]).long()
        
        if self.transform:
            states = self.transform(states)
        
        return {
            "states": states,
            "actions": actions,
            "length": torch.tensor(length),
        }
    
    @staticmethod
    def create_from_trajectories(
        states: np.ndarray,
        actions: np.ndarray,
        trajectory_lengths: np.ndarray,
        save_path: Path,
    ) -> None:
        """Create a dataset from trajectory data and save it.
        
        Args:
            states: Array of game states (num_trajectories, max_steps, 4, 4)
            actions: Array of actions (num_trajectories, max_steps)
            trajectory_lengths: Array of trajectory lengths (num_trajectories,)
            save_path: Path to save the dataset
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            states=states,
            actions=actions,
            trajectory_lengths=trajectory_lengths,
        ) 