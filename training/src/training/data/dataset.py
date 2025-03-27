from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


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
        self.rewards = self.data["rewards"]  # Shape: (num_trajectories, max_steps)
        self.trajectory_lengths = self.data["trajectory_lengths"]  # Shape: (num_trajectories,)
        
        # Calculate returns for each trajectory
        self.returns = self._calculate_returns()
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate the returns (sum of rewards) for each trajectory."""
        returns = np.zeros_like(self.rewards)
        for i in range(len(self.trajectory_lengths)):
            length = self.trajectory_lengths[i]
            # Calculate returns with discounting (gamma = 0.99)
            returns[i, length-1] = self.rewards[i, length-1]
            for t in range(length-2, -1, -1):
                returns[i, t] = self.rewards[i, t] + 0.99 * returns[i, t+1]
        return returns
    
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
                - rewards: Tensor of shape (trajectory_length,)
                - returns: Tensor of shape (trajectory_length,)
                - length: Scalar tensor with trajectory length
        """
        length = self.trajectory_lengths[idx]
        
        states = torch.from_numpy(self.states[idx, :length]).float()
        actions = torch.from_numpy(self.actions[idx, :length]).long()
        rewards = torch.from_numpy(self.rewards[idx, :length]).float()
        returns = torch.from_numpy(self.returns[idx, :length]).float()
        
        if self.transform:
            states = self.transform(states)
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns": returns,
            "length": torch.tensor(length),
        }
    
    @staticmethod
    def create_from_trajectories(
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        trajectory_lengths: np.ndarray,
        save_path: Path,
    ) -> None:
        """Create a dataset from trajectory data and save it.
        
        Args:
            states: Array of game states (num_trajectories, max_steps, 4, 4)
            actions: Array of actions (num_trajectories, max_steps)
            rewards: Array of rewards (num_trajectories, max_steps)
            trajectory_lengths: Array of trajectory lengths (num_trajectories,)
            save_path: Path to save the dataset
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            states=states,
            actions=actions,
            rewards=rewards,
            trajectory_lengths=trajectory_lengths,
        ) 