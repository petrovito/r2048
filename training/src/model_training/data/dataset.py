from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    """Dataset for loading and preprocessing 2048 game trajectories for policy gradient training."""
    
    def __init__(
        self,
        trajectories: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        transform: callable = None,
    ):
        """Initialize the dataset.
        
        Args:
            trajectories: List of tuples containing (states, actions, lengths)
            transform: Optional transform to apply to the data
        """
        self.trajectories = trajectories
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
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
        states, actions, length = self.trajectories[idx]
        
        if self.transform:
            states = self.transform(states)
        
        return {
            "states": states,
            "actions": actions,
            "length": length,
        }
    