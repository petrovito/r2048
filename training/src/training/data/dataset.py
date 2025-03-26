from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    """Dataset for loading and preprocessing 2048 game data."""
    
    def __init__(
        self,
        data_path: Path,
        transform: Optional[callable] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the data file
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load data
        self.data = np.load(data_path)
        self.positions = self.data["positions"]
        self.moves = self.data["moves"]
        
        # Convert moves to one-hot encoding
        self.moves_onehot = np.zeros((len(self.moves), 4))
        self.moves_onehot[np.arange(len(self.moves)), self.moves] = 1
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (position, move)
        """
        position = torch.from_numpy(self.positions[idx]).float()
        move = torch.from_numpy(self.moves_onehot[idx]).float()
        
        if self.transform:
            position = self.transform(position)
        
        return position, move
    
    @staticmethod
    def create_from_game_data(
        positions: np.ndarray,
        moves: np.ndarray,
        save_path: Path,
    ) -> None:
        """Create a dataset from game data and save it.
        
        Args:
            positions: Array of game positions
            moves: Array of moves
            save_path: Path to save the dataset
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            positions=positions,
            moves=moves,
        ) 