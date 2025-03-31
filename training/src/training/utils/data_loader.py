import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import random_split

from ..data.dataset import GameDataset

logger = logging.getLogger(__name__)

class GameDataLoader:
    def __init__(
        self,
        log_path: Path,
        train_val_split: float = 0.8,
        seed: int = 42
    ):
        """Initialize the data loader.
        
        Args:
            log_path: Path to the game log file
            train_val_split: Fraction of data to use for training
            seed: Random seed for reproducibility
        """
        self.log_path = log_path
        self.train_val_split = train_val_split
        self.seed = seed
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def _parse_single_trajectory(self, lines: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse a single trajectory from a list of lines."""
        states = []
        actions = []
        for line in lines:
            state_str, action = line.split()
            state = np.array([float(x) for x in state_str.split(',')]).reshape(4, 4)
            # Apply log2 to nonzero elements
            state[state != 0] = np.log2(state[state != 0])
            action_idx = {'U': 0, 'D': 1, 'L': 2, 'R': 3}[action]
            states.append(state)
            actions.append(action_idx)
        
        return torch.tensor(states), torch.tensor(actions), torch.tensor(len(states))


        
    def _parse_game_log(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Parse the game log file into a list of trajectories."""
        trajectories = []
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
            current_trajectory = []
            for line in lines:
                if line == 'NEW GAME\n':
                    if current_trajectory:
                        trajectories.append(self._parse_single_trajectory(current_trajectory))
                    current_trajectory = []
                else:
                    current_trajectory.append(line.strip())
            
            # Handle last trajectory if it exists
            if current_trajectory:
                trajectories.append(self._parse_single_trajectory(current_trajectory))
                    
        logger.info(f"Parsed {len(trajectories)} trajectories")
        return trajectories
    
    def load_data(self) -> None:
        """Load and process the game data, creating the dataset."""
        trajectories = self._parse_game_log()
        
        # Create the dataset with in-memory tensors
        self.dataset = GameDataset(trajectories)
        
    def split_data(self) -> Tuple[GameDataset, GameDataset]:
        """Split the dataset into training and validation sets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
            
        Raises:
            ValueError: If there are no validation samples after splitting
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        # Ensure at least one sample in each split
        total_samples = len(self.dataset)
        train_size = max(1, int(total_samples * self.train_val_split))
        val_size = total_samples - train_size
        
        if val_size == 0:
            raise ValueError(
                f"No validation samples available after splitting. "
                f"Total samples: {total_samples}, "
                f"Train split ratio: {self.train_val_split}"
            )
        
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Train size: {train_size}")
        logger.info(f"Val size: {val_size}")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        return self.train_dataset, self.val_dataset
    
    def get_datasets(self) -> Tuple[GameDataset, GameDataset]:
        """Load and split the data, returning train and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
            
        Raises:
            ValueError: If there are no validation samples after splitting
        """
        self.load_data()
        return self.split_data() 