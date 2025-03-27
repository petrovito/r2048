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
        save_path: Path,
        train_val_split: float = 0.8,
        seed: int = 42
    ):
        """Initialize the data loader.
        
        Args:
            log_path: Path to the game log file
            save_path: Path to save the processed dataset
            train_val_split: Fraction of data to use for training
            seed: Random seed for reproducibility
        """
        self.log_path = log_path
        self.save_path = save_path
        self.train_val_split = train_val_split
        self.seed = seed
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def _parse_game_log(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse a game log file into arrays suitable for training.
        
        Returns:
            Tuple of (states, actions, rewards, trajectory_lengths)
        """
        states = []
        actions = []
        rewards = []
        current_trajectory = []
        current_actions = []
        current_rewards = []
        
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line == "NEW GAME":
                    if current_trajectory:
                        states.append(current_trajectory)
                        actions.append(current_actions)
                        rewards.append(current_rewards)
                        current_trajectory = []
                        current_actions = []
                        current_rewards = []
                    continue
                
                # Parse state and action
                state_str, action = line.split()
                state = np.array([int(x) for x in state_str.split(',')]).reshape(4, 4)
                
                # Convert action to index (U=0, D=1, L=2, R=3)
                action_idx = {'U': 0, 'D': 1, 'L': 2, 'R': 3}[action]
                
                # Calculate reward using log2 values
                # For 0 tiles, use 0 as log2(0) is undefined
                # For non-zero tiles, use log2(value)
                mask = state > 0
                log2_state = np.zeros_like(state, dtype=float)
                log2_state[mask] = np.log2(state[mask])
                reward = np.sum(log2_state)
                
                current_trajectory.append(state)
                current_actions.append(action_idx)
                current_rewards.append(reward)
        
        # Add the last trajectory if not empty
        if current_trajectory:
            states.append(current_trajectory)
            actions.append(current_actions)
            rewards.append(current_rewards)
        
        if not states:
            raise ValueError("No valid trajectories found in the log file")
        
        # Convert to numpy arrays and pad to max length
        max_length = max(len(traj) for traj in states)
        num_trajectories = len(states)
        
        states_array = np.zeros((num_trajectories, max_length, 4, 4))
        actions_array = np.zeros((num_trajectories, max_length), dtype=int)
        rewards_array = np.zeros((num_trajectories, max_length))
        lengths_array = np.array([len(traj) for traj in states])
        
        for i in range(num_trajectories):
            states_array[i, :len(states[i])] = states[i]
            actions_array[i, :len(actions[i])] = actions[i]
            rewards_array[i, :len(rewards[i])] = rewards[i]
        
        logger.info(f"Loaded {num_trajectories} trajectories with max length {max_length}")
        logger.info(f"Trajectory lengths: {lengths_array}")
        logger.info(f"Total number of states: {sum(lengths_array)}")
        
        return states_array, actions_array, rewards_array, lengths_array
    
    def load_data(self) -> None:
        """Load and process the game data, creating the dataset."""
        states, actions, rewards, lengths = self._parse_game_log()
        
        # Save the processed data
        GameDataset.create_from_trajectories(
            states=states,
            actions=actions,
            rewards=rewards,
            trajectory_lengths=lengths,
            save_path=self.save_path,
        )
        
        # Create the dataset
        self.dataset = GameDataset(self.save_path)
        
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