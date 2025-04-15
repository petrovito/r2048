import logging
from pathlib import Path
import random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass
from collections import defaultdict
from typing import TypeVar, Optional

from .types import ParsedTrajectory
from .validation_sampler import ValidationSampler, LengthBasedSampler
from .file_parser import FileParser

from ..data.dataset import GameDataset

logger = logging.getLogger(__name__)


### Main Data Loader Class
class GameDataLoader:
    def __init__(
        self,
        train_file_paths: List[Path],
        extra_val_file_paths: Optional[List[Path]] = None,
        train_val_split: float = 0.8,
        val_max_per_bin: int = 100,
        val_bin_step: int = 10,
        seed: int = 42,
        validation_sampler: Optional[ValidationSampler] = None
    ):
        """Initialize the data loader with multiple file paths.

        Args:
            train_file_paths: List of paths to training game log files.
            extra_val_file_paths: List of paths to extra validation game log files.
            train_val_split: Fraction of training data to use for training (rest for validation).
            val_max_per_bin: Max number of trajectories per length bin in validation set.
            val_bin_step: Step size for trajectory length bins.
            seed: Random seed for reproducibility.
            validation_sampler: Instance of ValidationSampler for flexible validation sampling.
        """
        self.train_file_paths = train_file_paths
        self.extra_val_file_paths = extra_val_file_paths or []
        self.train_val_split = train_val_split
        self.val_max_per_bin = val_max_per_bin
        self.val_bin_step = val_bin_step
        self.seed = seed
        self.validation_sampler = validation_sampler or LengthBasedSampler(val_max_per_bin, val_bin_step)
        self.train_trajectories: List[ParsedTrajectory] = []
        self.extra_val_trajectories: List[ParsedTrajectory] = []
        self.train_subset: List[ParsedTrajectory] = []
        self.val_subset: List[ParsedTrajectory] = []
        self.pruned_val_trajectories: List[ParsedTrajectory] = []


    def parse_files(self, file_paths: List[Path]) -> List[ParsedTrajectory]:
        """Parse multiple game log files into a list of trajectories.

        Args:
            file_paths: List of file paths to parse.

        Returns:
            Combined list of ParsedTrajectory objects from all files.
        """
        file_parser = FileParser()
        all_trajectories: List[ParsedTrajectory] = []
        for file_path in file_paths:
            trajectories = file_parser.parse_game_log(file_path)
            all_trajectories.extend(trajectories)
        return all_trajectories

    def load_data(self) -> None:
        """Load and parse game data from training and extra validation files."""
        self.train_trajectories = self.parse_files(self.train_file_paths)
        self.extra_val_trajectories = self.parse_files(self.extra_val_file_paths)
        logger.info(f"Parsed {len(self.train_trajectories)} training trajectories")
        logger.info(f"Parsed {len(self.extra_val_trajectories)} extra validation trajectories")

    


    def split_data(self) -> None:
        """Split training trajectories into training and validation subsets."""
        self.train_subset, self.val_subset = split_list(self.train_trajectories, self.train_val_split)
        logger.info(f"Training subset size: {len(self.train_subset)}")
        logger.info(f"Validation subset size from training: {len(self.val_subset)}")


    def trajectories_to_tensors(
        self, trajectories: List[ParsedTrajectory]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert trajectories to tensor tuples.

        Args:
            trajectories: List of ParsedTrajectory objects.

        Returns:
            List of (states_tensor, actions_tensor, length_tensor) tuples.
        """
        tensor_list = []
        for traj in trajectories:
            np_states = np.array([float(x) for state in traj.states for x in state]).reshape(-1, 4, 4)
            np_states[np_states != 0] = np.log2(np_states[np_states != 0])
            states_tensor = torch.tensor(np_states, dtype=torch.float32)
            actions_tensor = torch.tensor(traj.actions, dtype=torch.long)
            length_tensor = torch.tensor(traj.length, dtype=torch.long)
            tensor_list.append((states_tensor, actions_tensor, length_tensor))
        return tensor_list

    def get_datasets(self) -> Tuple[GameDataset, GameDataset]:
        """Load, split, prune, and create training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        torch.manual_seed(self.seed)
        self.load_data()
        self.split_data()
        total_val_trajectories = self.extra_val_trajectories + self.val_subset
        self.pruned_val_trajectories = self.validation_sampler.sample(total_val_trajectories)
        train_trajectories = self.train_subset
        
        train_tensor_list = self.trajectories_to_tensors(train_trajectories)
        val_tensor_list = self.trajectories_to_tensors(self.pruned_val_trajectories)
        
        train_dataset = GameDataset(train_tensor_list)
        val_dataset = GameDataset(val_tensor_list)
        
        logger.info(f"Final training dataset size: {len(train_dataset)}")
        logger.info(f"Final validation dataset size: {len(val_dataset)}")
        return train_dataset, val_dataset
    
_T = TypeVar("_T")    
def split_list(lst: List[_T], ratio=0.8) -> Tuple[List[_T], List[_T]]:
    random.shuffle(lst)  # Shuffle the list in place
    split_idx = int(len(lst) * ratio)  # Calculate split index
    return lst[:split_idx], lst[split_idx:]