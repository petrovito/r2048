from collections import defaultdict
import logging
from typing import List
from abc import ABC, abstractmethod

import torch
from training.preprocess.types import ParsedTrajectory

logger = logging.getLogger(__name__)

class ValidationSampler(ABC):
    @abstractmethod
    def sample(self, total_val_trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """Abstract method to sample validation trajectories.

        Args:
            total_val_trajectories: Combined list of validation trajectories.

        Returns:
            Sampled list of ParsedTrajectory objects.
        """
        pass

class LengthBasedSampler(ValidationSampler):
    def __init__(self, val_max_per_bin: int, val_bin_step: int):
        self.val_max_per_bin = val_max_per_bin
        self.val_bin_step = val_bin_step

    def sample(self, total_val_trajectories: List[ParsedTrajectory]) -> List[ParsedTrajectory]:
        """Sample validation trajectories for balanced length distribution.

        Args:
            total_val_trajectories: Combined list of validation trajectories.

        Returns:
            Sampled list of ParsedTrajectory objects.
        """
        if not total_val_trajectories:
            logger.warning("No validation trajectories to sample.")
            return []

        # Determine bin edges based on maximum length
        max_length = max(traj.length for traj in total_val_trajectories)
        bins = defaultdict(list)
        for traj in total_val_trajectories:
            bin_idx = (traj.length - 1) // self.val_bin_step
            bins[bin_idx].append(traj)

        # Sample each bin to val_max_per_bin
        sampled_trajectories = []
        for bin_idx, traj_list in bins.items():
            if len(traj_list) > self.val_max_per_bin:
                indices = torch.randperm(len(traj_list))[:self.val_max_per_bin].tolist()
                selected = [traj_list[i] for i in indices]
            else:
                selected = traj_list
            sampled_trajectories.extend(selected)
        logger.info(f"Sampled validation set to {len(sampled_trajectories)} trajectories")
        return sampled_trajectories