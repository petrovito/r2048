from typing import Generator, List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import GameDataset
from ..types import Trajectory


class TrajectoryBatchSampler:
    """Custom batch sampler that groups trajectories based on total sample count."""
    
    def __init__(self, dataset: GameDataset, max_samples_per_batch: int):
        self.dataset = dataset
        self.max_samples_per_batch = max_samples_per_batch
        self.indices = list(range(len(dataset)))
    
    def __iter__(self) -> Generator[List[int], None, None]:
        np.random.shuffle(self.indices)
        current_batch: List[int] = []
        current_samples = 0
        
        for idx in self.indices:
            item = self.dataset[idx]
            item_samples = int(item["length"].item())
            
            if current_samples + item_samples > self.max_samples_per_batch and current_batch:
                yield current_batch
                current_batch = []
                current_samples = 0
            
            current_batch.append(idx)
            current_samples += item_samples
        
        if current_batch:
            yield current_batch
    
    def __len__(self):
        # This is an approximation since we don't know the exact number of batches
        # until we actually create them
        total_samples = sum(self.dataset[i]["length"].item() for i in self.indices)
        return (total_samples + self.max_samples_per_batch - 1) // self.max_samples_per_batch


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> List[Trajectory]:
    """Custom collate function that creates a list of Trajectory objects.
    
    Args:
        batch: List of dictionaries containing trajectory data
        
    Returns:
        List of Trajectory objects
    """
    return [
        Trajectory(
            states=item["states"],
            actions=item["actions"],
            length=int(item["length"].item()),
        )
        for item in batch
    ] 