from dataclasses import dataclass
import torch


@dataclass
class Trajectory:
    """Represents a single game trajectory with states and actions."""
    states: torch.Tensor  # Shape: (length, 4, 4)
    actions: torch.Tensor  # Shape: (length,)
    length: int