from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import torch


@dataclass
class TrainResult:
    """Stores the results of a training run."""
    train_loss: List[float]
    val_loss: List[float]
    val_exp_length: List[float]
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert the result to a dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_exp_length": self.val_exp_length,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame."""
        return pd.DataFrame(self.to_dict())
    
    
    @classmethod
    def from_csv(cls, path: str) -> "TrainResult":
        """Load a result from a CSV file."""
        df = pd.read_csv(path)
        return cls(
            train_loss=df["train_loss"].tolist(),
            val_loss=df["val_loss"].tolist(),
            val_exp_length=df["val_exp_length"].tolist(),
        )
    
    @property
    def best_val_length(self) -> float:
        """Get the best validation length."""
        return max(self.val_exp_length) if self.val_exp_length else 0
    
    @property
    def best_epoch(self) -> int:
        """Get the epoch with the best validation length."""
        return self.val_exp_length.index(self.best_val_length)

@dataclass
class ValidationMetrics:
    """Stores validation metrics."""
    val_loss: float
    exp_length: float

@dataclass
class Trajectory:
    """Represents a sequence of states, actions, and rewards in a game."""
    states: torch.Tensor
    actions: torch.Tensor
    length: int 