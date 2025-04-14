from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training the policy gradient model."""
    
    # Model parameters
    input_channels: int = 1
    num_actions: int = 4
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 3
    train_val_split: float = 0.8
    
    # Paths
    model_save_path: Path = field(default_factory=lambda: Path("models/policy_network.pt"))
    npz_save_path: Path = field(default_factory=lambda: Path("models/policy_network.npz"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_channels": self.input_channels,
            "num_actions": self.num_actions,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "model_save_path": str(self.model_save_path),
            "npz_save_path": str(self.npz_save_path),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        config_dict["model_save_path"] = Path(config_dict["model_save_path"])
        return cls(**config_dict) 