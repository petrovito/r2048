from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Data parameters
    num_workers: int = 4
    train_val_split: float = 0.8
    
    # Model parameters
    input_channels: int = 1
    input_height: int = 4
    input_width: int = 4
    num_classes: int = 4  # Number of possible moves
    
    # Logging and saving
    use_wandb: bool = True
    model_save_path: Path = Path("models/best_model.pt")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "train_val_split": self.train_val_split,
            "input_channels": self.input_channels,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "num_classes": self.num_classes,
            "use_wandb": self.use_wandb,
            "model_save_path": str(self.model_save_path),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        config_dict["model_save_path"] = Path(config_dict["model_save_path"])
        return cls(**config_dict) 