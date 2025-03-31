import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..models.policy_network import PolicyNetwork
from ..trainer import PolicyGradientTrainer
from ..utils.config import TrainingConfig

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(
        self,
        config: TrainingConfig,
        load_model_path: Optional[Path] = None,
    ):
        """Initialize the model loader.
        
        Args:
            config: Training configuration
            load_model_path: Optional path to an existing model to load
        """
        self.config = config
        self.load_model_path = load_model_path
        self.model = None
        self.trainer = None
        
    def initialize_model(self) -> None:
        """Initialize the model and trainer."""
        # Create model
        self.model = PolicyNetwork()
        
        # Create trainer
        self.trainer = PolicyGradientTrainer(self.model, self.config)
        
        # Load existing model if specified
        if self.load_model_path:
            logger.info(f"Loading model from {self.load_model_path}")
            self.trainer.load_model(self.load_model_path)
            logger.info("Model loaded successfully")
            
    def get_trainer(self) -> PolicyGradientTrainer:
        """Get the initialized trainer.
        
        Returns:
            The initialized PolicyGradientTrainer instance
        """
        if self.trainer is None:
            self.initialize_model()
        return self.trainer

    def save_model_as_npz(self, save_path: Path) -> None:
        """Save the model parameters as an npz file.
        
        Args:
            save_path: Path to save the npz file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        # Get model parameters
        param_dict = self.model.tensor_map()
        
        # Convert tensors to numpy arrays
        np_dict = {
            name: param.detach().cpu().numpy()
            for name, param in param_dict.items()
        }
        
        # Save as npz
        np.savez(save_path, **np_dict)
        logger.info(f"Model saved to {save_path}")


def main():
    """Main function to initialize and save a model."""
    if len(sys.argv) != 2:
        print("Usage: python model_loader.py <save_path>")
        sys.exit(1)
        
    save_path = Path(sys.argv[1])
    
    # Create config with default values
    config = TrainingConfig()
    
    # Initialize model loader
    loader = ModelLoader(config)
    loader.initialize_model()
    
    # Save model
    loader.save_model_as_npz(save_path)


if __name__ == "__main__":
    main() 