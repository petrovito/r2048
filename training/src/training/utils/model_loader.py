import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..models.policy_network import PolicyNetwork
from ..utils.config import TrainingConfig

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(
        self,
        config: TrainingConfig
    ):
        """Initialize the model loader.
        
        Args:
            config: Training configuration
            load_model_path: Optional path to an existing model to load
        """
        self.config: TrainingConfig = config
        self.model_path: Path = Path(config.model_save_path)
        self.model: PolicyNetwork = None

    def load_model(self) -> PolicyNetwork:
        self.model = PolicyNetwork()
        if self.model_path.exists():
            # Load existing model
            self.model.load_state_dict(torch.load(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        return self.model

    def save_model(self) -> None:
        """Save the model parameters as a PyTorch file.
        
        Args:
            save_path: Path to save the PyTorch file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")



            

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