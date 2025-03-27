import logging
from pathlib import Path
from typing import Optional

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
        self.model = PolicyNetwork(
            input_channels=self.config.input_channels,
            input_height=4,
            input_width=4,
            num_actions=4,
        )
        
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