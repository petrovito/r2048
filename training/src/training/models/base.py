from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models in the training framework."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> tuple[int, int, int, int]:
        """Get the expected input shape of the model.
        
        Returns:
            Tuple of (batch_size, channels, height, width)
        """
        pass
    
    @abstractmethod
    def get_output_shape(self) -> tuple[int, int]:
        """Get the output shape of the model.
        
        Returns:
            Tuple of (batch_size, num_classes)
        """
        pass 