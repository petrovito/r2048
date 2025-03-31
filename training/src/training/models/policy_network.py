import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class PolicyNetwork(BaseModel):
    """A simplified CNN model that outputs action probabilities for the 2048 game using policy gradient."""
    
    def __init__(
        self,
    ):
        super().__init__()

        self.input_channels = 1
        self.input_height = 4
        self.input_width = 4
        self.num_actions = 4

        self.conv1_channels = 4
        self.conv2_channels = 16
        self.conv2_padding = 0
        self.fc1_channels = 32
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, self.conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels, kernel_size=3, padding=self.conv2_padding)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Calculate the size of flattened features
        self.flatten_size = self.conv2_channels * 2 * 2 if self.conv2_padding == 0 else self.conv2_channels * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, self.fc1_channels)
        self.fc2 = nn.Linear(self.fc1_channels, self.num_actions)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Output action probabilities
        return F.softmax(x, dim=-1)
    
    def get_input_shape(self) -> tuple[int, int, int, int]:
        return (1, 1, 4, 4)  # (batch_size, channels, height, width)
    
    def get_output_shape(self) -> tuple[int, int]:
        return (1, 4)  # (batch_size, num_actions)

    def tensor_map(self) -> dict[str, torch.Tensor]:
        """Get a mapping of all model parameters to their names.
        
        Returns:
            Dictionary mapping parameter names to their tensors
        """
        return {
            'conv1.weight': self.conv1.weight,
            'conv1.bias': self.conv1.bias,
            'conv2.weight': self.conv2.weight,
            'conv2.bias': self.conv2.bias,
            'fc1.weight': self.fc1.weight,
            'fc1.bias': self.fc1.bias,
            'fc2.weight': self.fc2.weight,
            'fc2.bias': self.fc2.bias,
        } 