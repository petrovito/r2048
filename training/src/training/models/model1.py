import torch
import torch.nn as nn

from .base import BaseModel


class Model1(BaseModel):
    """A simple CNN model for 2048 game state prediction."""
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 4,
        input_width: int = 4,
        num_classes: int = 4,
    ):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        self.flatten_size = 128 * (input_height // 8) * (input_width // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_input_shape(self) -> tuple[int, int, int, int]:
        return (1, 1, 4, 4)  # (batch_size, channels, height, width)
    
    def get_output_shape(self) -> tuple[int, int]:
        return (1, 4)  # (batch_size, num_classes) 