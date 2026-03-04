import torch.nn as nn
from typing import Optional
import torch

class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network.
    
    Args:
        d_model: Dimension of input and output
        d_ff: Dimension of inner layer
    """
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Output tensor of same shape as input
        """
        return self.fc2(self.relu(self.fc1(x)))