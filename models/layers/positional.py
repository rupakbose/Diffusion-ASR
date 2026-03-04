import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model: Dimension of the embeddings
        max_seq_length: Maximum sequence length
    """
    def __init__(self, d_model: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]