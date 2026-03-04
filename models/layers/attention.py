import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism as described in 'Attention is All You Need' paper.
    
    Args:
        d_model: Dimension of the input embeddings
        num_heads: Number of parallel attention heads
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional mask tensor
            
        Returns:
            Output tensor after attention operation
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the input tensor into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple heads back to original shape.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_length, d_k)
            
        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for multi-head attention.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional mask tensor
            
        Returns:
            Output tensor after multi-head attention
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output