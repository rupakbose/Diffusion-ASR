import torch
import torch.nn as nn
from typing import Optional
from .layers import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding

class EncoderLayer(nn.Module):
    """Single encoder layer in the transformer model.
    
    Args:
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward inner layer
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for encoder layer.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Output tensor after self-attention and feed-forward
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """Transformer model implementation for sequence classification.
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary (number of classes)
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Dimension of feed-forward inner layer
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self, 
         
        tgt_vocab_size: int, 
        d_model: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 6, 
        d_ff: int = 64, 
        max_seq_length: int = 128, 
        dropout: float = 0.2
    ):
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.audio_projector = nn.Linear(768, d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.text_projector = nn.Embedding(tgt_vocab_size, d_model)
       

    def forward(self, src: torch.Tensor, maskedtxt: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer model.
        
        Args:
            src: Input tensor of shape (batch_size, 768, seq_length)
            maskedtxt: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, seq_length + 1, tgt_vocab_size)
        """
        src_embedded = self.audio_projector(src)
        text_embedded = self.text_projector(maskedtxt)

        concat_output = torch.cat((src_embedded, text_embedded), dim=1)
        fused_with_position_emb = self.positional_encoding(concat_output)
        enc_output = fused_with_position_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        output = self.fc(enc_output)
        output = {
            'audio_transform': output[:,:src.shape[1], :],
            'text_token': output[:,src.shape[1]:,:]
        }
        return output
    

class TransformerUC(nn.Module):
    """Transformer model implementation for sequence classification.
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary (number of classes)
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Dimension of feed-forward inner layer
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self, 
        
        tgt_vocab_size: int, 
        d_model: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 6, 
        d_ff: int = 64, 
        max_seq_length: int = 128, 
        dropout: float = 0.2
    ):
        super(TransformerUC, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.audio_projector = nn.Linear(768, d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

        self.fc_mu = nn.Linear(d_model, tgt_vocab_size)
        self.fc_sigma = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.text_projector = nn.Embedding(tgt_vocab_size, d_model)
        print(max_seq_length)
    

    def forward(self, src: torch.Tensor, maskedtxt: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer model.
        
        Args:
            src: Input tensor of shape (batch_size, 768, seq_length)
            maskedtxt: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, seq_length + 1, tgt_vocab_size)
        """
        src_embedded = self.audio_projector(src)
        text_embedded = self.text_projector(maskedtxt)
        concat_output = torch.cat((src_embedded, text_embedded), dim=1)
        fused_with_position_emb = self.positional_encoding(concat_output)
        enc_output = fused_with_position_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        output_mu = self.fc_mu(enc_output)[:,src.shape[1]:,:]
        output_sigma = torch.nn.functional.softplus(self.fc_sigma(enc_output))[:,src.shape[1]:,:]
        dist = torch.distributions.Normal(output_mu, torch.exp(output_sigma))
        logit = dist.rsample()
        output_text =  logit
        output = {
            'output_mu': output_mu,
            'output_sigma': output_sigma,
            'text_token': output_text
        }
        return output