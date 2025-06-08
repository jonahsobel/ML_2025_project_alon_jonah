"""
Custom PyTorch Transformer Implementation for Bitcoin Price Prediction
Pure from-scratch implementation without using nn.Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with scaled dot-product attention."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class LayerNorm(nn.Module):
    """Layer Normalization."""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def create_causal_mask(seq_len, device):
    """Create a causal mask for self-attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""
    
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerModel(nn.Module):
    """Complete Transformer Model for Bitcoin Price Prediction."""
    
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        # Input embedding (1D to d_model)
        self.input_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to single value
        self.output_projection = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, 1)
        batch_size, seq_len, _ = x.size()
        
        # Embed input and scale
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, mask)
        
        # Use the last token's representation for prediction
        last_token = x[:, -1, :]  # (batch_size, d_model)
        
        # Project to single output value
        output = self.output_projection(last_token)  # (batch_size, 1)
        
        return output


def build_model(d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                max_len=5000, dropout=0.1):
    """
    Build and return a TransformerModel on the appropriate device.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        
    Returns:
        model: TransformerModel instance on appropriate device
        device: The device (cuda/cpu) the model is on
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    
    print(f"Model built on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device
