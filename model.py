import torch
import torch.nn as nn
import math

# Device configuration: CPU or CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = embed_dim // num_heads
        self.h = num_heads

        # Learnable linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, mask=None):
        B, T, C = query.size()  # Batch, SeqLen, EmbedDim
        # Linear projections and reshape for multi-head
        Q = self.W_Q(query).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, attn = self.attention(Q, K, V, mask)
        # Concat heads
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.fc(x)
        x = self.dropout(x)
        # Residual connection + LayerNorm
        x = self.layernorm(x + query)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layernorm(x + residual)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Causal mask generator (prevent peeking into future time steps)
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(device)
    return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, mask=None):
        x = self.mha(x, x, x, mask)
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.decoder = nn.Linear(embed_dim, 1)  # Predict next value

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, input_dim)
        if mask is None:
            seq_len = x.size(1)
            mask = generate_square_subsequent_mask(seq_len)
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask)
        # Use the last time-step's representation for prediction
        out = self.decoder(x[:, -1, :])
        return out

# Example instantiation and device move
def build_model():
    model = TransformerModel(
        input_dim=1,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )
    return model.to(device)

if __name__ == "__main__":
    model = build_model()
    print(model)
