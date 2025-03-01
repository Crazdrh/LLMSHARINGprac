import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.shape[0]
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(attn_output + x)  # Residual Connection
        ffn_output = self.ffn(x)
        x = self.norm2(ffn_output + x)  # Residual Connection
        return x

class COAGPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers):
        super(COAGPTDecoder, self).__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

