import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding using sine/cosine functions
    for adding positional awareness to token embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class MathSolverTransformer(nn.Module):
    """
    A 10-layer sequence-to-sequence Transformer for math expression solving.
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=10,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=256
    ):
        super().__init__()

        # Embedding layers for source and target
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Final projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        src, tgt: (batch_size, seq_len) token IDs
        Mask/padding: optional for advanced use (prevent seeing future tokens, etc.)
        """
        # Embed and positionally encode
        src_emb = self.src_embedding(src)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Pass through Transformer
        transformer_out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # Project hidden states to logits
        logits = self.fc_out(transformer_out)
        return logits