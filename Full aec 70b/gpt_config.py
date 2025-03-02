"""
gpt_config.py

Holds the GPT70BConfig dataclass for advanced hyperparameter management.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class GPT70BConfig:
    """
    GPT70BConfig
    ============

    A robust configuration class for a GPT-style, decoder-only Transformer.

    Attributes
    ----------
    vocab_size : int
        The tokenizer's vocab size (deepseek_tokenizer).
    max_seq_len : int
        Maximum sequence length / context window.
    embed_dim : int
        Hidden dimension of token embeddings and attention layers.
    num_heads : int
        Number of multi-head attention heads.
    num_layers : int
        Number of stacked decoder blocks.
    ff_dim : int, optional
        Feed-forward dimension, commonly 4 * embed_dim.
    dropout : float
        Dropout probability.
    layer_norm_epsilon : float
        Epsilon in LayerNorm.
    gradient_checkpointing : bool
        If True, apply gradient checkpointing for memory savings.
    use_cache : bool
        If True, the model can return key/value states for incremental decoding.
    learning_rate : float
        Base learning rate for the optimizer (train.py).
    weight_decay : float
        Weight decay for the optimizer.
    """

    vocab_size: int = 50257
    max_seq_len: int = 2048
    embed_dim: int = 8192
    num_heads: int = 64
    num_layers: int = 80
    ff_dim: Optional[int] = None
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    gradient_checkpointing: bool = False
    use_cache: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.embed_dim

    def summary(self) -> str:
        """
        Return a formatted summary of these hyperparameters.
        """
        msg = (
            f"GPT70BConfig:\n"
            f"  vocab_size           = {self.vocab_size}\n"
            f"  max_seq_len          = {self.max_seq_len}\n"
            f"  embed_dim            = {self.embed_dim}\n"
            f"  num_heads            = {self.num_heads}\n"
            f"  num_layers           = {self.num_layers}\n"
            f"  ff_dim               = {self.ff_dim}\n"
            f"  dropout              = {self.dropout}\n"
            f"  layer_norm_epsilon   = {self.layer_norm_epsilon}\n"
            f"  gradient_checkpointing = {self.gradient_checkpointing}\n"
            f"  use_cache            = {self.use_cache}\n"
            f"  learning_rate        = {self.learning_rate}\n"
            f"  weight_decay         = {self.weight_decay}\n"
        )
        return msg
