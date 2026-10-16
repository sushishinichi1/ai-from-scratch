from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    attention_impl: str = 'torch'

    def __post_init__(self):
        if self.attention_impl not in ('torch', 'custom'):
            raise ValueError("attention_impl must be 'torch' or 'custom'")

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelConfig10M(ModelConfig):
    """Deeper TinyGPT; retain the 5M width, heads, and context size."""

    num_layers: int = 12
    attention_impl: str = 'custom'


@dataclass
class ModelConfig30M(ModelConfig):
    """Wider six-layer TinyGPT with about 31M parameters at vocabulary 1024."""

    embedding_dim: int = 640
    num_heads: int = 8
    num_layers: int = 6
    attention_impl: str = 'custom'


class CausalSelfAttention(nn.Module):
    """Explicit multi-head attention; returns only the projected output.

    Selected with attention_impl='custom'; the default 'torch' implementation
    retains compatibility with existing model state dictionaries.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.embedding_dim
        if config.num_heads <= 0 or dim <= 0 or dim % config.num_heads:
            raise ValueError('embedding_dim must be positive and divisible by num_heads')
        if config.block_size <= 0:
            raise ValueError('block_size must be positive')
        self.embedding_dim = dim
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.block_size = config.block_size
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        # True marks future positions that must be excluded from attention.
        self.register_buffer(
            'causal_mask',
            torch.ones(config.block_size, config.block_size, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.embedding_dim:
            raise ValueError('x must have shape (batch, length, embedding_dim)')
        batch, length, dim = x.shape
        if not 0 < length <= self.block_size:
            raise ValueError('length must be positive and within context size')

        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = self.query(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        # Every query compares with every key within its own head: (B, H, T, T).
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:length, :length], float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v  # (B, H, T, D)

        # Restore token order and concatenate heads: (B, T, H, D) -> (B, T, C).
        concatenated = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.output_projection(concatenated)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embedding_dim
        self.attention_impl = config.attention_impl
        if self.attention_impl == 'torch':
            self.attention = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)
        else:
            self.attention = CausalSelfAttention(config)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.register_buffer('causal_mask', torch.ones(config.block_size, config.block_size, dtype=torch.bool).triu(1), persistent=False)

    def forward(self, x):
        length = x.size(1)
        if self.attention_impl == 'torch':
            attended, _ = self.attention(x, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)
        else:
            attended = self.attention(x)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feed_forward(x))


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.config.block_size:
            raise ValueError('tokens must have shape (batch, length), with length within context size')
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))
