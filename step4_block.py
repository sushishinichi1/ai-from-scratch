import torch
import torch.nn as nn

torch.manual_seed(42)

device = "cuda"

batch_size = 2
block_size = 8
embedding_dim = 32
num_heads = 4


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x):
        T = x.size(1)

        mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        attn_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        # Residual connection
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)

        # Residual connection
        x = self.norm2(x + ff_output)

        return x


x = torch.randn(
    batch_size,
    block_size,
    embedding_dim,
    device=device
)

block = TransformerBlock().to(device)

output = block(x)

print("Input shape:")
print(x.shape)

print()

print("Output shape:")
print(output.shape)

print()

print("Transformer Block parameters:")
print(sum(p.numel() for p in block.parameters()))

print()

print("GPU memory:")
print(
    round(torch.cuda.memory_allocated() / 1024**2, 2),
    "MB"
)
