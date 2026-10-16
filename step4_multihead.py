import torch
import torch.nn as nn

torch.manual_seed(42)

device = "cuda"

batch_size = 1
block_size = 8
embedding_dim = 32
num_heads = 4

x = torch.randn(
    batch_size,
    block_size,
    embedding_dim,
    device=device
)

attention = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=num_heads,
    batch_first=True
).to(device)

mask = torch.triu(
    torch.ones(block_size, block_size, device=device),
    diagonal=1
).bool()

output, weights = attention(
    x,
    x,
    x,
    attn_mask=mask,
    need_weights=True,
    average_attn_weights=False
)

print("Input shape:")
print(x.shape)

print()

print("Output shape:")
print(output.shape)

print()

print("Attention weights shape:")
print(weights.shape)

print()

for head in range(num_heads):
    print(f"Head {head}:")
    print(
        torch.round(
            weights[0, head] * 100
        ) / 100
    )
    print()
