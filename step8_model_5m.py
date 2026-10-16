import torch
import torch.nn as nn

device = "cuda"

vocab_size = 26
block_size = 128

embedding_dim = 256
num_heads = 8
num_layers = 6


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

        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)

        x = self.norm2(x + ff_output)

        return x


class TinyGPT5M(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        self.output = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, tokens):
        B, T = tokens.shape

        token_emb = self.token_embedding(tokens)

        positions = torch.arange(
            T,
            device=tokens.device
        )

        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.output(x)


model = TinyGPT5M().to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

test_input = torch.randint(
    0,
    vocab_size,
    (4, 128),
    device=device
)

with torch.no_grad():
    output = model(test_input)

print("Model: TinyGPT5M")
print("Parameters:", parameters)
print(
    "Parameters (millions):",
    round(parameters / 1_000_000, 3)
)

print()
print("Input:")
print(test_input.shape)

print("Output:")
print(output.shape)

print()
print(
    "GPU memory:",
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2
    ),
    "MB"
)
