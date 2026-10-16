import torch
import torch.nn as nn

device = "cuda"

vocab_size = 22
block_size = 64

embedding_dim = 128
num_heads = 4
num_layers = 4


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


class TinyGPT(nn.Module):
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

        logits = self.output(x)

        return logits


model = TinyGPT().to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Model: TinyGPT")
print("Parameters:", parameters)
print("Parameters (millions):", round(parameters / 1_000_000, 3))

test_input = torch.randint(
    0,
    vocab_size,
    (2, 16),
    device=device
)

output = model(test_input)

print()
print("Input shape:")
print(test_input.shape)

print()
print("Output shape:")
print(output.shape)

print()
print("GPU memory:")
print(
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2
    ),
    "MB"
)
