from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = "cuda"

text = Path("train.txt").read_text(encoding="utf-8")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

data = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)

block_size = 64
batch_size = 8
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

        return self.output(x)


def get_batch():
    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


model = TinyGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Parameters:", parameters)
print()

for step in range(501):
    x, y = get_batch()

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:3d} | Loss: {loss.item():.4f}"
        )

print()
print("GPU memory:")
print(
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2
    ),
    "MB"
)
