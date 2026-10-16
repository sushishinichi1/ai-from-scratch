from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = "cuda"

train_text = Path("train_large.txt").read_text(encoding="utf-8")
val_text = Path("val.txt").read_text(encoding="utf-8")

all_text = train_text + val_text

chars = sorted(set(all_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

train_data = torch.tensor(
    [stoi[c] for c in train_text],
    dtype=torch.long
)

val_data = torch.tensor(
    [stoi[c] for c in val_text],
    dtype=torch.long
)

block_size = 64
batch_size = 32

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


def get_batch(data):
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


@torch.no_grad()
def estimate_loss(model, data, batches=20):
    model.eval()

    losses = []

    for _ in range(batches):
        x, y = get_batch(data)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


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

for step in range(1001):

    x, y = get_batch(train_data)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        train_loss = estimate_loss(
            model,
            train_data
        )

        val_loss = estimate_loss(
            model,
            val_data
        )

        print(
            f"Step {step:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
    "embedding_dim": embedding_dim,
    "num_heads": num_heads,
    "num_layers": num_layers,
}

torch.save(
    checkpoint,
    "tinygpt_large.pt"
)

print()
print("Saved: tinygpt_large.pt")

print(
    "GPU memory:",
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2
    ),
    "MB"
)
