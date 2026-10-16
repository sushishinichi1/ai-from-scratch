import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda"

checkpoint = torch.load(
    "tinygpt_5m.pt",
    map_location=device
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
embedding_dim = checkpoint["embedding_dim"]
num_heads = checkpoint["num_heads"]
num_layers = checkpoint["num_layers"]


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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully")
print("Parameters:", sum(p.numel() for p in model.parameters()))

prompt = "the "

generated = torch.tensor(
    [[stoi[c] for c in prompt]],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    for _ in range(200):

        input_tokens = generated[:, -block_size:]

        logits = model(input_tokens)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

result = "".join(
    itos[i]
    for i in generated[0].tolist()
)

print()
print("Generated text:")
print("-" * 40)
print(result)
