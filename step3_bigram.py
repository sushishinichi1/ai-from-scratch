from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda"

text = Path("sample.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
vocab_size = len(chars)

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.linear = nn.Linear(16, vocab_size)

    def forward(self, x):
        return self.linear(self.embedding(x))

model = BigramModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

x = data[:-1]
y = data[1:]

print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

for step in range(501):
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

print("\nGenerated text:")

model.eval()
current = torch.tensor([stoi["h"]], device=device)
output = ["h"]

with torch.no_grad():
    for _ in range(100):
        logits = model(current)
        probs = F.softmax(logits[-1], dim=-1)
        next_id = torch.multinomial(probs, 1)
        output.append(itos[next_id.item()])
        current = next_id

print("".join(output))
