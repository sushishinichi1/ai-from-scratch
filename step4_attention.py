import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = "cuda"

text = "hello wo"
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

tokens = torch.tensor(
    [[stoi[c] for c in text]],
    dtype=torch.long,
    device=device
)

vocab_size = len(chars)
embedding_dim = 16
head_size = 16

embedding = nn.Embedding(vocab_size, embedding_dim).to(device)

key = nn.Linear(embedding_dim, head_size, bias=False).to(device)
query = nn.Linear(embedding_dim, head_size, bias=False).to(device)
value = nn.Linear(embedding_dim, head_size, bias=False).to(device)

x = embedding(tokens)

K = key(x)
Q = query(x)
V = value(x)

# 各文字同士の「どれくらい関係があるか」
scores = Q @ K.transpose(-2, -1)

scores = scores / (head_size ** 0.5)

# 未来の文字を見えなくする
T = tokens.shape[1]

mask = torch.tril(
    torch.ones(T, T, device=device)
)

scores = scores.masked_fill(mask == 0, float("-inf"))

weights = F.softmax(scores, dim=-1)

# Attentionの結果
out = weights @ V

print("Text:")
print(text)

print()
print("Attention weights:")
print(torch.round(weights[0] * 100) / 100)

print()
print("Output shape:")
print(out.shape)
