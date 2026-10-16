from pathlib import Path
import torch

text = Path("sample.txt").read_text(encoding="utf-8")

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 8

x = data[:block_size]
y = data[1:block_size + 1]

print("Input IDs:")
print(x)

print()
print("Target IDs:")
print(y)

print()
print("Input text:")
print(repr(decode(x.tolist())))

print()
print("Target text:")
print(repr(decode(y.tolist())))

print()
print("Prediction pairs:")

for i in range(block_size):
    input_text = decode(x[:i + 1].tolist())
    target_char = decode([y[i].item()])

    print(repr(input_text), "->", repr(target_char))
