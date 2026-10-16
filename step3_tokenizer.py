from pathlib import Path

text = Path("sample.txt").read_text(encoding="utf-8")

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

print("Characters:", len(text))
print("Vocabulary size:", len(chars))

print()
print("Original:")
print(repr(text[:30]))

encoded = encode(text[:30])

print()
print("Token IDs:")
print(encoded)

print()
print("Decoded:")
print(repr(decode(encoded)))

print()
print("Vocabulary:")
for ch in chars:
    print(repr(ch), "->", stoi[ch])
