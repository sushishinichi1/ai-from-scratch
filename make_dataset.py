import random
from pathlib import Path

random.seed(42)

subjects = [
    "the model",
    "the computer",
    "the transformer",
    "the network",
    "the program",
    "the machine",
    "the student",
    "the researcher",
    "the system",
    "the language model",
]

verbs = [
    "learns",
    "predicts",
    "reads",
    "processes",
    "studies",
    "uses",
    "finds",
    "builds",
    "changes",
    "understands",
]

objects = [
    "text",
    "patterns",
    "language",
    "tokens",
    "data",
    "words",
    "sentences",
    "information",
    "simple rules",
    "new examples",
]

extras = [
    "from data",
    "during training",
    "with attention",
    "step by step",
    "on the computer",
    "using neural networks",
    "from many examples",
    "with small errors",
    "over time",
    "during each step",
]

sentences = []

for _ in range(1500):
    sentence = (
        f"{random.choice(subjects)} "
        f"{random.choice(verbs)} "
        f"{random.choice(objects)} "
        f"{random.choice(extras)}."
    )

    sentences.append(sentence)

random.shuffle(sentences)

split = int(len(sentences) * 0.9)

train_sentences = sentences[:split]
val_sentences = sentences[split:]

train_text = "\n".join(train_sentences) + "\n"
val_text = "\n".join(val_sentences) + "\n"

Path("train_large.txt").write_text(
    train_text,
    encoding="utf-8"
)

Path("val.txt").write_text(
    val_text,
    encoding="utf-8"
)

print("Total sentences:", len(sentences))
print("Train sentences:", len(train_sentences))
print("Validation sentences:", len(val_sentences))

print()
print("Train characters:", len(train_text))
print("Validation characters:", len(val_text))

print()
print("Example training sentences:")

for sentence in train_sentences[:5]:
    print(sentence)
