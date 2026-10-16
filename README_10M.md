# TinyGPT 10M with standard BPE 512

The training CLI now defaults to `bpe_512.json`, BPE tokenization, and custom
attention. Vocabulary size is read from the tokenizer. The original 5M model
configuration and checkpoint format remain available.

| Setting | 5M | 10M |
| --- | ---: | ---: |
| Parameters with BPE 512 | 5,034,496 | 9,773,056 |
| Context | 128 | 128 |
| Embedding | 256 | 256 |
| Heads | 8 | 8 |
| Layers | 6 | 12 |
| Batch | 16 | 16 |
| AdamW learning rate | 0.0005 | 0.0005 |

Both variants use float32, seed 42, gradient clipping at 1.0, and evaluate every
100 updates using 10 batches and evaluation seed 12345. They use the same
Shakespeare train/validation splits and model implementation.

Check 10M on CPU without optimizer updates or checkpoint writes:

```bash
python train_shakespeare.py --model-size 10m --cpu-check
```

Train 10M for a total of 1000 updates:

```bash
python train_shakespeare.py --model-size 10m --max-updates 1000
```

The 10M training command first checks a full batch on CPU, including backward,
finite gradients, and validation forward. GPU training starts only after that
check passes. The check preserves weights and the training RNG state.

10M checkpoints are saved in `checkpoints/shakespeare/10m/custom_bpe_512/`.
10M CSV logs are saved in `logs/10m/`. A fresh 10M training run refuses to
overwrite existing 10M checkpoints; use `--resume` to continue one.
`--max-updates` is the total update target, including any resumed updates.

```bash
python train_shakespeare.py --model-size 10m \
  --resume checkpoints/shakespeare/10m/custom_bpe_512/latest.pt \
  --max-updates 1000
```

The 5M command is `python train_shakespeare.py --model-size 5m --max-updates 1000`.
It retains its existing output paths; running it without `--resume` starts fresh
and can replace 5M checkpoints. Existing character/PyTorch-attention training can
be selected explicitly with `--tokenizer char --attention-impl torch`; other BPE
files can still be selected for 5M with `--bpe-tokenizer`.

10M fixes BPE 512 and custom attention. The existing `generate.py` can load both
model sizes directly from their saved configurations.
