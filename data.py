from pathlib import Path

import torch

from bpe_tokenizer import BPETokenizer
from tokenizer import CharacterTokenizer


ROOT = Path(__file__).resolve().parent


def load_shakespeare(root=ROOT, tokenizer_type='char', bpe_tokenizer_path=None):
    if tokenizer_type not in ('char', 'bpe'):
        raise ValueError("tokenizer_type must be 'char' or 'bpe'")
    if tokenizer_type == 'char' and bpe_tokenizer_path is not None:
        raise ValueError('A BPE tokenizer path requires tokenizer_type=bpe')
    train_text = (root / 'shakespeare_train.txt').read_text(encoding='utf-8')
    val_text = (root / 'shakespeare_val.txt').read_text(encoding='utf-8')
    if tokenizer_type == 'char':
        tokenizer = CharacterTokenizer.from_text(train_text + val_text)
    else:
        path = Path(bpe_tokenizer_path) if bpe_tokenizer_path is not None else root / 'bpe_tokenizer.json'
        tokenizer = BPETokenizer.load(path)
    train = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return tokenizer, train, val


def get_batch(data, block_size, batch_size, device, generator=None):
    if len(data) <= block_size:
        raise ValueError('Dataset must contain more tokens than context size')
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[start:start + block_size] for start in starts.tolist()])
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts.tolist()])
    return x.to(device), y.to(device)
