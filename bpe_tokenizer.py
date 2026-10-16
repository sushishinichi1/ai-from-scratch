"""Character-based BPE using only the Python standard library.

Run ``python bpe_tokenizer.py --vocab-size 512`` for one vocabulary, or
``python bpe_tokenizer.py --compare`` to compare 256, 512 and 1024 on CPU.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time


class BPETokenizer:
    def __init__(self):
        self.vocab = []
        self.merges = []  # (left_id, right_id, merged_id), in training order
        self.stoi = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _merge(ids, pair, new_id):
        """Replace non-overlapping occurrences, from left to right."""
        result = []
        index = 0
        while index < len(ids):
            if (index + 1 < len(ids)
                    and ids[index] == pair[0] and ids[index + 1] == pair[1]):
                result.append(new_id)
                index += 2
            else:
                result.append(ids[index])
                index += 1
        return result

    def train(self, text, vocab_size):
        for _ in self._train_steps(text, vocab_size):
            pass
        return self

    def _train_steps(self, text, vocab_size):
        """Yield merged training IDs so comparisons can share the merge prefix."""
        if not text:
            raise ValueError('Training text must not be empty')
        chars = sorted(set(text))
        if type(vocab_size) is not int or vocab_size < len(chars):
            raise ValueError('vocab_size must be an integer at least as large as the character vocabulary')
        self.vocab = chars
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.merges = []
        ids = [self.stoi[char] for char in text]
        yield ids
        while self.vocab_size < vocab_size:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break  # No adjacent tokens remain; the target is an upper bound.
            # Resolve equal frequencies by token IDs for reproducible training.
            pair = min(counts, key=lambda pair: (-counts[pair], pair))
            new_id = self.vocab_size
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            self.merges.append((*pair, new_id))
            ids = self._merge(ids, pair, new_id)
            yield ids

    def encode(self, text):
        if not self.vocab:
            raise ValueError('Train or load the tokenizer before encoding')
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f'Characters outside vocabulary: {sorted(unknown)!r}')
        ids = [self.stoi[char] for char in text]
        for left, right, new_id in self.merges:
            ids = self._merge(ids, (left, right), new_id)
        return ids

    def decode(self, ids):
        pieces = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
                raise ValueError(f'Invalid token ID: {token_id!r}')
            pieces.append(self.vocab[token_id])
        return ''.join(pieces)

    def to_dict(self):
        if not self.vocab:
            raise ValueError('Train or load the tokenizer before saving')
        return {'version': 1, 'vocab': list(self.vocab),
                'merges': [list(merge) for merge in self.merges]}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload):
        """Restore the vocabulary and ordered merges from JSON or a checkpoint."""
        if isinstance(payload, dict) and payload.get('version') == 3:
            from custom_byte_bpe_tokenizer import CustomByteBPETokenizer
            return CustomByteBPETokenizer.from_dict(payload)
        if isinstance(payload, dict) and payload.get('version') == 2:
            from byte_bpe_tokenizer import ByteBPETokenizer
            return ByteBPETokenizer.from_dict(payload)
        if not isinstance(payload, dict) or payload.get('version') != 1:
            raise ValueError('Unsupported tokenizer format version')
        vocab = payload.get('vocab')
        merges = payload.get('merges')
        if (not isinstance(vocab, list) or not vocab
                or any(not isinstance(token, str) or not token for token in vocab)
                or not isinstance(merges, list)):
            raise ValueError('Invalid vocabulary or merges')
        base_size = len(vocab) - len(merges)
        chars = vocab[:base_size]
        if base_size <= 0 or any(len(char) != 1 for char in chars) or len(set(chars)) != base_size:
            raise ValueError('Invalid character vocabulary')
        for new_id, merge in enumerate(merges, start=base_size):
            if (not isinstance(merge, list) or len(merge) != 3
                    or any(type(value) is not int for value in merge)):
                raise ValueError('Invalid merge')
            left, right, stored_id = merge
            if (stored_id != new_id or not 0 <= left < new_id or not 0 <= right < new_id
                    or vocab[new_id] != vocab[left] + vocab[right]):
                raise ValueError('Invalid merge order or vocabulary')
        tokenizer = cls()
        tokenizer.vocab = list(vocab)
        tokenizer.merges = [tuple(merge) for merge in merges]
        tokenizer.stoi = {char: index for index, char in enumerate(chars)}
        return tokenizer


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', type=Path, default=root / 'shakespeare.txt')
    parser.add_argument('--train-text', type=Path, default=root / 'shakespeare_train.txt')
    parser.add_argument('--val-text', type=Path, default=root / 'shakespeare_val.txt')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--vocab-size', type=int, default=256, help='Target vocabulary size, e.g. 256, 512 or 1024')
    mode.add_argument('--compare', action='store_true', help='Train and compare 256, 512 and 1024 using the same corpus')
    parser.add_argument('--output', type=Path, help='Single-tokenizer output path (default: bpe_<vocab_size>.json)')
    parser.add_argument('--output-dir', type=Path, default=root, help='Directory for vocabulary JSONs and comparison CSV')
    args = parser.parse_args(argv)
    if args.vocab_size <= 0:
        parser.error('--vocab-size must be positive')
    if args.compare and args.output is not None:
        parser.error('--output is for a single vocabulary; use --output-dir with --compare')
    return args


def model_parameter_count(vocab_size):
    # BPE itself uses only the standard library. Import the model only for reporting.
    from model import ModelConfig, TinyGPT

    model = TinyGPT(ModelConfig(vocab_size, attention_impl='custom'))
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    from tokenizer import CharacterTokenizer

    args = parse_args()
    text = args.text.read_text(encoding='utf-8')
    train_text = args.train_text.read_text(encoding='utf-8')
    val_text = args.val_text.read_text(encoding='utf-8')
    if not train_text or not val_text:
        raise ValueError('Train and validation text must not be empty')
    sizes = [256, 512, 1024] if args.compare else [args.vocab_size]
    character = CharacterTokenizer.from_text(text)
    if min(sizes) < character.vocab_size:
        raise ValueError('Target vocabulary is smaller than the character vocabulary')
    train_char_count = len(character.encode(train_text))
    character.encode(val_text)  # Check coverage before writing any output.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bpe = BPETokenizer()
    rows = []
    started = time.perf_counter()
    print(f'Training BPE on CPU: {args.text.name}, target vocabularies={sizes}', flush=True)
    print(f'Train characters / CharacterTokenizer tokens: {train_char_count:,}', flush=True)

    def record(size, full_ids):
        path = args.output if args.output is not None else args.output_dir / f'bpe_{size}.json'
        if bpe.decode(full_ids) != text:
            raise AssertionError('Training corpus round-trip mismatch')
        bpe.save(path)
        loaded = BPETokenizer.load(path)
        if loaded.to_dict() != bpe.to_dict():
            raise AssertionError('JSON vocabulary or merge order mismatch')
        train_ids = loaded.encode(train_text)
        val_ids = loaded.encode(val_text)
        if loaded.decode(train_ids) != train_text or loaded.decode(val_ids) != val_text:
            raise AssertionError('Train/validation encode -> decode mismatch')
        row = {
            'requested_vocab_size': size,
            'vocab_size': loaded.vocab_size,
            'train_tokens': len(train_ids),
            'validation_tokens': len(val_ids),
            'train_char_tokens': train_char_count,
            'compression_ratio': len(train_ids) / train_char_count,
            'model_parameters': model_parameter_count(loaded.vocab_size),
            'tokenizer_file': str(path),
        }
        rows.append(row)
        print(f'Saved {path.name}: vocab={loaded.vocab_size}, train tokens={len(train_ids):,}, '
              f'compression ratio={row["compression_ratio"]:.2%}, parameters={row["model_parameters"]:,}', flush=True)
        print('CPU encode -> decode: exact match for train/validation after JSON save/load', flush=True)
        if not args.compare:
            print(f'Original characters: {len(text):,} | Full corpus BPE tokens: {len(full_ids):,}')
            print('First 20 merges:')
            for rank, (left, right, new_id) in enumerate(bpe.merges[:20], start=1):
                print(f'   {rank:2}. {bpe.vocab[left]!r} + {bpe.vocab[right]!r} -> {bpe.vocab[new_id]!r} (ID {new_id})')

    # Smaller vocabularies are prefixes of the same deterministic BPE training.
    for full_ids in bpe._train_steps(text, max(sizes)):
        if bpe.vocab_size % 64 == 0:
            print(f'CPU training: vocab={bpe.vocab_size}, elapsed={time.perf_counter() - started:.1f}s', flush=True)
        if bpe.vocab_size in sizes:
            record(bpe.vocab_size, full_ids)
    for size in sizes:
        if not any(row['requested_vocab_size'] == size for row in rows):
            print(f'No pairs remain: requested vocabulary={size}, actual={bpe.vocab_size}', flush=True)
            record(size, full_ids)
    print('\nvocab_size | train tokens | compression (BPE / char) | model parameters')
    for row in rows:
        print(f'{row["vocab_size"]:10} | {row["train_tokens"]:12,} | {row["compression_ratio"]:24.2%} | {row["model_parameters"]:16,}')
    if args.compare:
        report = args.output_dir / 'bpe_vocab_comparison.csv'
        with report.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Comparison saved: {report}', flush=True)


if __name__ == '__main__':
    main()
