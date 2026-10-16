"""Prepare every WikiText-2 raw train/validation row and train BPE 512."""

import argparse
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq

from bpe_tokenizer import BPETokenizer
from byte_bpe_tokenizer import ByteBPETokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets' / 'general_english_wikitext2_full'
SOURCE = 'https://huggingface.co/datasets/Salesforce/wikitext'
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
EXPECTED_ROWS = {'train': 36_718, 'validation': 3_760}


def load_full_split(data_dir, split):
    filename = f'{split}-00000-of-00001.parquet'
    url = f'{SOURCE}/resolve/{REVISION}/wikitext-2-raw-v1/{filename}?download=true'
    path = data_dir / 'raw' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f'Downloading full {split} file...', flush=True)
        temporary = path.with_suffix('.download')
        request = Request(url, headers={'User-Agent': 'TinyGPT-full-corpus/1.0'})
        with urlopen(request, timeout=30) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    table = pq.read_table(path, columns=['text'])
    if table.num_rows != EXPECTED_ROWS[split]:
        raise ValueError(f'Incomplete {split}: {table.num_rows} rows')
    rows = table.column('text').to_pylist()
    if not all(isinstance(text, str) for text in rows):
        raise ValueError(f'Non-text row in {split}')
    # Keep every row exactly as supplied, including titles and punctuation.
    text = ''.join(rows)
    print(f'{split}: {len(rows):,}/{EXPECTED_ROWS[split]:,} rows, '
          f'{len(text):,} characters', flush=True)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    # Downloads can be reused; completed datasets are never overwritten.
    for filename in ['manifest.json', 'bpe_512.json', 'train_tokens.npy', 'validation_tokens.npy']:
        if (data_dir / filename).exists():
            raise FileExistsError(f'Prepared output already exists: {data_dir / filename}')
    texts = {split: load_full_split(data_dir, split) for split in EXPECTED_ROWS}
    for split, text in texts.items():
        path = data_dir / f'{split}.txt'
        encoded = text.encode('utf-8')
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f'Existing corpus differs from source: {path}')
        if not path.exists():
            path.write_bytes(encoded)

    print('Training byte-level BPE 512 on the entire train split (CPU)...', flush=True)
    tokenizer = ByteBPETokenizer().train(texts['train'], 512)
    if tokenizer.vocab_size != 512:
        raise ValueError(f'Unexpected vocabulary size: {tokenizer.vocab_size}')
    tokenizer.save(data_dir / 'bpe_512.json')
    loaded = BPETokenizer.load(data_dir / 'bpe_512.json')
    if loaded.to_dict() != tokenizer.to_dict():
        raise AssertionError('Saved tokenizer differs from trained tokenizer')
    token_counts = {}
    for split, text in texts.items():
        ids = loaded.encode(text)
        if loaded.decode(ids) != text:
            raise AssertionError(f'{split} encode/decode is not an exact match')
        np.save(data_dir / f'{split}_tokens.npy', np.asarray(ids, dtype=np.int32),
                allow_pickle=False)
        token_counts[split] = len(ids)
        print(f'{split}: {len(ids):,} BPE tokens; exact encode/decode match', flush=True)
        del ids
    manifest = {
        'source': SOURCE, 'source_revision': REVISION,
        'dataset': 'Salesforce/wikitext', 'config': 'wikitext-2-raw-v1',
        'licenses_as_listed_by_source': ['cc-by-sa-3.0', 'gfdl'],
        'train_rows': EXPECTED_ROWS['train'], 'validation_rows': EXPECTED_ROWS['validation'],
        'split_method': 'Entire original train and validation splits',
        'preprocessing': 'Concatenate all source row texts without normalization or filtering',
        'tokenizer_training_split': 'Entire train split only',
        'tokenizer_algorithm': 'Byte-level BPE, complete 256-byte alphabet',
        'tokenizer_encoding': 'Encode every line with original line endings retained',
        'train_characters': len(texts['train']),
        'validation_characters': len(texts['validation']),
        'corpus_characters': sum(map(len, texts.values())),
        'train_tokens': token_counts['train'], 'validation_tokens': token_counts['validation'],
        'corpus_tokens': sum(token_counts.values()),
        'vocab_size': loaded.vocab_size, 'encode_decode_exact_match': True,
    }
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Full corpus and train-only BPE 512 ready.', flush=True)


if __name__ == '__main__':
    main()
