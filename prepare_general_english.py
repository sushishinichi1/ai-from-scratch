"""Prepare a small WikiText-2 corpus and a train-only BPE 512 tokenizer."""

import argparse
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bpe_tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets' / 'general_english'
SOURCE = 'https://huggingface.co/datasets/Salesforce/wikitext'


def paragraph(text):
    text = unicodedata.normalize('NFC', text).strip()
    if len(text) < 150 or re.fullmatch(r'=+.*=+', text):
        return None
    if sum(char.isalpha() for char in text) / len(text) < 0.5:
        return None
    # Undo WikiText's spacing around punctuation without discarding Unicode.
    for marker, replacement in [(' @-@ ', '-'), (' @,@ ', ','), (' @.@ ', '.')]:
        text = text.replace(marker, replacement)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    return text + '\n\n'


def collect_split(split, target_chars, allowed_chars=None):
    paragraphs, row_indices = [], []
    chars = 0
    offset = 0
    skipped_unknown = 0
    while chars < target_chars:
        query = urlencode({'dataset': 'Salesforce/wikitext',
                           'config': 'wikitext-2-raw-v1', 'split': split,
                           'offset': offset, 'length': 100})
        request = Request('https://datasets-server.huggingface.co/rows?' + query,
                          headers={'User-Agent': 'TinyGPT-corpus-preparation/1.0'})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        for row in payload['rows']:
            text = paragraph(row['row']['text'])
            if text is None:
                continue
            # This character BPE has no unknown token. Keep validation separate
            # rather than adding its unseen characters to the training vocabulary.
            if allowed_chars is not None and not set(text) <= allowed_chars:
                skipped_unknown += 1
                continue
            paragraphs.append(text)
            row_indices.append(row['row_idx'])
            chars += len(text)
            if chars >= target_chars:
                break
        offset += len(payload['rows'])
        if offset >= payload['num_rows_total'] and chars < target_chars:
            raise ValueError(f'Not enough usable text in {split}')
    print(f'{split}: {chars:,} characters, {len(paragraphs)} paragraphs', flush=True)
    return ''.join(paragraphs), row_indices, skipped_unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-chars', type=int, default=200_000)
    parser.add_argument('--validation-chars', type=int, default=25_000)
    args = parser.parse_args()
    if min(args.train_chars, args.validation_chars) < 1_000:
        parser.error('Each split must contain at least 1000 characters')
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise FileExistsError(f'Prepared data already exists: {DATA_DIR}')

    train_text, train_rows, _ = collect_split('train', args.train_chars)
    val_text, val_rows, skipped_unknown = collect_split(
        'validation', args.validation_chars, set(train_text))
    tokenizer = BPETokenizer()
    for train_ids in tokenizer._train_steps(train_text, 512):
        if tokenizer.vocab_size % 128 == 0:
            print(f'BPE vocabulary: {tokenizer.vocab_size}', flush=True)
    if tokenizer.vocab_size != 512:
        raise ValueError('Corpus did not produce the requested BPE vocabulary')
    val_ids = tokenizer.encode(val_text)
    if tokenizer.decode(train_ids) != train_text or tokenizer.decode(val_ids) != val_text:
        raise AssertionError('Train/validation encode/decode mismatch')

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(DATA_DIR / 'bpe_512.json')
    loaded = BPETokenizer.load(DATA_DIR / 'bpe_512.json')
    for text, ids in [(train_text, train_ids), (val_text, val_ids)]:
        if loaded.encode(text) != ids or loaded.decode(ids) != text:
            raise AssertionError('Saved tokenizer encode/decode mismatch')
    for name, text, ids in [('train', train_text, train_ids),
                            ('validation', val_text, val_ids)]:
        (DATA_DIR / f'{name}.txt').write_text(text, encoding='utf-8')
        (DATA_DIR / f'{name}_tokens.json').write_text(json.dumps(ids), encoding='utf-8')
    manifest = {
        'source': SOURCE, 'dataset': 'Salesforce/wikitext',
        'config': 'wikitext-2-raw-v1',
        'licenses_as_listed_by_source': ['cc-by-sa-3.0', 'gfdl'],
        'preprocessing': 'NFC, prose paragraphs >=150 chars, WikiText punctuation spacing',
        'split_method': 'Original dataset train/validation splits; leading usable paragraphs',
        'tokenizer_training_split': 'train',
        'validation_paragraphs_skipped_for_unseen_characters': skipped_unknown,
        'train_rows': train_rows, 'validation_rows': val_rows,
        'train_characters': len(train_text), 'validation_characters': len(val_text),
        'corpus_characters': len(train_text) + len(val_text),
        'train_tokens': len(train_ids), 'validation_tokens': len(val_ids),
        'corpus_tokens': len(train_ids) + len(val_ids),
        'vocab_size': loaded.vocab_size, 'encode_decode_exact_match': True,
    }
    (DATA_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {manifest["corpus_characters"]:,} characters / '
          f'{manifest["corpus_tokens"]:,} BPE tokens; exact round-trip OK', flush=True)


if __name__ == '__main__':
    main()
