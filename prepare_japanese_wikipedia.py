"""Prepare an article-disjoint Japanese Wikipedia corpus and own byte BPE."""

import argparse
import html
import json
from pathlib import Path
import random
import re
import shutil
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq

from bpe_tokenizer import BPETokenizer
from custom_byte_bpe_tokenizer import CustomByteBPETokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets' / 'japanese_wikipedia'
SOURCE = 'https://huggingface.co/datasets/wikimedia/wikipedia'
REVISION = 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
SHARD = '20231101.ja/train-00000-of-00015.parquet'


def clean_body(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'<(ref|script|style)\b[^>]*>.*?</\1\s*>', '', text, flags=re.S | re.I)
    text = re.sub(r'<[A-Za-z/!][^>]*>', '', text)
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.S)
    while re.search(r'\{\{[^{}]*\}\}', text):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:ファイル|画像|File|Image|Category|カテゴリ):[^\]]*\]\]', '', text, flags=re.I)
    text = re.sub(r'\[\[([^\[\]]*)\]\]', lambda match: match[1].split('|')[-1], text)
    text = re.sub(r'\[https?://\S+(?:\s+([^\]]*))?\]', lambda match: match[1] or '', text)
    text = re.sub(r"'{2,5}", '', text)
    text = re.sub(r'^\s*=+\s*(.*?)\s*=+\s*$', r'\1', text, flags=re.M)
    text = html.unescape(text)
    # The upstream corpus is already parsed; remove obvious residual markup lines.
    lines = [line.rstrip() for line in text.splitlines()
             if not re.search(r'\{\{|\}\}|\[\[|\]\]|^\s*(?:\{\||\|\}|\|[-+])', line)]
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-characters', type=int, default=15_000_000)
    args = parser.parse_args()
    if not 10_000_000 <= args.target_characters <= 20_000_000:
        parser.error('Use a target between 10M and 20M characters')
    if any((DATA_DIR / name).exists() for name in ['manifest.json', 'train.txt', 'bpe_4096.json']):
        raise FileExistsError('Prepared Japanese data already exists')
    raw_path = DATA_DIR / 'raw' / Path(SHARD).name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        url = f'{SOURCE}/resolve/{REVISION}/{SHARD}?download=true'
        with urlopen(Request(url, headers={'User-Agent': 'TinyGPT-Japanese/1.0'}), timeout=60) as response:
            with raw_path.with_suffix('.download').open('wb') as output:
                shutil.copyfileobj(response, output)
        raw_path.with_suffix('.download').replace(raw_path)

    articles, seen_ids, seen_bodies = [], set(), set()
    characters = 0
    parquet = pq.ParquetFile(raw_path)
    for batch in parquet.iter_batches(batch_size=128, columns=['id', 'url', 'title', 'text']):
        for row in batch.to_pylist():
            article_id = str(row['id'])
            body = clean_body(row['text'])
            if (len(body) < 200 or article_id in seen_ids or body in seen_bodies
                    or '曖昧さ回避' in row['title']):
                continue
            seen_ids.add(article_id)
            seen_bodies.add(body)
            articles.append({'id': article_id, 'url': row['url'], 'title': row['title'], 'text': body})
            characters += len(body)
            if characters >= args.target_characters:
                break
        if characters >= args.target_characters:
            break
    if characters < args.target_characters:
        raise ValueError('Selected source shard contains insufficient usable text')
    random.Random(42).shuffle(articles)
    validation_count = max(1, len(articles) // 10)
    records = {'validation': articles[:validation_count], 'train': articles[validation_count:]}
    ids = {split: {row['id'] for row in rows} for split, rows in records.items()}
    if ids['train'] & ids['validation']:
        raise AssertionError('Article ID overlap between train and validation')
    if {row['text'] for row in records['train']} & {row['text'] for row in records['validation']}:
        raise AssertionError('Duplicate article body across splits')
    texts = {}
    for split, rows in records.items():
        text = '\n\n'.join(row['text'] for row in rows) + '\n\n'
        texts[split] = text
        (DATA_DIR / f'{split}.txt').write_bytes(text.encode('utf-8'))
        with (DATA_DIR / f'{split}_articles.jsonl').open('x', encoding='utf-8') as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'{split}: {len(rows):,} articles, {len(text):,} characters', flush=True)
    del articles, records, seen_bodies

    print('Training own byte BPE 4096 on train only...', flush=True)
    tokenizer = CustomByteBPETokenizer()
    train_ids = tokenizer.train_with_ids(texts['train'], 4096)
    tokenizer.save(DATA_DIR / 'bpe_4096.json')
    loaded = BPETokenizer.load(DATA_DIR / 'bpe_4096.json')
    if loaded.to_dict() != tokenizer.to_dict():
        raise AssertionError('Tokenizer JSON mismatch')
    # Compare the independent heap encoder against the training merge pass.
    encoded_train = loaded.encode_array(texts['train'])
    if not np.array_equal(train_ids, encoded_train):
        raise AssertionError('Training merges and saved-tokenizer encoding differ')
    del encoded_train
    validation_ids = loaded.encode_array(texts['validation'])
    for split, token_ids in [('train', train_ids), ('validation', validation_ids)]:
        if loaded.decode(token_ids, errors='strict') != texts[split]:
            raise AssertionError(f'{split} UTF-8 round-trip mismatch')
        np.save(DATA_DIR / f'{split}_tokens.npy', token_ids, allow_pickle=False)
        print(f'{split}: {len(token_ids):,} tokens; exact UTF-8 round-trip OK', flush=True)
    for sample in ['日本語とEnglish、記号：±∞✓。絵文字🙂🚀👩\u200d💻', '\x00\t\nテスト\r\n',
                   '未学習でも𠮷野家と🦄🎉を表現できる']:
        if loaded.decode(loaded.encode(sample), errors='strict') != sample:
            raise AssertionError('Multilingual/symbol/emoji round-trip mismatch')
    manifest = {
        'source': SOURCE, 'source_revision': REVISION, 'config': '20231101.ja',
        'source_shard': SHARD, 'licenses_as_listed_by_source': ['cc-by-sa-3.0', 'gfdl'],
        'selection': 'Leading usable whole articles until target, then seeded article shuffle',
        'seed': 42, 'validation_article_fraction': 0.1, 'article_overlap': 0,
        'duplicate_article_body_overlap': 0,
        'train_article_ids': sorted(ids['train']), 'validation_article_ids': sorted(ids['validation']),
        'train_characters': len(texts['train']), 'validation_characters': len(texts['validation']),
        'train_tokens': len(train_ids), 'validation_tokens': len(validation_ids),
        'vocab_size': loaded.vocab_size, 'base_vocabulary_size': 256,
        'tokenizer': 'custom-byte-level-bpe', 'merge_training_split': 'train only',
        'preprocessing': 'Strip residual HTML, wiki templates/tables/links and markup',
        'encode_decode_exact_match': True,
    }
    (DATA_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Japanese corpus and own byte BPE ready.', flush=True)


if __name__ == '__main__':
    main()
