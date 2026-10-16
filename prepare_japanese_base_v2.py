"""Clean Japanese Base v2 sources; optionally compare Byte-BPE 1024 on CPU."""

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np

from bpe_tokenizer import BPETokenizer
from custom_byte_bpe_tokenizer import CustomByteBPETokenizer

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'datasets/japanese_base_v2'
WIKIPEDIA = ROOT / 'datasets/japanese_wikipedia'


def compare_tokenizers():
    manifest_path = OUTPUT / 'cleaning_summary.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Complete and verify wiki cleaning before comparing tokenizers')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    texts = {split: (OUTPUT / f'{split}.txt').read_bytes().decode('utf-8') for split in ('train', 'validation')}
    for split, text in texts.items():
        manifest[f'{split}_characters'] = len(text)
    existing_path = OUTPUT / 'bpe_1024_existing.json'
    if not existing_path.exists():
        shutil.copyfile(WIKIPEDIA / 'bpe_1024.json', existing_path)
    existing = BPETokenizer.load(existing_path)
    comparison = {'existing': {}, 'new': {}, 'vocab_size': 1024, 'base_vocabulary_size': 256,
                  'new_tokenizer_merge_training_split': 'train only', 'round_trip_exact': True}
    for split in ('train', 'validation'):
        ids = existing.encode_array(texts[split])
        assert existing.decode(ids, errors='strict') == texts[split]
        comparison['existing'][f'{split}_tokens'] = len(ids)
        np.save(OUTPUT / f'{split}_tokens_existing.npy', ids, allow_pickle=False)
        print(f'Existing tokenizer / {split}: {len(ids):,} tokens; exact round-trip passed', flush=True)
        del ids
        gc.collect()
    new_path = OUTPUT / 'bpe_1024.json'
    if new_path.exists():
        raise FileExistsError('New Base v2 tokenizer already exists')
    print('Training new custom Byte-BPE 1024 on Base v2 train only (CPU)...', flush=True)
    tokenizer = CustomByteBPETokenizer()
    merged_train = tokenizer.train_with_ids(texts['train'], 1024)
    assert tokenizer.vocab_size == 1024 and tokenizer.vocab[:256] == [bytes([i]) for i in range(256)]
    tokenizer.save(new_path)
    tokenizer = BPETokenizer.load(new_path)
    for split in ('train', 'validation'):
        ids = tokenizer.encode_array(texts[split])
        if split == 'train':
            assert np.array_equal(ids, merged_train)
            del merged_train
        assert tokenizer.decode(ids, errors='strict') == texts[split]
        comparison['new'][f'{split}_tokens'] = len(ids)
        np.save(OUTPUT / f'{split}_tokens.npy', ids, allow_pickle=False)
        print(f'New tokenizer / {split}: {len(ids):,} tokens; exact round-trip passed', flush=True)
        del ids
        gc.collect()
    for kind in ('existing', 'new'):
        comparison[kind]['total_tokens'] = comparison[kind]['train_tokens'] + comparison[kind]['validation_tokens']
    comparison['recommended_tokenizer'] = 'new' if comparison['new']['validation_tokens'] < comparison['existing']['validation_tokens'] else 'existing'
    manifest['tokenizer_comparison'] = comparison
    manifest['tokenizer_training_performed'] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULTS ' + json.dumps({'total_characters': manifest['total_characters'],
                                'train_characters': manifest['train_characters'],
                                'validation_characters': manifest['validation_characters'],
                                'tokenizer_comparison': comparison}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenize-only', action='store_true')
    args = parser.parse_args()
    if args.tokenize_only:
        compare_tokenizers()
    else:
        from clean_japanese_base_v2 import main as clean_sources
        clean_sources()
