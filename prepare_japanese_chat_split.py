"""Create a deterministic, near-duplicate-safe train/validation split."""

import difflib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'datasets/japanese_base_v2/chat_instruction_tiny.jsonl'
OUT_DIR = ROOT / 'datasets/japanese_base_v2/chat_instruction_split'
EVAL_QUESTIONS = [
    'アルゴリズムとは何ですか。',
    '天気と気候の違いは何ですか。',
    '友人が手伝ってくれました。何と伝えるとよいですか。',
    'なぜ海の水は塩辛いのですか。',
    'コンピュータのCPUは何をする部品ですか。',
]


def norm(text):
    return re.sub(r'\s+', '', text).lower()


def similar(left, right):
    return difflib.SequenceMatcher(None, norm(left), norm(right)).ratio()


def main():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines()]
    # Place exact/near eval prompts in validation so all five questions remain unseen in train.
    reserved = set()
    eval_matches = {q: [] for q in EVAL_QUESTIONS}
    for i, row in enumerate(rows):
        for question in EVAL_QUESTIONS:
            ratio = similar(row['question'], question)
            if ratio >= 0.82:
                eval_matches[question].append({'id': row['id'], 'question': row['question'], 'similarity': round(ratio, 3)})
                reserved.add(i)

    # Group identical or near-identical full instruction pairs before assigning split.
    groups = []
    for i, row in enumerate(rows):
        placed = False
        for group in groups:
            if any(similar(row['question'] + row['answer'], rows[j]['question'] + rows[j]['answer']) >= 0.92 for j in group):
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    val_indices = set(reserved)
    # Deterministic fill to approximately 10%, without splitting duplicate groups.
    target = max(11, round(len(rows) * 0.10))
    for group in groups:
        if val_indices & set(group):
            val_indices.update(group)
    for group in groups:
        if len(val_indices) >= target:
            break
        if not val_indices.intersection(group):
            val_indices.update(group)

    train_rows = [row for i, row in enumerate(rows) if i not in val_indices]
    val_rows = [row for i, row in enumerate(rows) if i in val_indices]
    # Prove the split has no cross-partition exact/near duplicate pair.
    cross_matches = []
    for train_row in train_rows:
        for val_row in val_rows:
            qsim = similar(train_row['question'], val_row['question'])
            pair_sim = similar(train_row['question'] + train_row['answer'], val_row['question'] + val_row['answer'])
            if qsim >= 0.92 or pair_sim >= 0.92:
                cross_matches.append([train_row['id'], val_row['id'], round(max(qsim, pair_sim), 3)])
    if cross_matches:
        raise RuntimeError(f'Cross-split near duplicates found: {cross_matches[:5]}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in (('train', train_rows), ('validation', val_rows)):
        (OUT_DIR / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in subset),
            encoding='utf-8')
    report = {
        'total': len(rows), 'train': len(train_rows), 'validation': len(val_rows),
        'seed': 42, 'cross_split_near_duplicates': cross_matches,
        'evaluation_question_matches': eval_matches,
    }
    (OUT_DIR / 'split_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
