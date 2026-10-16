"""Dataset helpers for Japanese instruction tuning with the unchanged Byte-BPE."""

import json
from pathlib import Path

import torch


USER_MARKER = '<|user|>'
ASSISTANT_MARKER = '<|assistant|>'


def format_prompt(question):
    return f'{USER_MARKER}\n{question}\n{ASSISTANT_MARKER}\n'


def load_instruction_examples(path, tokenizer, block_size=128):
    examples = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            prompt = format_prompt(row['question'])
            full_text = prompt + row['answer'] + '\n'
            if row['text'] != full_text:
                raise ValueError(f'Chat format mismatch at line {line_number}')
            # Keep the answer boundary token-exact. A BPE merge must not combine
            # bytes from the user prompt and assistant answer into one target.
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(row['answer'])
            token_ids = prompt_ids + answer_ids
            if tokenizer.decode(tokenizer.encode(full_text)) != full_text:
                raise ValueError(f'Tokenizer round-trip mismatch at line {line_number}')
            serialized_length = len(tokenizer.encode(full_text))
            if tokenizer.decode(token_ids) != prompt + row['answer']:
                raise ValueError(f'Boundary-safe tokenizer round-trip mismatch at line {line_number}')
            if not 1 <= len(token_ids) <= block_size or serialized_length > block_size:
                raise ValueError(f'Instruction at line {line_number} exceeds context size')
            if not answer_ids:
                raise ValueError(f'Empty response at line {line_number}')
            # The target mask is indexed by token position; after shifting,
            # targets at and after len(prompt_ids) are assistant answer tokens.
            loss_mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
            examples.append({
                'id': row['id'], 'category': row['category'],
                'token_ids': token_ids, 'loss_mask': loss_mask,
                'prompt_token_count': len(prompt_ids), 'answer_token_count': len(answer_ids),
                'serialized_token_count': serialized_length,
            })
    if not examples:
        raise ValueError('Instruction dataset is empty')
    return examples


def collate_instruction_batch(examples, indices, device, pad_token_id=0):
    selected = [examples[int(i)] for i in indices]
    max_length = max(len(example['token_ids']) for example in selected)
    if max_length < 2:
        raise ValueError('Each conversation must contain at least two tokens')
    tokens = torch.full((len(selected), max_length), pad_token_id, dtype=torch.long)
    token_loss_mask = torch.zeros((len(selected), max_length), dtype=torch.bool)
    for row_index, example in enumerate(selected):
        length = len(example['token_ids'])
        tokens[row_index, :length] = torch.tensor(example['token_ids'], dtype=torch.long)
        token_loss_mask[row_index, :length] = torch.tensor(example['loss_mask'], dtype=torch.bool)
    x = tokens[:, :-1].to(device)
    y = tokens[:, 1:].to(device)
    y_loss_mask = token_loss_mask[:, 1:].to(device)
    return x, y, y_loss_mask
