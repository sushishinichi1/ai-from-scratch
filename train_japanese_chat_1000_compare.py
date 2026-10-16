"""Compare two short assistant-only SFT learning rates on the Japanese 30M Base."""

import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from chat_data import collate_instruction_batch, format_prompt, load_instruction_examples
from generate import generate
from model import ModelConfig, TinyGPT


ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'checkpoints/japanese_base_v2/wide30m_lr0003_bpe1024/best.pt'
TOKENIZER = ROOT / 'datasets/japanese_base_v2/bpe_1024.json'
DATA = ROOT / 'datasets/japanese_base_v2/chat_1000'
OUT = ROOT / 'checkpoints/japanese_base_v2/chat30m_1000'
LOG = ROOT / 'logs/japanese_base_v2/chat30m_1000'
SEED = 42
BATCH = 8
UPDATES = 100
EVAL_EVERY = 20
TEST_QUESTIONS = json.loads((DATA / 'fixed_test_questions.json').read_text(encoding='utf-8'))


def response_loss(model, batch, reduction='mean'):
    x, y, mask = batch
    logits = model(x)
    losses = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                             y.reshape(-1), reduction='none').view_as(y)
    if not mask.any():
        raise ValueError('Batch has no assistant answer targets')
    chosen = losses.masked_select(mask)
    return chosen.sum() if reduction == 'sum' else chosen.mean()


def evaluate(model, examples):
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), BATCH):
            ids = list(range(start, min(start + BATCH, len(examples))))
            batch = collate_instruction_batch(examples, ids, 'cuda')
            loss = response_loss(model, batch, 'sum')
            count = int(batch[2].sum().item())
            total += loss.item()
            tokens += count
    model.train(was_training)
    value = total / tokens
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite validation/train loss')
    return value


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def ngram_repetition(text, n):
    chars = [c for c in text if not c.isspace()]
    grams = [tuple(chars[i:i+n]) for i in range(max(0, len(chars)-n+1))]
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def generate_test(model, tokenizer):
    model.eval()
    rows = []
    for question in TEST_QUESTIONS:
        prompt = format_prompt(question)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        full_text = generate(model, tokenizer, prompt, max_new_tokens=120,
                             temperature=0.7, top_k=40)
        continuation = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        answer = continuation.split('<|user|>', 1)[0].split('<|assistant|>', 1)[0].strip()
        rows.append({
            'question': question, 'answer': answer,
            'temperature': 0.7, 'top_k': 40, 'max_new_tokens': 120, 'seed': SEED,
            'utf8_safe': '\ufffd' not in full_text,
            'character_count': len(answer),
            '3gram_repetition_rate': ngram_repetition(answer, 3),
            '4gram_repetition_rate': ngram_repetition(answer, 4),
        })
    return rows


def run_condition(lr, suffix, train_examples, validation_examples, base_state, tokenizer, config):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = TinyGPT(config)
    model.load_state_dict(base_state)
    model.to(device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01)
    batch_rng = torch.Generator().manual_seed(SEED)
    history = [{'update': 0, 'train_loss': evaluate(model, train_examples),
                'validation_loss': evaluate(model, validation_examples)}]
    best_loss = history[0]['validation_loss']
    best_update = 0
    best_model_state = cpu_tree(model.state_dict())
    best_optimizer_state = cpu_tree(optimizer.state_dict())

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, UPDATES + 1):
        indices = torch.randint(len(train_examples), (BATCH,), generator=batch_rng).tolist()
        batch = collate_instruction_batch(train_examples, indices, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss = response_loss(model, batch)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Nonfinite loss at {suffix} update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % EVAL_EVERY == 0:
            metrics = {'update': step, 'train_loss': evaluate(model, train_examples),
                       'validation_loss': evaluate(model, validation_examples)}
            history.append(metrics)
            if metrics['validation_loss'] < best_loss:
                best_loss = metrics['validation_loss']
                best_update = step
                best_model_state = cpu_tree(model.state_dict())
                best_optimizer_state = cpu_tree(optimizer.state_dict())
            print(f"{suffix} update={step} train={metrics['train_loss']:.6f} "
                  f"val={metrics['validation_loss']:.6f} best={best_loss:.6f}@{best_update}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_model_state)
    generations = generate_test(model, tokenizer)

    ckpt_dir = OUT / suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / 'best.pt'
    if checkpoint_path.exists():
        raise FileExistsError(f'Refusing to overwrite {checkpoint_path}')
    checkpoint = {
        'model_config': config.to_dict(), 'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state, 'tokenizer_type': 'bpe',
        'tokenizer_config': tokenizer.to_dict(), 'step': best_update,
        'base_checkpoint': str(BASE), 'train_path': str(DATA / 'train.jsonl'),
        'validation_path': str(DATA / 'validation.jsonl'),
        'training_config': {'lr': lr, 'batch_size': BATCH, 'updates': UPDATES,
                            'seed': SEED, 'dtype': 'float32', 'gradient_clipping': 1.0,
                            'objective': 'assistant-answer-only', 'context': config.block_size},
        'loss_history': history,
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        'lr': lr, 'best_update': best_update, 'best_validation_loss': best_loss,
        'history': history, 'training_seconds': elapsed,
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
        'checkpoint': str(checkpoint_path), 'generations': generations,
        'mean_3gram_repetition_rate': sum(x['3gram_repetition_rate'] for x in generations)/len(generations),
        'mean_4gram_repetition_rate': sum(x['4gram_repetition_rate'] for x in generations)/len(generations),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the 100-update comparison')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    base_state = torch.load(BASE, map_location='cpu', weights_only=True)
    tokenizer = BPETokenizer.from_dict(base_state['tokenizer_config'])
    if tokenizer.to_dict() != BPETokenizer.load(TOKENIZER).to_dict() or tokenizer.vocab_size != 1024:
        raise ValueError('The Base checkpoint tokenizer is not the established Byte-BPE 1024')
    config = ModelConfig(**base_state['model_config'])
    if (config.embedding_dim, config.num_heads, config.num_layers, config.block_size,
            config.attention_impl, config.vocab_size) != (640, 8, 6, 128, 'custom', 1024):
        raise ValueError('Unexpected 30M Base configuration')
    train_examples = load_instruction_examples(DATA/'train.jsonl', tokenizer, 128)
    validation_examples = load_instruction_examples(DATA/'validation.jsonl', tokenizer, 128)
    if len(train_examples) != 800 or len(validation_examples) != 100 or len(TEST_QUESTIONS) != 20:
        raise ValueError('Unexpected train/validation/test sizes')
    if any(len(e['token_ids']) > 128 for e in train_examples+validation_examples):
        raise ValueError('An instruction exceeds context 128')

    results = []
    for lr, suffix in ((0.0001, 'lr0001'), (0.00005, 'lr00005')):
        results.append(run_condition(lr, suffix, train_examples, validation_examples,
                                     base_state['model_state_dict'], tokenizer, config))
    LOG.mkdir(parents=True, exist_ok=True)
    metrics_path = LOG/'lr_comparison_100updates.json'
    metrics_path.write_text(json.dumps({
        'instruction_count': 1000, 'train_count': 800, 'validation_count': 100,
        'test_count': 100, 'fixed_test_questions': TEST_QUESTIONS,
        'test_question_generations': 20, 'evaluation': 'test excluded from training and checkpoint selection',
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        print(f"RESULT {result['lr']}: best={result['best_update']} val={result['best_validation_loss']:.6f}; "
              f"train100={result['history'][-1]['train_loss']:.6f}; "
              f"mean_rep3={result['mean_3gram_repetition_rate']:.4f}; "
              f"mean_rep4={result['mean_4gram_repetition_rate']:.4f}; "
              f"VRAM={result['peak_vram_allocated_mib']:.1f}MiB; "
              f"time={result['training_seconds']:.2f}s", flush=True)
    print(f'Metrics: {metrics_path}', flush=True)


if __name__ == '__main__':
    main()
