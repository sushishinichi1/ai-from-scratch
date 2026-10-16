"""Continue the existing Japanese Base v2 30M run without changing its settings."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import get_batch
from generate import generate, load_model
from model import ModelConfig30M, TinyGPT
from train_japanese_base_v2_30m import (
    BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, GRADIENT_ACCUMULATION,
    GRADIENT_CLIPPING, LEARNING_RATE, LOG_DIR, SEED,
)
from train_shakespeare import estimate_loss, tokenizer_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('--eval-updates', type=int, nargs='+', required=True)
    parser.add_argument('--save-updates', type=int, nargs='+', default=[])
    parser.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--log-dir', type=Path, default=LOG_DIR)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    args.eval_updates = set(args.eval_updates) | {args.max_updates}
    args.save_updates = set(args.save_updates) | {args.max_updates}
    return args


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    log_dir = args.log_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.resume.resolve().parent != checkpoint_dir:
        raise ValueError('Resume only the dedicated 30M checkpoint directory')
    for step in args.save_updates:
        path = checkpoint_dir / f'update_{step}.pt'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')

    tokenizer_path = DATA_DIR / 'bpe_1024.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != 1024:
        raise ValueError('Expected Base v2 Byte-BPE 1024')
    data = {}
    for split in ('train', 'validation'):
        tokens = torch.tensor(np.load(DATA_DIR / f'{split}_tokens.npy', allow_pickle=False), dtype=torch.long)
        if tokens.ndim != 1 or tokens.min().item() < 0 or tokens.max().item() >= tokenizer.vocab_size:
            raise ValueError(f'Invalid {split} token cache')
        data[split] = tokens

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = ModelConfig30M(vocab_size=tokenizer.vocab_size, block_size=128)
    model = TinyGPT(config).to(device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    run_config = {
        'batch_size': BATCH_SIZE, 'gradient_accumulation': GRADIENT_ACCUMULATION,
        'effective_tokens_per_update': BATCH_SIZE * GRADIENT_ACCUMULATION * config.block_size,
        'lr': args.lr, 'seed': SEED, 'dtype': 'float32', 'amp': False, 'tf32': False,
        'gradient_clipping': GRADIENT_CLIPPING, 'optimizer': 'AdamW',
        'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
        'random_initialization': False, 'resume_checkpoint': str(args.resume.resolve()),
    }
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    if checkpoint['model_config'] != config.to_dict():
        raise ValueError('Resume model configuration differs')
    if checkpoint['tokenizer_type'] != 'bpe' or checkpoint['tokenizer_config'] != tokenizer.to_dict():
        raise ValueError('Resume tokenizer differs from Base v2')
    if Path(checkpoint['dataset_directory']).resolve() != DATA_DIR.resolve():
        raise ValueError('Resume corpus directory differs')
    for key, value in run_config.items():
        if key not in {'random_initialization', 'resume_checkpoint'} and checkpoint['run_config'].get(key) != value:
            raise ValueError(f'Resume setting differs: {key}')
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32
           for tensor in checkpoint['model_state_dict'].values()):
        raise TypeError('Resume weights must be FP32')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for group in optimizer.param_groups:
        if (group['lr'] != args.lr or group['betas'] != (0.9, 0.999)
                or group['eps'] != 1e-8 or group['weight_decay'] != 0.01):
            raise ValueError('Resume optimizer settings differ')
    start_step = checkpoint['step']
    if not start_step < args.max_updates:
        raise ValueError('Target must be greater than the checkpoint update')
    history = list(checkpoint['loss_history'])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_update = checkpoint.get('best_update', 0)
    generator.set_state(checkpoint['batch_generator_state'])
    torch.set_rng_state(checkpoint['cpu_rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
    del checkpoint
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_id = time.time_ns()
    records = []
    consecutive_val_increases = 0
    validation_history = [row['validation_loss'] for row in history if 'validation_loss' in row]
    for previous, current in zip(validation_history, validation_history[1:]):
        consecutive_val_increases = consecutive_val_increases + 1 if current > previous else 0
    previous_val = validation_history[-1] if validation_history else None

    def save(path, step):
        tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
        state = {
            'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': tokenizer_type,
            'tokenizer_config': tokenizer_config, 'step': step,
            'best_val_loss': best_val_loss, 'best_update': best_update,
            'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'batch_generator_state': generator.get_state(),
            'training_config': {'batch_size': BATCH_SIZE, 'lr': args.lr},
            'run_config': run_config, 'dataset_directory': str(DATA_DIR),
            'tokenizer_path': str(tokenizer_path), 'loss_history': list(history),
        }
        temporary = path.with_suffix('.pt.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def record(step):
        nonlocal best_val_loss, best_update, previous_val, consecutive_val_increases
        row = {'update': step, 'train_loss': estimate_loss(model, data['train'], 'cuda'),
               'validation_loss': estimate_loss(model, data['validation'], 'cuda')}
        history[:] = [item for item in history if item['update'] != step]
        history.append(row)
        records.append(row)
        value = row['validation_loss']
        consecutive_val_increases = consecutive_val_increases + 1 if previous_val is not None and value > previous_val else 0
        previous_val = value
        if value < best_val_loss:
            best_val_loss, best_update = value, step
            save(checkpoint_dir / 'best.pt', step)
        print(f'Update {step}: Train Loss={row["train_loss"]:.6f}, Val Loss={value:.6f}; '
              f'best update={best_update}', flush=True)
        if consecutive_val_increases >= 2 and value > best_val_loss + 0.5:
            return 'Validation Loss rose by more than 0.5 from its best value at consecutive evaluations'
        return None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}; parameters={parameter_count:,}; '
          f'updates {start_step} -> {args.max_updates}; FP32; optimizer and RNG states restored', flush=True)
    if not validation_history:
        record(start_step)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stopped_reason = None
    completed_step = start_step
    model.train()
    try:
        for step in range(start_step + 1, args.max_updates + 1):
            x, y = get_batch(data['train'], config.block_size, BATCH_SIZE, 'cuda', generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.dtype != torch.float32:
                raise TypeError('Expected FP32 logits')
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f'Nonfinite training loss at update {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
            optimizer.step()
            completed_step = step
            if step in args.eval_updates:
                stopped_reason = record(step)
            elif step % 100 == 0:
                print(f'Completed update {step}/{args.max_updates}', flush=True)
            if step in args.save_updates:
                save(checkpoint_dir / f'update_{step}.pt', step)
            if stopped_reason:
                save(checkpoint_dir / f'stopped_update_{step}.pt', step)
                break
    except (FloatingPointError, RuntimeError) as exc:
        stopped_reason = f'{type(exc).__name__}: {exc}'
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_checkpoint = checkpoint_dir / (
        f'stopped_update_{completed_step}.pt' if stopped_reason else f'update_{completed_step}.pt'
    )
    if not final_checkpoint.exists():
        final_checkpoint = checkpoint_dir / 'best.pt'
    metrics = {
        'model_config': config.to_dict(), 'parameters': parameter_count, 'run_config': run_config,
        'resumed_from_step': start_step, 'updates': completed_step, 'target_updates': args.max_updates,
        'loss_history': records, 'cumulative_loss_history': history, 'elapsed_seconds': elapsed,
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
        'best_update': best_update, 'best_validation_loss': best_val_loss,
        'stopped_reason': stopped_reason,
        'checkpoint': str(final_checkpoint),
        'best_checkpoint': str(checkpoint_dir / 'best.pt'),
    }
    log_path = log_dir / f'continued_{start_step}to{args.max_updates}_{run_id}.json'
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Elapsed={elapsed:.3f}s; peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB; best update={best_update}; '
          f'best Val Loss={best_val_loss:.6f}', flush=True)
    if stopped_reason:
        print(f'STOPPED: {stopped_reason}; metrics={log_path}', flush=True)
        return
    if args.generate:
        del model, optimizer, logits, loss, x, y
        torch.cuda.empty_cache()
        best_model, best_tokenizer = load_model(checkpoint_dir / 'best.pt', 'cuda')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        generated = generate(best_model, best_tokenizer, '人工知能の未来は',
                             max_new_tokens=500, temperature=0.8, top_k=40)
        metrics['generation'] = {
            'prompt': '人工知能の未来は', 'temperature': 0.8, 'top_k': 40,
            'max_new_tokens': 500, 'seed': SEED, 'utf8_safe': True, 'text': generated,
            'checkpoint': str(checkpoint_dir / 'best.pt'),
        }
        (log_dir / f'generation_{completed_step}_seed42_{run_id}.txt').write_text(generated, encoding='utf-8')
        log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        print(generated, flush=True)
    print(f'Metrics: {log_path}', flush=True)


if __name__ == '__main__':
    main()
