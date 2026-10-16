"""Run a fresh, 100-update FP32 trial of the wider Japanese Base v2 model."""

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import get_batch
from model import ModelConfig30M, TinyGPT
from train_shakespeare import estimate_loss, tokenizer_state


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets/japanese_base_v2'
CHECKPOINT_DIR = ROOT / 'checkpoints/japanese_base_v2/wide30m_bpe1024'
LOG_DIR = ROOT / 'logs/japanese_base_v2/wide30m_bpe1024'
UPDATES = 100
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 0.0005
SEED = 42
GRADIENT_CLIPPING = 1.0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this trial')
    checkpoint_path = CHECKPOINT_DIR / 'update_100.pt'
    if checkpoint_path.exists():
        raise FileExistsError(f'Refusing to overwrite {checkpoint_path}')

    tokenizer_path = DATA_DIR / 'bpe_1024.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != 1024 or tokenizer.vocab[:256] != [bytes([i]) for i in range(256)]:
        raise ValueError('Expected the existing Base v2 Byte-BPE 1024')
    train_tokens = torch.tensor(np.load(DATA_DIR / 'train_tokens.npy', allow_pickle=False), dtype=torch.long)
    if train_tokens.ndim != 1 or train_tokens.min().item() < 0 or train_tokens.max().item() >= tokenizer.vocab_size:
        raise ValueError('Invalid Base v2 train token cache')

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = ModelConfig30M(vocab_size=tokenizer.vocab_size, block_size=128)
    model = TinyGPT(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 28_000_000 <= parameter_count <= 32_000_000:
        raise ValueError('Parameter count is outside the requested range')
    print(f'Config: {config.to_dict()}; parameters={parameter_count:,}', flush=True)

    # Check one full-context sample without changing the initialized weights or training batch stream.
    cpu_generator = torch.Generator().manual_seed(SEED)
    x, y = get_batch(train_tokens, config.block_size, 1, 'cpu', cpu_generator)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
    if not math.isfinite(loss.item()):
        raise FloatingPointError('Nonfinite CPU forward loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    del x, y, logits, loss, cpu_generator
    print('CPU forward/backward: passed (one sample, context 128, no optimizer step)', flush=True)

    model.to(device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    run_config = {
        'batch_size': BATCH_SIZE, 'gradient_accumulation': GRADIENT_ACCUMULATION,
        'effective_tokens_per_update': BATCH_SIZE * GRADIENT_ACCUMULATION * config.block_size,
        'lr': LEARNING_RATE, 'seed': SEED, 'dtype': 'float32', 'amp': False, 'tf32': False,
        'gradient_clipping': GRADIENT_CLIPPING, 'optimizer': 'AdamW',
        'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
        'random_initialization': True, 'resume_checkpoint': None,
    }
    loss_history = []

    def record(step):
        value = estimate_loss(model, train_tokens, 'cuda', batches=10)
        loss_history.append({'update': step, 'train_loss': value})
        print(f'Update {step}: Train Loss={value:.6f}', flush=True)

    record(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    for step in range(1, UPDATES + 1):
        x, y = get_batch(train_tokens, config.block_size, BATCH_SIZE, 'cuda', generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.dtype != torch.float32:
            raise TypeError('Expected FP32 logits')
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Nonfinite loss at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
        optimizer.step()
        if step in (20, 50, 100):
            record(step)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {
        'model_config': config.to_dict(), 'parameters': parameter_count, 'run_config': run_config,
        'updates': UPDATES, 'cpu_forward_backward_passed': True,
        'dataset_directory': str(DATA_DIR), 'tokenizer_path': str(tokenizer_path),
        'evaluation': {'split': 'train', 'batches': 10, 'batch_size': 16, 'seed': 12345},
        'loss_history': loss_history, 'elapsed_seconds': elapsed,
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    log_path = LOG_DIR / f'trial_100_{run_id}.json'
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
    checkpoint = {
        'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': tokenizer_type,
        'tokenizer_config': tokenizer_config, 'step': UPDATES,
        'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
        'batch_generator_state': generator.get_state(),
        'training_config': {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE},
        'run_config': run_config, 'dataset_directory': str(DATA_DIR),
        'tokenizer_path': str(tokenizer_path), 'loss_history': loss_history,
    }
    temporary_path = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f'Elapsed={elapsed:.3f}s; peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB', flush=True)
    print(f'Metrics: {log_path}', flush=True)


if __name__ == '__main__':
    main()
