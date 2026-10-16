"""Fresh 300-update Base v3 comparison run; never resumes Base v2 weights."""

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
DATA_DIR = ROOT / 'datasets/japanese_base_v3'
TOKENIZER_PATH = ROOT / 'datasets/japanese_base_v2/bpe_1024.json'
CHECKPOINT_DIR = ROOT / 'checkpoints/japanese_base_v3/wide30m_bpe1024_lr0003'
LOG_DIR = ROOT / 'logs/japanese_base_v3/wide30m_bpe1024_lr0003'
UPDATES = 300
BATCH_SIZE = 16
SEED = 42
LR = 0.0003
CLIP = 1.0
EVAL_UPDATES = (100, 200, 300)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the requested short training run')
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {CHECKPOINT_DIR}')
    if LOG_DIR.exists() and any(LOG_DIR.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {LOG_DIR}')

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    if tokenizer.vocab_size != 1024 or tokenizer.vocab[:256] != [bytes([i]) for i in range(256)]:
        raise ValueError('Expected unchanged Japanese Base v2 Byte-BPE 1024')
    train = torch.tensor(np.load(DATA_DIR / 'train_tokens.npy', allow_pickle=False), dtype=torch.long)
    val = torch.tensor(np.load(DATA_DIR / 'validation_tokens.npy', allow_pickle=False), dtype=torch.long)
    for split, tokens in (('train', train), ('validation', val)):
        if tokens.ndim != 1 or tokens.numel() <= 128 or tokens.min().item() < 0 or tokens.max().item() >= 1024:
            raise ValueError(f'Invalid v3 {split} token cache')

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = ModelConfig30M(vocab_size=tokenizer.vocab_size, block_size=128)
    model = TinyGPT(config).to(device='cuda', dtype=torch.float32)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 30_936_064:
        raise ValueError(f'Unexpected parameter count {parameters:,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01)
    batch_rng = torch.Generator().manual_seed(SEED)
    config_record = {
        'model': config.to_dict(), 'parameters': parameters, 'batch_size': BATCH_SIZE,
        'effective_tokens_per_update': BATCH_SIZE * config.block_size,
        'lr': LR, 'seed': SEED, 'dtype': 'float32', 'tf32': False,
        'gradient_clipping': CLIP, 'optimizer': 'AdamW', 'betas': [0.9, 0.999],
        'eps': 1e-8, 'weight_decay': 0.01, 'random_initialization': True,
        'resumed_from': None, 'tokenizer': str(TOKENIZER_PATH), 'dataset': str(DATA_DIR),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    log_path = LOG_DIR / f'fresh_0to300_{run_id}.json'
    history = []
    metrics = {'config': config_record, 'loss_history': history,
               'evaluation_batches': 10, 'evaluation_batch_size': BATCH_SIZE,
               'evaluation_seed': 12345}

    def evaluate(step):
        row = {'update': step,
               'train_loss': estimate_loss(model, train, 'cuda', batches=10),
               'validation_loss': estimate_loss(model, val, 'cuda', batches=10)}
        history.append(row)
        print(f"Update {step}: Train Loss={row['train_loss']:.6f}, "
              f"Val Loss={row['validation_loss']:.6f}", flush=True)
        return row

    def save(path, step, best_update, best_loss):
        tok_type, tok_config = tokenizer_state(tokenizer)
        state = {
            'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': tok_type,
            'tokenizer_config': tok_config, 'step': step,
            'best_update': best_update, 'best_val_loss': best_loss,
            'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'batch_generator_state': batch_rng.get_state(), 'run_config': config_record,
            'loss_history': list(history), 'resume_checkpoint': None,
        }
        temporary = path.with_suffix('.pt.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    print(f"Fresh v3 initialization on {torch.cuda.get_device_name(0)}; "
          f"parameters={parameters:,}; FP32; lr={LR}", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_loss, best_update = float('inf'), 0
    model.train()
    for step in range(1, UPDATES + 1):
        x, y = get_batch(train, config.block_size, BATCH_SIZE, 'cuda', batch_rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Nonfinite training loss at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP, error_if_nonfinite=True)
        optimizer.step()
        if step in EVAL_UPDATES:
            row = evaluate(step)
            if row['validation_loss'] < best_loss:
                best_loss, best_update = row['validation_loss'], step
                save(CHECKPOINT_DIR / 'best.pt', step, best_update, best_loss)
            metrics.update({'best_update': best_update, 'best_validation_loss': best_loss,
                            'updates': step, 'elapsed_seconds': time.perf_counter() - started})
            torch.cuda.synchronize()
            metrics['peak_vram_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
            metrics['peak_vram_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024**2
            log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    save(CHECKPOINT_DIR / 'update_300.pt', UPDATES, best_update, best_loss)
    metrics.update({'updates': UPDATES, 'best_update': best_update,
                    'best_validation_loss': best_loss, 'elapsed_seconds': elapsed,
                    'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                    'checkpoint': str(CHECKPOINT_DIR / 'update_300.pt'),
                    'best_checkpoint': str(CHECKPOINT_DIR / 'best.pt')})
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Elapsed={elapsed:.2f}s; peak VRAM allocated="
          f"{metrics['peak_vram_allocated_mib']:.1f} MiB; best update={best_update}; "
          f"best Val={best_loss:.6f}", flush=True)


if __name__ == '__main__':
    main()
