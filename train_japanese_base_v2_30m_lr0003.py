"""Train the fresh 30M Base v2 run with lr 0.0003 through update 3000."""

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
from train_japanese_base_v2_30m import BATCH_SIZE, DATA_DIR, GRADIENT_CLIPPING, SEED
from train_shakespeare import estimate_loss, tokenizer_state


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / 'checkpoints/japanese_base_v2/wide30m_lr0003_bpe1024'
LOG_DIR = ROOT / 'logs/japanese_base_v2/wide30m_lr0003_bpe1024'
TOTAL_UPDATES = 3000
LEARNING_RATE = 0.0003
EVAL_UPDATES = (500, 1000, 1500, 2000, 2500, 3000)
SAVE_UPDATES = (1000, 2000, 3000)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.glob('*.pt')):
        raise FileExistsError('Refusing to overwrite existing 30M lr 0.0003 checkpoints')

    tokenizer_path = DATA_DIR / 'bpe_1024.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != 1024 or tokenizer.vocab[:256] != [bytes([i]) for i in range(256)]:
        raise ValueError('Expected the existing Base v2 Byte-BPE 1024')
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 30_936_064:
        raise ValueError(f'Unexpected parameter count: {parameter_count}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    run_config = {
        'batch_size': BATCH_SIZE, 'gradient_accumulation': 1,
        'effective_tokens_per_update': BATCH_SIZE * config.block_size,
        'lr': LEARNING_RATE, 'seed': SEED, 'dtype': 'float32', 'amp': False, 'tf32': False,
        'gradient_clipping': GRADIENT_CLIPPING, 'optimizer': 'AdamW',
        'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
        'random_initialization': True, 'resume_checkpoint': None,
        'matched_initial_weights_and_batch_order': True,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    loss_history = []
    metrics = {
        'model_config': config.to_dict(), 'parameters': parameter_count,
        'run_config': run_config, 'updates': 0, 'loss_history': loss_history,
        'checkpoint_directory': str(CHECKPOINT_DIR),
        'tokenizer_path': str(tokenizer_path), 'dataset_directory': str(DATA_DIR),
    }
    log_path = LOG_DIR / f'run_0to3000_{run_id}.json'

    def save(path, step, best_value, best_update):
        tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
        state = {
            'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': tokenizer_type,
            'tokenizer_config': tokenizer_config, 'step': step,
            'best_val_loss': best_value, 'best_update': best_update,
            'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'batch_generator_state': generator.get_state(),
            'training_config': {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE},
            'run_config': run_config, 'dataset_directory': str(DATA_DIR),
            'tokenizer_path': str(tokenizer_path), 'loss_history': list(loss_history),
        }
        temporary = path.with_suffix('.pt.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    print(f'Fresh initialization; GPU={torch.cuda.get_device_name(0)}; '
          f'parameters={parameter_count:,}; lr={LEARNING_RATE}; FP32', flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_val_loss, best_update = float('inf'), 0
    model.train()
    for step in range(1, TOTAL_UPDATES + 1):
        x, y = get_batch(data['train'], config.block_size, BATCH_SIZE, 'cuda', generator)
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

        if step in EVAL_UPDATES:
            row = {
                'update': step,
                'train_loss': estimate_loss(model, data['train'], 'cuda'),
                'validation_loss': estimate_loss(model, data['validation'], 'cuda'),
            }
            loss_history.append(row)
            metrics['updates'] = step
            metrics['best_update'] = best_update
            metrics['best_validation_loss'] = best_val_loss
            torch.cuda.synchronize()
            metrics['elapsed_seconds'] = time.perf_counter() - started
            metrics['peak_vram_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024 ** 2
            metrics['peak_vram_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024 ** 2
            log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Update {step}: Train Loss={row["train_loss"]:.6f}, '
                  f'Val Loss={row["validation_loss"]:.6f}', flush=True)
            if row['validation_loss'] < best_val_loss:
                best_val_loss, best_update = row['validation_loss'], step
                save(CHECKPOINT_DIR / 'best.pt', step, best_val_loss, best_update)
            if step in SAVE_UPDATES:
                save(CHECKPOINT_DIR / f'update_{step}.pt', step, best_val_loss, best_update)

    torch.cuda.synchronize()
    metrics['elapsed_seconds'] = time.perf_counter() - started
    metrics['peak_vram_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    metrics['peak_vram_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024 ** 2
    metrics['best_update'] = best_update
    metrics['best_validation_loss'] = best_val_loss
    metrics['checkpoint'] = str(CHECKPOINT_DIR / f'update_{TOTAL_UPDATES}.pt')
    metrics['best_checkpoint'] = str(CHECKPOINT_DIR / 'best.pt')
    print(f'Elapsed={metrics["elapsed_seconds"]:.3f}s; '
          f'peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB; '
          f'best update={best_update}; best Val Loss={best_val_loss:.6f}', flush=True)

    del model, optimizer, logits, loss, x, y
    torch.cuda.empty_cache()
    best_model, best_tokenizer = load_model(CHECKPOINT_DIR / 'best.pt', 'cuda')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generated = generate(best_model, best_tokenizer, '人工知能の未来は',
                         max_new_tokens=500, temperature=0.8, top_k=40)
    metrics['generation'] = {
        'prompt': '人工知能の未来は', 'temperature': 0.8, 'top_k': 40,
        'max_new_tokens': 500, 'seed': SEED, 'utf8_safe': True,
        'checkpoint': str(CHECKPOINT_DIR / 'best.pt'), 'text': generated,
    }
    (LOG_DIR / f'generation_3000_seed42_{run_id}.txt').write_text(generated, encoding='utf-8')
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(generated, flush=True)


if __name__ == '__main__':
    main()
