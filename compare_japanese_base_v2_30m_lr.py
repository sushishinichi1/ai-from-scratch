"""Compare four learning-rate schedules from shared initial 30M weights."""

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
from train_japanese_base_v2_30m import BATCH_SIZE, DATA_DIR, GRADIENT_CLIPPING, SEED
from train_shakespeare import estimate_loss


UPDATES = 300
CASES = (
    ('A', 0.0003, 0),
    ('B', 0.0002, 0),
    ('C', 0.0001, 0),
    ('D', 0.0003, 100),
)
LOG_DIR = Path(__file__).resolve().parent / 'logs/japanese_base_v2/wide30m_lr_trials'


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    tokenizer = BPETokenizer.load(DATA_DIR / 'bpe_1024.json')
    if tokenizer.vocab_size != 1024:
        raise ValueError('Expected the existing Base v2 Byte-BPE 1024')
    data = {
        split: torch.tensor(np.load(DATA_DIR / f'{split}_tokens.npy', allow_pickle=False), dtype=torch.long)
        for split in ('train', 'validation')
    }
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    config = ModelConfig30M(vocab_size=tokenizer.vocab_size, block_size=128)
    initial_model = TinyGPT(config)
    initial_weights = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    del initial_model
    if parameter_count != 30_936_064:
        raise ValueError('Unexpected model parameter count')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'lr_comparison_{time.time_ns()}.json'
    report = {
        'model_config': config.to_dict(), 'parameters': parameter_count,
        'batch_size': BATCH_SIZE, 'seed': SEED, 'dtype': 'float32', 'amp': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01},
        'gradient_clipping': GRADIENT_CLIPPING, 'updates_per_case': UPDATES,
        'evaluation': {'batches': 10, 'batch_size': 16, 'seed': 12345},
        'dataset_directory': str(DATA_DIR), 'tokenizer_path': str(DATA_DIR / 'bpe_1024.json'),
        'shared_initial_weights': True, 'shared_batch_order_seed': SEED, 'cases': [],
    }

    def write_report():
        log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def run_case(label, peak_lr, warmup_updates):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = TinyGPT(config)
        model.load_state_dict(initial_weights)
        model.to(device='cuda', dtype=torch.float32).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        generator = torch.Generator().manual_seed(SEED)
        result = {'case': label, 'peak_lr': peak_lr, 'warmup_updates': warmup_updates, 'loss_history': []}
        report['cases'].append(result)
        print(f'{label}: lr={peak_lr}; warmup={warmup_updates}; fresh shared weights and batch order', flush=True)
        started = time.perf_counter()
        for step in range(1, UPDATES + 1):
            lr = peak_lr * min(step / warmup_updates, 1.0) if warmup_updates else peak_lr
            for group in optimizer.param_groups:
                group['lr'] = lr
            x, y = get_batch(data['train'], config.block_size, BATCH_SIZE, 'cuda', generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f'{label}: NaN/Inf loss at update {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
            optimizer.step()
            if step in (100, 200, 300):
                row = {'update': step, 'lr': lr,
                       'train_loss': estimate_loss(model, data['train'], 'cuda'),
                       'validation_loss': estimate_loss(model, data['validation'], 'cuda')}
                result['loss_history'].append(row)
                write_report()
                print(f'{label} update {step}: Train={row["train_loss"]:.6f}; '
                      f'Val={row["validation_loss"]:.6f}', flush=True)
        torch.cuda.synchronize()
        result['elapsed_seconds'] = time.perf_counter() - started
        result['nan_inf'] = False
        write_report()

    for label, peak_lr, warmup_updates in CASES:
        run_case(label, peak_lr, warmup_updates)
        torch.cuda.empty_cache()
    best = min(report['cases'], key=lambda case: case['loss_history'][-1]['validation_loss'])
    report['best_case_at_300'] = best['case']
    report['difference_from_10m_val_at_300'] = best['loss_history'][-1]['validation_loss'] - 4.6488
    write_report()
    print(f'Best case at update 300: {best["case"]}; '
          f'Val={best["loss_history"][-1]["validation_loss"]:.6f}', flush=True)
    print(f'Metrics: {log_path}', flush=True)


if __name__ == '__main__':
    main()
