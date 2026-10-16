"""Fresh Japanese Base run: one CPU backward check, then 300 GPU updates."""

import csv
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
from model import ModelConfig, TinyGPT
from train_shakespeare import estimate_loss, save_checkpoint


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets' / 'japanese_wikipedia'
CHECKPOINT_DIR = ROOT / 'checkpoints' / 'japanese_wikipedia' / 'wide_bpe4096'
LOG_DIR = ROOT / 'logs' / 'japanese_wikipedia'
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
UPDATES = 300


def cpu_backward_once(model, train_data):
    rng_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    try:
        x, y = get_batch(train_data, 128, BATCH_SIZE, 'cpu',
                         torch.Generator().manual_seed(12345))
        logits = model(x)
        if tuple(logits.shape) != (BATCH_SIZE, 128, model.config.vocab_size):
            raise AssertionError('Unexpected CPU forward shape')
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
        if not torch.isfinite(loss).item():
            raise FloatingPointError('CPU loss is NaN/Inf')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f'CPU gradient missing or NaN/Inf: {name}')
        print('One CPU forward/backward passed (no optimizer update).', flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_state)
        torch.set_num_threads(threads)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if any(CHECKPOINT_DIR.glob('*.pt')):
        raise FileExistsError('Japanese checkpoints already exist; fresh run will not overwrite them')
    tokenizer = BPETokenizer.load(DATA_DIR / 'bpe_4096.json')
    manifest = json.loads((DATA_DIR / 'manifest.json').read_text(encoding='utf-8'))
    if (tokenizer.vocab_size != 4096 or not manifest['encode_decode_exact_match']
            or manifest['article_overlap'] != 0):
        raise ValueError('Prepared Japanese corpus/tokenizer requirements are not satisfied')
    data = {name: torch.tensor(np.load(DATA_DIR / f'{name}_tokens.npy', allow_pickle=False),
                               dtype=torch.long) for name in ['train', 'validation']}
    for name, tokens in data.items():
        if tokens.ndim != 1 or len(tokens) <= 128 or tokens.min() < 0 or tokens.max() >= tokenizer.vocab_size:
            raise ValueError(f'Invalid {name} token cache')
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, embedding_dim=360,
                         num_heads=8, num_layers=6, attention_impl='custom', block_size=128)
    model = TinyGPT(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'Model parameters: {parameter_count:,} | Configuration: {config.to_dict()}', flush=True)
    cpu_backward_once(model, data['train'])
    model = model.to('cuda')
    torch.cuda.manual_seed_all(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    loss_history = []
    best_val_loss, best_update = float('inf'), 0
    data_state = {'dataset_manifest': manifest}

    def save(path, step):
        save_checkpoint(path, model, optimizer, tokenizer, step, best_val_loss, data_state,
                        batch_generator_state=generator.get_state(), best_update=best_update)

    print(f'GPU: {torch.cuda.get_device_name(0)}; fresh random initialization, seed=42', flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (LOG_DIR / f'losses_300updates_{run_id}.csv').open('x', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['update', 'train_loss', 'validation_loss'])
        writer.writeheader()
        model.train()
        for step in range(1, UPDATES + 1):
            x, y = get_batch(data['train'], 128, BATCH_SIZE, 'cuda', generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f'GPU loss is NaN/Inf at update {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 100 == 0:
                train_loss = estimate_loss(model, data['train'], 'cuda')
                val_loss = estimate_loss(model, data['validation'], 'cuda')
                record = {'update': step, 'train_loss': train_loss, 'validation_loss': val_loss}
                loss_history.append(record)
                writer.writerow(record)
                output.flush()
                if val_loss < best_val_loss:
                    best_val_loss, best_update = val_loss, step
                    save(CHECKPOINT_DIR / 'best.pt', step)
                print(f'Update {step}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}', flush=True)
            elif step % 50 == 0:
                print(f'Completed update {step}/300', flush=True)
    save(CHECKPOINT_DIR / 'update_300.pt', UPDATES)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {
        'updates': UPDATES, 'seed': 42, 'random_initialization': True,
        'model_config': config.to_dict(), 'parameters': parameter_count,
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'weight_decay': optimizer.param_groups[0]['weight_decay'],
        'train_characters': manifest['train_characters'],
        'validation_characters': manifest['validation_characters'],
        'train_tokens': manifest['train_tokens'], 'vocab_size': tokenizer.vocab_size,
        'loss_history': loss_history, 'elapsed_seconds': elapsed,
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
        'best_update': best_update, 'best_validation_loss': best_val_loss,
        'checkpoint': str(CHECKPOINT_DIR / 'update_300.pt'),
        'best_checkpoint': str(CHECKPOINT_DIR / 'best.pt'),
        'cpu_forward_backward_count': 1,
    }
    log = LOG_DIR / f'japanese_300updates_{run_id}.json'
    log.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Training elapsed={elapsed:.3f}s; peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB; best update={best_update}', flush=True)
    best_model, best_tokenizer = load_model(CHECKPOINT_DIR / 'best.pt', 'cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    generated = generate(best_model, best_tokenizer, '人工知能の未来は',
                         temperature=0.8, top_k=40, max_new_tokens=300)
    (LOG_DIR / f'generation_300_seed42_{run_id}.txt').write_text(generated, encoding='utf-8')
    metrics['generation'] = {'prompt': '人工知能の未来は', 'temperature': 0.8,
                              'top_k': 40, 'max_new_tokens': 300, 'seed': 42,
                              'checkpoint': str(CHECKPOINT_DIR / 'best.pt'), 'text': generated}
    log.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(generated, flush=True)


if __name__ == '__main__':
    main()
