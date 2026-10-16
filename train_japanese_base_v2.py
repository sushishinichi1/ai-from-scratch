"""Train or resume Japanese Base v2 with the same FP32 configuration."""

import argparse
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
from train_shakespeare import estimate_loss, tokenizer_state


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets/japanese_base_v2'
CHECKPOINT_DIR = ROOT / 'checkpoints/japanese_base_v2/wide_bpe1024'
LOG_DIR = ROOT / 'logs/japanese_base_v2/wide_bpe1024'
UPDATES = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
SEED = 42
GRADIENT_CLIPPING = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int, default=UPDATES)
    parser.add_argument('--eval-updates', type=int, nargs='+')
    parser.add_argument('--save-updates', type=int, nargs='+')
    parser.add_argument('--generation-max-new-tokens', type=int, default=300)
    args = parser.parse_args()
    if args.max_updates <= 0:
        parser.error('--max-updates must be positive')
    if args.generation_max_new_tokens < 0:
        parser.error('--generation-max-new-tokens must be nonnegative')
    args.eval_updates = set(args.eval_updates or range(100, args.max_updates + 1, 100)) | {args.max_updates}
    args.save_updates = set(args.save_updates or []) | {args.max_updates}
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run')
    if args.resume is None and any(CHECKPOINT_DIR.glob('*.pt')):
        raise FileExistsError('Base v2 checkpoints already exist; refusing to overwrite or resume')
    if args.resume is not None and args.resume.resolve().parent != CHECKPOINT_DIR.resolve():
        raise ValueError('Resume only a checkpoint in the Base v2 checkpoint directory')
    for step in args.save_updates:
        if (CHECKPOINT_DIR / f'update_{step}.pt').exists():
            raise FileExistsError(f'Checkpoint update_{step}.pt already exists')

    manifest = json.loads((DATA_DIR / 'cleaning_summary.json').read_text(encoding='utf-8'))
    comparison = manifest['tokenizer_comparison']
    if not comparison['round_trip_exact'] or comparison['new_tokenizer_merge_training_split'] != 'train only':
        raise ValueError('Base v2 tokenizer preparation is incomplete')
    if manifest['document_overlap'] or manifest['group_overlap']:
        raise ValueError('The corpus has train/validation document overlap')
    tokenizer_path = DATA_DIR / 'bpe_1024.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != 1024 or tokenizer.vocab[:256] != [bytes([i]) for i in range(256)]:
        raise ValueError('Use the new Base v2 Byte-BPE 1024 with the 256-byte base vocabulary')
    data = {}
    for split in ('train', 'validation'):
        tokens = torch.tensor(np.load(DATA_DIR / f'{split}_tokens.npy', allow_pickle=False), dtype=torch.long)
        if (tokens.ndim != 1 or len(tokens) != comparison['new'][f'{split}_tokens']
                or tokens.min().item() < 0 or tokens.max().item() >= tokenizer.vocab_size):
            raise ValueError(f'Invalid new-tokenizer {split} cache')
        data[split] = tokens
        print(f'{split}: {len(tokens):,} tokens (new Base v2 tokenizer)', flush=True)

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=128, embedding_dim=360,
                         num_heads=8, num_layers=6, attention_impl='custom')
    model = TinyGPT(config).to(device='cuda', dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    run_config = {
        'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'seed': SEED, 'dtype': 'float32',
        'amp': False, 'tf32': False, 'gradient_clipping': GRADIENT_CLIPPING,
        'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
        'random_initialization': True, 'resume_checkpoint': None,
    }
    start_step, best_val_loss, best_update = 0, float('inf'), 0
    prior_history = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['model_config'] != config.to_dict():
            raise ValueError('Resume model configuration differs from Base v2')
        if checkpoint['tokenizer_type'] != 'bpe' or checkpoint['tokenizer_config'] != tokenizer.to_dict():
            raise ValueError('Resume tokenizer differs from the new Base v2 tokenizer')
        if checkpoint['data_hash']['dataset_manifest'] != manifest:
            raise ValueError('Resume corpus metadata differs')
        for key, value in run_config.items():
            if key not in {'random_initialization', 'resume_checkpoint'} and checkpoint['run_config'].get(key) != value:
                raise ValueError(f'Resume training setting differs: {key}')
        if any(tensor.is_floating_point() and tensor.dtype != torch.float32
               for tensor in checkpoint['model_state_dict'].values()):
            raise TypeError('Resume model weights must be FP32')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for group in optimizer.param_groups:
            if (group['lr'] != LEARNING_RATE or group['betas'] != (0.9, 0.999)
                    or group['eps'] != 1e-8 or group['weight_decay'] != 0.01):
                raise ValueError('Resume optimizer settings differ')
        start_step = checkpoint['step']
        if start_step >= args.max_updates:
            raise ValueError('Target update must be greater than the checkpoint update')
        best_val_loss, best_update = checkpoint['best_val_loss'], checkpoint['best_update']
        prior_history = list(checkpoint.get('loss_history', []))
        generator.set_state(checkpoint['batch_generator_state'])
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        run_config['random_initialization'] = False
        run_config['resume_checkpoint'] = str(args.resume.resolve())
        del checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    loss_history = []

    def save(path, step):
        tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
        checkpoint = {
            'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': tokenizer_type,
            'tokenizer_config': tokenizer_config, 'step': step,
            'best_val_loss': best_val_loss, 'best_update': best_update,
            'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
            'batch_generator_state': generator.get_state(),
            'training_config': {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE},
            'run_config': run_config, 'dataset_directory': str(DATA_DIR),
            'tokenizer_path': str(tokenizer_path), 'loss_history': prior_history + list(loss_history),
            'data_hash': {'dataset_manifest': manifest},
        }
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    print(f'Parameters: {parameter_count:,}; config={config.to_dict()}', flush=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}; updates {start_step} -> {args.max_updates}; '
          'FP32; clipping=1.0; optimizer and RNG states restored' if args.resume is not None
          else f'GPU: {torch.cuda.get_device_name(0)}; fresh initialization; FP32; clipping=1.0', flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (LOG_DIR / f'losses_{start_step}to{args.max_updates}_{run_id}.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['update', 'train_loss', 'validation_loss'])
        writer.writeheader()
        model.train()
        for step in range(start_step + 1, args.max_updates + 1):
            x, y = get_batch(data['train'], config.block_size, BATCH_SIZE, 'cuda', generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.dtype != torch.float32:
                raise TypeError('Training logits must stay in FP32')
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f'Nonfinite training loss at update {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
            optimizer.step()
            if step in args.eval_updates:
                record = {'update': step, 'train_loss': estimate_loss(model, data['train'], 'cuda'),
                          'validation_loss': estimate_loss(model, data['validation'], 'cuda')}
                loss_history.append(record)
                writer.writerow(record)
                stream.flush()
                if record['validation_loss'] < best_val_loss:
                    best_val_loss, best_update = record['validation_loss'], step
                    save(CHECKPOINT_DIR / 'best.pt', step)
                print(f'Update {step}: Train Loss={record["train_loss"]:.6f}, '
                      f'Val Loss={record["validation_loss"]:.6f}', flush=True)
            elif step % 50 == 0:
                print(f'Completed update {step}/{args.max_updates}', flush=True)
            if step in args.save_updates:
                save(CHECKPOINT_DIR / f'update_{step}.pt', step)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {
        'updates': args.max_updates, 'resumed_from_step': start_step,
        'parameters': parameter_count, 'model_config': config.to_dict(),
        'run_config': run_config, 'tokenizer_path': str(tokenizer_path),
        'train_characters': manifest['train_characters'],
        'validation_characters': manifest['validation_characters'],
        'train_tokens': len(data['train']), 'validation_tokens': len(data['validation']),
        'loss_history': loss_history, 'elapsed_seconds': elapsed,
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
        'best_update': best_update, 'best_validation_loss': best_val_loss,
        'checkpoint': str(CHECKPOINT_DIR / f'update_{args.max_updates}.pt'),
        'best_checkpoint': str(CHECKPOINT_DIR / 'best.pt'),
    }
    log_path = LOG_DIR / f'base_v2_{start_step}to{args.max_updates}_{run_id}.json'
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Elapsed={elapsed:.3f}s; peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB; best update={best_update}', flush=True)
    del model, optimizer, logits, loss, x, y
    torch.cuda.empty_cache()
    best_model, best_tokenizer = load_model(CHECKPOINT_DIR / 'best.pt', 'cuda')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generated = generate(best_model, best_tokenizer, '人工知能の未来は', temperature=0.8,
                         top_k=40, max_new_tokens=args.generation_max_new_tokens)
    metrics['generation'] = {
        'prompt': '人工知能の未来は', 'temperature': 0.8, 'top_k': 40,
        'max_new_tokens': args.generation_max_new_tokens, 'seed': SEED, 'utf8_safe': True,
        'checkpoint': str(CHECKPOINT_DIR / 'best.pt'), 'text': generated,
    }
    (LOG_DIR / f'generation_{args.max_updates}_seed42_{run_id}.txt').write_text(generated, encoding='utf-8')
    log_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(generated, flush=True)


if __name__ == '__main__':
    main()
