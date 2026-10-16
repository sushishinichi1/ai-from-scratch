"""Train the wide 10M general-English Base model for a short GPU run."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import get_batch
from generate import generate, load_model
from model import ModelConfig, TinyGPT
from train_shakespeare import (BATCH_SIZE, LEARNING_RATE, estimate_loss,
                               load_checkpoint, save_checkpoint, validate_checkpoint)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets' / 'general_english'
CHECKPOINT_DIR = ROOT / 'checkpoints' / 'general_english' / 'wide10m_bpe512'
LOG_DIR = ROOT / 'logs' / 'general_english'
UPDATES = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--log-dir', type=Path, default=LOG_DIR)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int, default=UPDATES,
                        help='Total update target, including the resumed updates')
    parser.add_argument('--eval-updates', type=int, nargs='+',
                        help='Completed update counts at which to record train/validation loss')
    parser.add_argument('--checkpoint-interval', type=int, default=0)
    parser.add_argument('--save-best', action='store_true')
    parser.add_argument('--update-existing-best', action='store_true',
                        help='Update best.pt while retaining other existing checkpoints')
    parser.add_argument('--preserve-existing-checkpoints', action='store_true')
    parser.add_argument('--generate', action='store_true',
                        help='Generate 300 tokens with the technology prompt and seed 42 after training')
    parser.add_argument('--generate-from-best', action='store_true',
                        help='Load best.pt for generation after training')
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    log_dir = args.log_dir.resolve()
    if args.max_updates <= 0:
        parser.error('--max-updates must be positive')
    if args.checkpoint_interval < 0:
        parser.error('--checkpoint-interval must be nonnegative')
    if args.eval_updates is not None and any(step < 0 for step in args.eval_updates):
        parser.error('--eval-updates must be nonnegative')
    if (args.update_existing_best or args.generate_from_best) and not args.save_best:
        parser.error('--update-existing-best and --generate-from-best require --save-best')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this GPU run')
    if args.resume is None and any(checkpoint_dir.glob('*.pt')):
        raise FileExistsError(f'Checkpoints already exist: {checkpoint_dir}')
    tokenizer = BPETokenizer.load(data_dir / 'bpe_512.json')
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    splits = {}
    for name in ['train', 'validation']:
        binary_tokens = data_dir / f'{name}_tokens.npy'
        if binary_tokens.exists():
            import numpy as np
            ids = np.load(binary_tokens, allow_pickle=False)
            decoded = tokenizer.decode(ids.tolist())
        else:
            ids = json.loads((data_dir / f'{name}_tokens.json').read_text(encoding='utf-8'))
            decoded = tokenizer.decode(ids)
        text = (data_dir / f'{name}.txt').read_bytes().decode('utf-8')
        if decoded != text:
            raise ValueError(f'{name} token cache does not match corpus')
        splits[name] = torch.tensor(ids, dtype=torch.long)
    if tokenizer.vocab_size != 512:
        raise ValueError('This run requires a BPE 512 vocabulary')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=128,
                         embedding_dim=360, num_heads=8, num_layers=6,
                         attention_impl='custom')
    data_hash = {'dataset_manifest': manifest}
    checkpoint = load_checkpoint(args.resume) if args.resume is not None else None
    if checkpoint is not None:
        validate_checkpoint(checkpoint, config, tokenizer, data_hash)
    model = TinyGPT(config).to('cuda')
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 9_000_000 <= parameter_count <= 11_000_000:
        raise ValueError(f'Unexpected parameter count: {parameter_count}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(42)
    start_step, best_val_loss, best_update = 0, float('inf'), 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']
        best_update = checkpoint.get('best_update', start_step)
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        batch_state = checkpoint.get('batch_generator_state')
        # The initial 100-update run stored its batch RNG in the metrics log.
        if batch_state is None:
            for path in sorted(log_dir.glob('*.json')):
                metrics = json.loads(path.read_text(encoding='utf-8'))
                if (metrics.get('updates') == start_step
                        and metrics.get('model_config') == config.to_dict()
                        and metrics.get('seed') == 42
                        and metrics.get('batch_size') == BATCH_SIZE
                        and 'batch_generator_state' in metrics):
                    batch_state = torch.tensor(metrics['batch_generator_state'], dtype=torch.uint8)
                    break
        if batch_state is not None:
            generator.set_state(batch_state)
        else:
            # Recover the same sequence for legacy checkpoints with no RNG log.
            for _ in range(start_step):
                torch.randint(len(splits['train']) - config.block_size,
                              (BATCH_SIZE,), generator=generator)
        if start_step >= args.max_updates:
            print(f'Already at update {start_step}; no further training needed.', flush=True)
            return
        if (not args.preserve_existing_checkpoints
                and args.resume.resolve() == (checkpoint_dir / 'latest.pt').resolve()):
            archive = checkpoint_dir / f'update_{start_step}.pt'
            if not archive.exists():
                shutil.copy2(args.resume, archive)
        del checkpoint
    print(f'GPU: {torch.cuda.get_device_name(0)} | Parameters: {parameter_count:,}', flush=True)
    print(f'Updates {start_step} -> {args.max_updates}; batch={BATCH_SIZE}, '
          f'context=128, lr={optimizer.param_groups[0]["lr"]}', flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.time_ns()
    protected_paths = {path.resolve() for path in checkpoint_dir.glob('*.pt')}

    def output_checkpoint(name):
        path = checkpoint_dir / name
        if args.preserve_existing_checkpoints and path.resolve() in protected_paths:
            path = path.with_name(f'{path.stem}_{run_id}{path.suffix}')
        return path

    def store_checkpoint(path, step):
        save_checkpoint(path, model, optimizer, tokenizer, step, best_val_loss,
                        data_hash, batch_generator_state=generator.get_state(),
                        best_update=best_update)

    evaluation_steps = set(args.eval_updates or []) | {args.max_updates}
    loss_history = []
    saved_checkpoints = {}
    best_path = ((checkpoint_dir / 'best.pt' if args.update_existing_best
                  else output_checkpoint('best.pt')) if args.save_best else None)
    loss_log = log_dir / f'wide10m_{start_step}to{args.max_updates}_{run_id}.csv'

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with loss_log.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['update', 'train_loss', 'validation_loss'])
        writer.writeheader()

        def evaluate(step):
            nonlocal best_val_loss, best_update
            train_loss = estimate_loss(model, splits['train'], 'cuda')
            val_loss = estimate_loss(model, splits['validation'], 'cuda')
            record = {'update': step, 'train_loss': train_loss, 'validation_loss': val_loss}
            loss_history.append(record)
            writer.writerow(record)
            handle.flush()
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss, best_update = val_loss, step
            if best_path is not None and (improved or not best_path.exists()):
                store_checkpoint(best_path, step)
            print(f'Update {step}: Train Loss={train_loss:.6f}, '
                  f'Val Loss={val_loss:.6f}', flush=True)

        if start_step in evaluation_steps:
            evaluate(start_step)
        model.train()
        for step in range(start_step + 1, args.max_updates + 1):
            x, y = get_batch(splits['train'], config.block_size, BATCH_SIZE, 'cuda', generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f'Nonfinite loss at update {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in evaluation_steps:
                evaluate(step)
            elif step % 50 == 0:
                print(f'Completed update {step}/{args.max_updates}', flush=True)
            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                path = output_checkpoint(f'update_{step}.pt')
                store_checkpoint(path, step)
                saved_checkpoints[str(step)] = str(path)

        if args.preserve_existing_checkpoints:
            final_path = (Path(saved_checkpoints[str(args.max_updates)])
                          if str(args.max_updates) in saved_checkpoints
                          else output_checkpoint(f'update_{args.max_updates}.pt'))
        else:
            final_path = checkpoint_dir / 'latest.pt'
        if str(args.max_updates) not in saved_checkpoints or not args.preserve_existing_checkpoints:
            store_checkpoint(final_path, args.max_updates)
        saved_checkpoints[str(args.max_updates)] = str(final_path)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024 ** 2
    train_loss = loss_history[-1]['train_loss']
    val_loss = loss_history[-1]['validation_loss']
    metrics = {
        'updates': args.max_updates, 'resumed_from_step': start_step,
        'train_loss': train_loss, 'validation_loss': val_loss,
        'model_config': config.to_dict(), 'parameters': parameter_count,
        'seed': 42, 'batch_size': BATCH_SIZE,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'weight_decay': optimizer.param_groups[0]['weight_decay'],
        'loss_evaluation': f'Mean of 10 fixed-seed batches after update {args.max_updates}',
        'loss_history': loss_history, 'loss_log': str(loss_log),
        'elapsed_seconds': elapsed_seconds,
        'peak_vram_allocated_mib': peak_allocated_mib,
        'peak_vram_reserved_mib': peak_reserved_mib,
        'best_update': best_update, 'best_validation_loss': best_val_loss,
        'best_checkpoint': str(best_path) if best_path is not None else None,
        'saved_checkpoints': saved_checkpoints,
        'corpus_characters': manifest['corpus_characters'],
        'corpus_tokens': manifest['corpus_tokens'], 'vocab_size': tokenizer.vocab_size,
        'checkpoint': str(final_path),
        'batch_generator_state': generator.get_state().tolist(),
    }
    log_path = log_dir / f'wide10m_{args.max_updates}updates_{run_id}.json'
    log_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Elapsed={elapsed_seconds:.3f}s | Peak allocated={peak_allocated_mib:.3f} MiB | '
          f'Peak reserved={peak_reserved_mib:.3f} MiB | Best update={best_update} | '
          f'Best Val Loss={best_val_loss:.6f}', flush=True)
    if args.generate or args.generate_from_best:
        generation_checkpoint = best_path if args.generate_from_best else final_path
        generation_model, generation_tokenizer = (load_model(generation_checkpoint, 'cuda')
                                                  if args.generate_from_best
                                                  else (model, tokenizer))
        generation_model.eval()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        text = generate(generation_model, generation_tokenizer, 'The future of technology',
                        max_new_tokens=300, temperature=0.8, top_k=40)
        generation_path = log_dir / f'generation_{args.max_updates}_seed42_{run_id}.txt'
        generation_path.write_text(text, encoding='utf-8')
        metrics['generation'] = {
            'prompt': 'The future of technology', 'temperature': 0.8,
            'top_k': 40, 'max_new_tokens': 300, 'seed': 42,
            'checkpoint': str(generation_checkpoint), 'text': text, 'output': str(generation_path),
        }
        log_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(text, flush=True)


if __name__ == '__main__':
    main()
