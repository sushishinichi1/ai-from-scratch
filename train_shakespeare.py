import argparse
import csv
import hashlib
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import ROOT, get_batch, load_shakespeare
from model import ModelConfig, ModelConfig10M, TinyGPT
from tokenizer import CharacterTokenizer


CHECKPOINT_DIR = ROOT / 'checkpoints' / 'shakespeare'
LOG_DIR = ROOT / 'logs'
TOTAL_UPDATES = 5000
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EVAL_INTERVAL = 100
STANDARD_BPE_TOKENIZER = ROOT / 'bpe_512.json'


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def tokenizer_state(tokenizer):
    if isinstance(tokenizer, BPETokenizer):
        return 'bpe', tokenizer.to_dict()
    if isinstance(tokenizer, CharacterTokenizer):
        return 'char', {'chars': list(tokenizer.chars)}
    raise ValueError('Unsupported tokenizer type')


def save_checkpoint(path, model, optimizer, tokenizer, step, best_val_loss, data_hash,
                    *, batch_generator_state=None, best_update=None):
    tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
    checkpoint = {
        'model_config': model.config.to_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer_type': tokenizer_type,
        'tokenizer_config': tokenizer_config,
        'step': step,  # Number of completed optimizer updates.
        'best_val_loss': best_val_loss,
        'cpu_rng_state': torch.get_rng_state(),
        'cuda_rng_states': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        'training_config': {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE},
        'data_hash': data_hash,
    }
    if batch_generator_state is not None:
        checkpoint['batch_generator_state'] = batch_generator_state
    if best_update is not None:
        checkpoint['best_update'] = best_update
    if tokenizer_type == 'char':
        # Keep the legacy field for existing character-checkpoint consumers.
        checkpoint['chars'] = list(tokenizer.chars)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


@torch.no_grad()
def estimate_loss(model, data, device, batches=10):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(12345)
    try:
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, model.config.block_size, BATCH_SIZE, device, generator)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError('Evaluation loss is NaN or Inf')
            losses.append(value)
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train TinyGPT up to a total optimizer update count')
    parser.add_argument('--model-size', choices=['5m', '10m'], default='5m', help='Model configuration (default: 5m; 10m has 12 layers)')
    parser.add_argument('--attention-impl', choices=['torch', 'custom'], default='custom', help='Attention implementation (default: custom)')
    parser.add_argument('--tokenizer', choices=['char', 'bpe'], default='bpe', help='Tokenizer (default: bpe with bpe_512.json)')
    parser.add_argument('--bpe-tokenizer', type=Path, help='BPE tokenizer JSON path (default: bpe_512.json; requires --tokenizer bpe)')
    parser.add_argument('--resume', type=Path, help='Checkpoint to resume')
    parser.add_argument('--max-updates', type=int, help='Total update target, including completed updates when resuming (default: 5000 for 5m, 1000 for 10m)')
    parser.add_argument('--cpu-check', action='store_true', help='Check dataset encoding and one forward/backward pass on CPU without optimizer updates or checkpoint writes')
    args = parser.parse_args(argv)
    if args.max_updates is None:
        args.max_updates = 1000 if args.model_size == '10m' else TOTAL_UPDATES
    if args.max_updates <= 0:
        parser.error('--max-updates must be a positive integer')
    if args.bpe_tokenizer is not None and args.tokenizer != 'bpe':
        parser.error('--bpe-tokenizer requires --tokenizer bpe')
    if args.tokenizer == 'bpe' and args.bpe_tokenizer is None:
        args.bpe_tokenizer = STANDARD_BPE_TOKENIZER
    if args.model_size == '10m':
        if args.tokenizer != 'bpe' or args.attention_impl != 'custom':
            parser.error('10m requires --tokenizer bpe and --attention-impl custom')
        if args.bpe_tokenizer.resolve() != STANDARD_BPE_TOKENIZER.resolve():
            parser.error('10m uses the standard bpe_512.json tokenizer')
    return args


def output_paths(attention_impl, tokenizer_type='char', bpe_vocab_size=None, model_size='5m'):
    if attention_impl not in ('torch', 'custom'):
        raise ValueError('Unknown attention implementation')
    if tokenizer_type not in ('char', 'bpe'):
        raise ValueError('Unknown tokenizer type')
    if model_size not in ('5m', '10m'):
        raise ValueError('Unknown model size')
    name = f'{attention_impl}_{tokenizer_type}'
    if tokenizer_type == 'bpe' and bpe_vocab_size is not None:
        if type(bpe_vocab_size) is not int or bpe_vocab_size <= 0:
            raise ValueError('BPE vocabulary size must be a positive integer')
        if bpe_vocab_size != 256:
            # Keep existing 256 checkpoints in custom_bpe / torch_bpe.
            name += f'_{bpe_vocab_size}'
    if model_size == '10m':
        return CHECKPOINT_DIR / '10m' / name, LOG_DIR / '10m' / f'shakespeare_10m_{name}_{time.time_ns()}.csv'
    return CHECKPOINT_DIR / name, LOG_DIR / f'shakespeare_{name}_{time.time_ns()}.csv'


def validate_tokenizer_type(checkpoint, tokenizer_type):
    # Checkpoints created before tokenizer selection used characters only.
    saved_type = checkpoint.get('tokenizer_type', 'char')
    if saved_type != tokenizer_type:
        raise ValueError(
            f'Checkpoint tokenizer mismatch: checkpoint={saved_type!r}, requested={tokenizer_type!r}. '
            f'Use --tokenizer {saved_type} or select a matching checkpoint.'
        )


def validate_checkpoint(checkpoint, config, tokenizer, data_hash):
    tokenizer_type, tokenizer_config = tokenizer_state(tokenizer)
    validate_tokenizer_type(checkpoint, tokenizer_type)
    saved_tokenizer_config = checkpoint.get('tokenizer_config')
    if saved_tokenizer_config is None and tokenizer_type == 'char' and 'chars' in checkpoint:
        saved_tokenizer_config = {'chars': checkpoint['chars']}
    if saved_tokenizer_config != tokenizer_config:
        raise ValueError(
            f'Checkpoint {tokenizer_type} tokenizer configuration mismatch: '
            'vocabulary or BPE merge order differs from the current tokenizer.'
        )
    saved_config = ModelConfig(**checkpoint['model_config'])
    if saved_config.attention_impl != config.attention_impl:
        raise ValueError(
            f'Checkpoint attention_impl mismatch: checkpoint={saved_config.attention_impl!r}, '
            f'requested={config.attention_impl!r}. Use --attention-impl {saved_config.attention_impl} '
            'or select a matching checkpoint.'
        )
    if saved_config.to_dict() != config.to_dict():
        raise ValueError('Checkpoint model configuration does not match')
    if checkpoint['data_hash'] != data_hash:
        raise ValueError('Dataset differs from the checkpoint dataset')
    if checkpoint['training_config'] != {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE}:
        raise ValueError('Checkpoint training configuration does not match')


def check_on_cpu(model, tokenizer, train_data, val_data):
    """Exercise the full-size training batch without updating model weights."""
    config = model.config
    for name, data in [('train', train_data), ('validation', val_data)]:
        if data.dtype != torch.long or data.ndim != 1 or len(data) <= config.block_size:
            raise ValueError(f'Invalid {name} dataset shape or dtype')
        if data.min().item() < 0 or data.max().item() >= tokenizer.vocab_size:
            raise ValueError(f'{name} token IDs are outside the tokenizer vocabulary')
        print(f'{name}: tokens={len(data):,}, shape={tuple(data.shape)}, dtype={data.dtype}', flush=True)

    generator = torch.Generator().manual_seed(12345)
    model.train()
    model.zero_grad(set_to_none=True)
    x, y = get_batch(train_data, config.block_size, BATCH_SIZE, 'cpu', generator)
    logits = model(x)
    expected = (BATCH_SIZE, config.block_size, config.vocab_size)
    if tuple(logits.shape) != expected or y.shape != x.shape:
        raise AssertionError('Unexpected batch or logits shape')
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
    if not torch.isfinite(loss).item():
        raise FloatingPointError('CPU forward loss is NaN or Inf')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f'Missing or nonfinite gradient: {name}')
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), error_if_nonfinite=True)
    if grad_norm.item() == 0:
        raise AssertionError('CPU backward produced zero gradients')
    print(f'train batch: x={tuple(x.shape)}, y={tuple(y.shape)}, logits={tuple(logits.shape)}', flush=True)
    print(f'CPU forward/backward OK: loss={loss.item():.4f}, gradient norm={grad_norm.item():.4f}', flush=True)
    model.zero_grad(set_to_none=True)
    del logits, loss, x, y
    val_loss = estimate_loss(model, val_data, 'cpu', batches=1)
    print(f'CPU validation forward OK: loss={val_loss:.4f}', flush=True)
    print('No optimizer updates, checkpoint writes, or GPU training were performed.', flush=True)


def main():
    args = parse_args()
    if not args.cpu_check and not torch.cuda.is_available():
        print('CUDA is unavailable. Training was not started.')
        return

    torch.manual_seed(42)
    device = torch.device('cpu' if args.cpu_check else 'cuda')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        validate_tokenizer_type(checkpoint, args.tokenizer)
    print(f'Encoding train/validation with {args.tokenizer} tokenizer on CPU...', flush=True)
    if args.tokenizer == 'bpe':
        print(f'BPE tokenizer file: {args.bpe_tokenizer or ROOT / "bpe_tokenizer.json"}', flush=True)
    tokenizer, train_data, val_data = load_shakespeare(
        tokenizer_type=args.tokenizer, bpe_tokenizer_path=args.bpe_tokenizer)
    config_class = ModelConfig10M if args.model_size == '10m' else ModelConfig
    config = config_class(tokenizer.vocab_size, attention_impl=args.attention_impl)
    checkpoint_dir, log_path = output_paths(
        args.attention_impl, args.tokenizer, tokenizer.vocab_size if args.tokenizer == 'bpe' else None,
        model_size=args.model_size)
    data_hash = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('shakespeare_train.txt', 'shakespeare_val.txt')}
    if checkpoint is not None:
        validate_checkpoint(checkpoint, config, tokenizer, data_hash)

    if args.model_size == '10m' and not args.cpu_check and checkpoint is None and any(checkpoint_dir.glob('*.pt')):
        raise FileExistsError(f'10M checkpoints already exist in {checkpoint_dir}; use --resume to continue.')

    model = TinyGPT(config)
    parameter_count = sum(p.numel() for p in model.parameters())
    if args.model_size == '10m' and not 9_000_000 <= parameter_count <= 11_000_000:
        raise ValueError(f'10M model is outside the 9M-11M parameter range: {parameter_count:,}')
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if args.cpu_check or args.model_size == '10m':
        print(f'Tokenizer: {args.tokenizer} | Vocabulary: {tokenizer.vocab_size} | Attention: {args.attention_impl}', flush=True)
        print(f'Model size: {args.model_size} | Model: {config.to_dict()} | Parameters: {parameter_count:,}', flush=True)
        print('Checking the full training batch on CPU before any optimizer updates...', flush=True)
        rng_state = torch.get_rng_state()
        check_on_cpu(model, tokenizer, train_data, val_data)
        torch.set_rng_state(rng_state)
        if args.cpu_check:
            return

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    step, best_val_loss = 0, float('inf')
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        del checkpoint
    if step >= args.max_updates:
        print(f'Checkpoint already completed {step} updates (target: {args.max_updates}); nothing to train.')
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f'GPU: {torch.cuda.get_device_name()} | Model size: {args.model_size} | Parameters: {parameter_count:,}')
    print(f'Attention: {args.attention_impl} | Tokenizer: {args.tokenizer} | Checkpoints: {checkpoint_dir}')
    print(f'Model: {config.to_dict()} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Seed: 42 | Dtype: float32')
    print(f'Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}')
    print(f'Vocabulary: {tokenizer.vocab_size} | Updates: {step} -> {args.max_updates} | Log: {log_path}', flush=True)
    try:
        with log_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['step', 'train_loss', 'val_loss', 'elapsed_seconds', 'peak_allocated_mib', 'peak_reserved_mib'])
            for update in range(step + 1, args.max_updates + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                x, y = get_batch(train_data, config.block_size, BATCH_SIZE, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Loss is NaN or Inf before update {update}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                step = update
                del logits, loss, x, y
                if step % EVAL_INTERVAL == 0 or step == args.max_updates:
                    train_loss = estimate_loss(model, train_data, device)
                    val_loss = estimate_loss(model, val_data, device)
                    elapsed = time.perf_counter() - started
                    allocated = torch.cuda.max_memory_allocated() / 1024**2
                    reserved = torch.cuda.max_memory_reserved() / 1024**2
                    writer.writerow([step, train_loss, val_loss, elapsed, allocated, reserved])
                    handle.flush()
                    print(f'Update {step:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s | Peak GPU: {allocated:.1f} MiB allocated, {reserved:.1f} MiB reserved', flush=True)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(checkpoint_dir / 'best.pt', model, optimizer, tokenizer, step, best_val_loss, data_hash)
                if step % 500 == 0 or step == args.max_updates:
                    save_checkpoint(checkpoint_dir / 'latest.pt', model, optimizer, tokenizer, step, best_val_loss, data_hash)
                    print(f'Saved latest.pt at update {step}', flush=True)
    except KeyboardInterrupt:
        # An interrupt may happen partway through an optimizer update. Keep the
        # last atomically saved checkpoint rather than labeling partial weights.
        print(f'Interrupted. Resume from the last saved checkpoint (last completed update: {step}).')
    finally:
        print(f'Peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB; reserved {torch.cuda.max_memory_reserved() / 1024**2:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
