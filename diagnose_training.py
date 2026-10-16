"""Controlled 5M/10M diagnostics: four fresh 100-update runs, no checkpoint writes."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from data import ROOT, get_batch, load_shakespeare
from model import ModelConfig, ModelConfig10M, TinyGPT
from train_shakespeare import BATCH_SIZE, check_on_cpu, parse_args as training_args


SEED = 42
UPDATES = 100
CASES = [('5m_lr_0.0005', '5m', 0.0005), ('10m_lr_0.0005', '10m', 0.0005),
         ('10m_lr_0.0003', '10m', 0.0003), ('10m_lr_0.0001', '10m', 0.0001)]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_hashes():
    paths = sorted(set((ROOT / 'checkpoints').rglob('*.pt')) | set(ROOT.glob('*.pt')))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def optimizer_settings(optimizer):
    return [{key: value for key, value in group.items() if key != 'params'}
            for group in optimizer.param_groups]


def audit_saved_training():
    result = {}
    for size, path in [('5m', ROOT / 'checkpoints/shakespeare/custom_bpe_512/latest.pt'),
                       ('10m', ROOT / 'checkpoints/shakespeare/10m/custom_bpe_512/latest.pt')]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        groups = [{key: value for key, value in group.items() if key != 'params'}
                  for group in checkpoint['optimizer_state_dict']['param_groups']]
        steps = [float(state['step']) for state in checkpoint['optimizer_state_dict']['state'].values()]
        args = training_args(['--model-size', size, '--max-updates', '1000'])
        result[size] = {'checkpoint_read_for_audit_only': str(path.relative_to(ROOT)),
                        'model_config': checkpoint['model_config'],
                        'optimizer': groups, 'step': checkpoint['step'],
                        'optimizer_step_range': [min(steps), max(steps)],
                        'data_hash': checkpoint['data_hash'],
                        'tokenizer_hash': hashlib.sha256(json.dumps(checkpoint['tokenizer_config'], sort_keys=True).encode()).hexdigest(),
                        'default_resume_argument': args.resume}
    assert result['5m']['optimizer'] == result['10m']['optimizer']
    assert result['5m']['data_hash'] == result['10m']['data_hash']
    assert result['5m']['tokenizer_hash'] == result['10m']['tokenizer_hash']
    assert all(case['default_resume_argument'] is None for case in result.values())
    return result


@torch.no_grad()
def activation_stats(output):
    value = output.detach().float()
    flat = value.reshape(-1, value.size(-1))
    center = flat.mean(dim=0)
    numbers = torch.stack((value.mean(), value.std(unbiased=False),
                           flat.std(dim=0, unbiased=False).mean(),
                           F.cosine_similarity(flat, center.unsqueeze(0), dim=-1).mean(),
                           value.abs().max()))
    mean, std, across_token_std, cosine, max_abs = numbers.cpu().tolist()
    return {'mean': mean, 'std': std, 'across_token_std': across_token_std,
            'cosine_to_mean': cosine, 'max_abs': max_abs,
            'finite': bool(torch.isfinite(value).all().item()), 'shape': list(value.shape)}


class ActivationRecorder:
    def __init__(self, model):
        self.enabled = False
        self.values = {}
        self.handles = []
        modules = {'token_embedding': model.token_embedding,
                   'position_embedding': model.position_embedding,
                   **{f'block_{i + 1:02d}': block for i, block in enumerate(model.blocks)},
                   'final_layer_norm': model.norm, 'lm_head': model.output}
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))
        self.handles.append(model.blocks[0].register_forward_pre_hook(self._embedding_hook))

    def _hook(self, name):
        def record(module, inputs, output):
            if self.enabled:
                self.values[name] = activation_stats(output)
        return record

    def _embedding_hook(self, module, inputs):
        if self.enabled:
            self.values['embedding_sum'] = activation_stats(inputs[0])

    def begin(self):
        self.values = {}
        self.enabled = True

    def end(self):
        self.enabled = False
        assert all(value['finite'] for value in self.values.values()), 'Nonfinite activation'
        return self.values

    def close(self):
        for handle in self.handles:
            handle.remove()


def group_name(name):
    if name.startswith('blocks.'):
        return f'block_{int(name.split(".")[1]) + 1:02d}'
    if name.startswith('norm.'):
        return 'final_layer_norm'
    if name.startswith('output.'):
        return 'lm_head'
    return name.split('.')[0]


@torch.no_grad()
def gradient_stats(model):
    groups = {}
    parameters = {}
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = parameter.grad.detach().float()
        finite = bool(torch.isfinite(grad).all().item())
        numbers = torch.stack((grad.square().sum(), grad.abs().max(), (grad == 0).float().sum())).cpu().tolist()
        squares, max_abs, zeros = numbers
        norm = math.sqrt(squares)
        parameters[name] = {'norm': norm, 'rms': norm / math.sqrt(grad.numel()),
                            'max_abs': max_abs, 'zero_fraction': zeros / grad.numel(), 'finite': finite}
        group = groups.setdefault(group_name(name), {'squared_norm': 0.0, 'elements': 0,
                                                     'zeros': 0, 'max_abs': 0.0, 'finite': True,
                                                     'zero_parameters': [], 'near_zero_parameters': []})
        group['squared_norm'] += squares
        group['elements'] += grad.numel()
        group['zeros'] += zeros
        group['max_abs'] = max(group['max_abs'], max_abs)
        group['finite'] &= finite
        if norm == 0:
            group['zero_parameters'].append(name)
        elif parameters[name]['rms'] < 1e-12:
            group['near_zero_parameters'].append(name)
    for group in groups.values():
        group['norm'] = math.sqrt(group.pop('squared_norm'))
        group['rms'] = group['norm'] / math.sqrt(group['elements'])
        group['zero_fraction'] = group.pop('zeros') / group['elements']
    assert not missing, f'Missing gradients: {missing}'
    assert all(group['finite'] for group in groups.values()), 'Nonfinite gradient'
    return {'groups': groups, 'parameters': parameters, 'missing': missing}


def audit_causal_attention(model, device):
    model.eval()
    generator = torch.Generator().manual_seed(12345)
    dim = model.config.embedding_dim
    length = min(17, model.config.block_size)
    input_value = torch.randn(2, length, dim, generator=generator).to(device)
    layer_results = []
    for i, block in enumerate(model.blocks):
        attention = block.attention
        expected_mask = torch.ones(model.config.block_size, model.config.block_size,
                                   dtype=torch.bool, device=device).triu(1)
        assert torch.equal(attention.causal_mask, expected_mask)
        assert torch.equal(block.causal_mask, expected_mask)
        x = input_value.detach().clone().requires_grad_(True)
        output = attention(x)
        q = attention.query(x).view(2, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = attention.key(x).view(2, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        v = attention.value(x).view(2, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        reference = attention.output_projection(reference.transpose(1, 2).contiguous().view(2, length, dim))
        error = float((output - reference).abs().max().item())
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-4)
        cutoff = length // 2
        future_gradient = torch.autograd.grad(output[:, :cutoff].square().mean(), x)[0][:, cutoff:]
        future_grad_max = float(future_gradient.abs().max().item())
        assert future_grad_max == 0.0
        layer_results.append({'layer': i + 1, 'mask_matches_upper_triangle': True,
                              'sdpa_max_abs_error': error, 'future_input_gradient_max': future_grad_max})
    prefix_results = []
    with torch.no_grad():
        for sequence_length in sorted(set([1, min(17, model.config.block_size),
                                           min(64, model.config.block_size), model.config.block_size])):
            tokens = torch.randint(model.config.vocab_size, (2, sequence_length), generator=generator).to(device)
            cutoff = max(1, sequence_length // 2)
            changed = tokens.clone()
            changed[:, cutoff:] = (changed[:, cutoff:] + 1) % model.config.vocab_size
            error = float((model(tokens)[:, :cutoff] - model(changed)[:, :cutoff]).abs().max().item())
            assert error <= 1e-6, f'Future leakage at length {sequence_length}: {error}'
            prefix_results.append({'length': sequence_length, 'prefix_length': cutoff, 'max_abs_error': error})
    model.zero_grad(set_to_none=True)
    model.train()
    return {'layers': layer_results, 'full_model_prefix_tests': prefix_results, 'passed': True}


@torch.no_grad()
def probe_loss(model, x, y, recorder, collect=False):
    if collect:
        recorder.begin()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1)).item()
    assert math.isfinite(loss)
    activations = recorder.end() if collect else None
    return loss, activations


def run_case(label, size, lr, initial_state, expected_hash, batches, probe, output_dir, original_optimizer):
    torch.manual_seed(SEED)
    config_type = ModelConfig10M if size == '10m' else ModelConfig
    vocab_size = initial_state['token_embedding.weight'].size(0)
    model = TinyGPT(config_type(vocab_size, attention_impl='custom'))
    model.load_state_dict(initial_state)
    assert state_hash(model.state_dict()) == expected_hash
    model = model.to('cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    assert len(optimizer.state) == 0
    settings = optimizer_settings(optimizer)
    expected_settings = [{**group, 'lr': lr} for group in original_optimizer]
    assert settings == expected_settings
    recorder = ActivationRecorder(model)
    probe_x, probe_y = (tensor.to('cuda') for tensor in probe)
    loss0, activation0 = probe_loss(model, probe_x, probe_y, recorder, collect=True)
    rows = [{'update': 0, 'batch_loss_before_update': None, 'fixed_train_probe_loss_after_update': loss0,
             'grad_norm_before_clip': None, 'grad_norm_after_clip': None, 'finite': True}]
    details = [{'update': 0, 'fixed_probe_activations': activation0}]
    print(f'{label}: update 0 fixed train probe loss={loss0:.6f}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for update, (cpu_x, cpu_y) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        x, y = cpu_x.to('cuda'), cpu_y.to('cuda')
        collect = update <= 20 or update in (25, 50, 75, 100)
        if collect:
            recorder.begin()
        logits = model(x)
        batch_activations = recorder.end() if collect else None
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
        assert torch.isfinite(loss).item()
        batch_loss = loss.item()
        loss.backward()
        grad_details = gradient_stats(model) if collect else None
        before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
        assert math.isfinite(before)
        after = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True).item()
        assert after <= 1.00001
        optimizer.step()
        del logits, loss, x, y
        assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        fixed_loss, fixed_activations = probe_loss(model, probe_x, probe_y, recorder, collect=collect)
        rows.append({'update': update, 'batch_loss_before_update': batch_loss,
                     'fixed_train_probe_loss_after_update': fixed_loss,
                     'grad_norm_before_clip': before, 'grad_norm_after_clip': after, 'finite': True})
        if collect:
            details.append({'update': update, 'batch_activations_before_update': batch_activations,
                            'gradients_before_clip': grad_details, 'fixed_probe_activations': fixed_activations})
        if update <= 20 or update % 20 == 0:
            print(f'{label}: update {update:3d} batch loss={batch_loss:.6f}, fixed probe={fixed_loss:.6f}, grad={before:.6f}', flush=True)
    assert len(rows) == 101
    steps = [float(state['step']) for state in optimizer.state.values()]
    assert min(steps) == max(steps) == UPDATES
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {'case': label, 'model_size': size, 'lr': lr, 'seed': SEED,
              'model_config': model.config.to_dict(), 'initial_state_sha256': expected_hash,
              'fresh_optimizer': True, 'resumed': False, 'optimizer': settings,
              'optimizer_step_range_after_run': [min(steps), max(steps)],
              'updates': UPDATES, 'elapsed_seconds_with_diagnostics': elapsed,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
              'curve': rows, 'details': details}
    with (output_dir / f'{label}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f'{label}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{label}: completed exactly 100 updates in {elapsed:.1f}s; no checkpoint saved', flush=True)
    recorder.close()
    del model, optimizer, recorder
    torch.cuda.empty_cache()
    return report


def self_test():
    repeated = torch.tensor([[[1., -1.], [1., -1.]]])
    stats = activation_stats(repeated)
    assert stats['mean'] == 0 and stats['std'] == 1 and stats['across_token_std'] == 0
    config = ModelConfig(16, block_size=8, embedding_dim=8, num_heads=2, num_layers=2, attention_impl='custom')
    torch.manual_seed(SEED)
    model = TinyGPT(config)
    recorder = ActivationRecorder(model)
    recorder.begin()
    x = torch.randint(16, (2, 8))
    logits = model(x)
    recorded = recorder.end()
    assert {'token_embedding', 'embedding_sum', 'block_01', 'block_02', 'final_layer_norm', 'lm_head'} <= set(recorded)
    F.cross_entropy(logits.reshape(-1, 16), x.reshape(-1)).backward()
    gradients = gradient_stats(model)
    expected_norm = math.sqrt(sum(group['norm']**2 for group in gradients['groups'].values()))
    measured_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
    assert math.isclose(expected_norm, measured_norm, rel_tol=1e-5)
    recorder.close()
    audit_causal_attention(model, 'cpu')
    model.blocks[0].attention.causal_mask.zero_()
    try:
        audit_causal_attention(model, 'cpu')
    except AssertionError:
        pass
    else:
        raise AssertionError('Causal audit failed to detect a deliberately broken mask')
    print('Diagnostic self-test passed: activation collapse metric, gradient norms, hooks, and causal-mask detection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the four requested 100-update experiments')
    started = time.perf_counter()
    output_dir = args.output_dir or ROOT / 'logs/diagnostics' / str(time.time_ns())
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_before = checkpoint_hashes()
    source_names = ['model.py', 'train_shakespeare.py', 'data.py', 'bpe_tokenizer.py', 'bpe_512.json']
    source_hashes = {name: file_hash(ROOT / name) for name in source_names}
    saved_audit = audit_saved_training()
    print(f'Output: {output_dir}\nGPU: {torch.cuda.get_device_name(0)}', flush=True)
    print('Encoding BPE 512 once on CPU for all four runs...', flush=True)
    tokenizer, train, val = load_shakespeare(tokenizer_type='bpe', bpe_tokenizer_path=ROOT / 'bpe_512.json')
    assert tokenizer.vocab_size == 512
    generator = torch.Generator().manual_seed(SEED)
    batches = [get_batch(train, 128, BATCH_SIZE, 'cpu', generator) for _ in range(UPDATES)]
    batch_hashes = [{'update': i + 1, 'x': tensor_hash(x), 'y': tensor_hash(y)}
                    for i, (x, y) in enumerate(batches)]
    probe = get_batch(train, 128, BATCH_SIZE, 'cpu', torch.Generator().manual_seed(12345))

    torch.manual_seed(SEED)
    base5 = TinyGPT(ModelConfig(tokenizer.vocab_size, attention_impl='custom'))
    native5_rng = torch.get_rng_state().clone()
    native5_head_hash = tensor_hash(base5.output.weight)
    torch.manual_seed(SEED)
    base10 = TinyGPT(ModelConfig10M(tokenizer.vocab_size))
    native10_rng = torch.get_rng_state().clone()
    native10_head_hash = tensor_hash(base10.output.weight)
    native_starts = {size: torch.randint(len(train) - 128, (BATCH_SIZE,),
                                        generator=torch.Generator().set_state(state)).tolist()
                     for size, state in [('5m', native5_rng), ('10m', native10_rng)]}
    base5.load_state_dict({name: base10.state_dict()[name] for name in base5.state_dict()})
    assert all(torch.equal(tensor, base10.state_dict()[name]) for name, tensor in base5.state_dict().items())
    initial5 = {name: tensor.clone() for name, tensor in base5.state_dict().items()}
    initial10 = {name: tensor.clone() for name, tensor in base10.state_dict().items()}
    hashes = {'5m': state_hash(initial5), '10m': state_hash(initial10)}
    cpu_rng_before = torch.get_rng_state().clone()
    print('Auditing the original CPU check for changes to 10M weights or RNG...', flush=True)
    check_on_cpu(base10, tokenizer, train, val)
    assert state_hash(base10.state_dict()) == hashes['10m']
    assert torch.equal(cpu_rng_before, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in base10.parameters())
    causal = {}
    for size, model in [('5m', base5), ('10m', base10)]:
        causal[size] = audit_causal_attention(model.to('cuda'), 'cuda')
        model.cpu()
    print('Causal audit passed: every attention layer, SDPA reference, future gradients, and full-model prefix invariance.', flush=True)
    del base5, base10
    torch.cuda.empty_cache()

    frequencies = torch.bincount(train, minlength=tokenizer.vocab_size).double()
    probabilities = frequencies / frequencies.sum()
    unigram_train_entropy = float(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum())
    unigram_probe_loss = float(-probabilities[probe[1]].log().mean())
    common = {
        'seed': SEED, 'batch_seed': SEED, 'probe_seed': 12345, 'batch_size': BATCH_SIZE,
        'context': 128, 'tokenizer': 'bpe_512.json', 'attention_impl': 'custom',
        'train_tokens': len(train), 'validation_tokens': len(val), 'training_batch_hashes': batch_hashes,
        'probe_batch_hashes': {'x': tensor_hash(probe[0]), 'y': tensor_hash(probe[1])},
        'initialization': 'Native 10M initialization with seed 42; all shared 5M weights copied from that state; each LR run reloads its immutable in-memory state.',
        'initial_state_hashes': hashes, 'all_shared_initial_weights_equal': True,
        'original_trainer_rng_audit': {'first_batch_starts': native_starts,
                                      'same_batches_with_original_trainer': native_starts['5m'] == native_starts['10m'],
                                      'native_5m_head_sha256': native5_head_hash,
                                      'native_10m_head_sha256': native10_head_hash},
        'cpu_check_audit': {'weights_unchanged': True, 'rng_unchanged': True, 'gradients_cleared': True},
        'saved_training_audit': saved_audit, 'causal_attention_audit': causal,
        'uniform_random_loss': math.log(512), 'unigram_train_entropy': unigram_train_entropy,
        'unigram_fixed_probe_loss': unigram_probe_loss,
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32,
        'source_hashes': source_hashes, 'checkpoint_hashes_before': checkpoint_before,
    }
    (output_dir / 'audit.json').write_text(json.dumps(common, indent=2) + '\n')
    reports = []
    for label, size, lr in CASES:
        reports.append(run_case(label, size, lr, initial5 if size == '5m' else initial10,
                                hashes[size], batches, probe, output_dir, saved_audit['5m']['optimizer']))
    checkpoint_after = checkpoint_hashes()
    assert checkpoint_before == checkpoint_after, 'An existing checkpoint changed or a diagnostic checkpoint was created'
    assert source_hashes == {name: file_hash(ROOT / name) for name in source_names}
    combined_path = output_dir / 'curves.csv'
    with combined_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['update', *[report['case'] for report in reports]])
        for update in range(UPDATES + 1):
            writer.writerow([update, *[report['curve'][update]['fixed_train_probe_loss_after_update'] for report in reports]])
    summary = {'common': common, 'cases': [{key: value for key, value in report.items() if key not in ('details', 'curve')} for report in reports],
               'checkpoints_preserved': {'count': len(checkpoint_before), 'sha256_verified': True, 'new_checkpoints': 0},
               'total_optimizer_updates': UPDATES * len(CASES), 'total_wall_seconds': time.perf_counter() - started,
               'loss_samples': {report['case']: {str(update): report['curve'][update] for update in [0, 1, 5, 10, 20, 50, 100]} for report in reports}}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'output_dir': str(output_dir), 'total_optimizer_updates': 400,
                      'checkpoint_count_unchanged': len(checkpoint_before),
                      'fixed_probe_losses': {report['case']: [report['curve'][0]['fixed_train_probe_loss_after_update'], report['curve'][-1]['fixed_train_probe_loss_after_update']] for report in reports}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
