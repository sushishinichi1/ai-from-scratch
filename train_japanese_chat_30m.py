"""Check Japanese chat SFT on the 30M Base v2 checkpoint for 50 updates."""

import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from chat_data import collate_instruction_batch, format_prompt, load_instruction_examples
from generate import generate
from model import ModelConfig, TinyGPT


ROOT = Path(__file__).resolve().parent
BASE_CHECKPOINT = ROOT / 'checkpoints/japanese_base_v2/wide30m_lr0003_bpe1024/best.pt'
TOKENIZER_PATH = ROOT / 'datasets/japanese_base_v2/bpe_1024.json'
SPLIT_DIR = ROOT / 'datasets/japanese_base_v2/chat_instruction_split'
TRAIN_PATH = SPLIT_DIR / 'train.jsonl'
VALIDATION_PATH = SPLIT_DIR / 'validation.jsonl'
SPLIT_REPORT_PATH = SPLIT_DIR / 'split_report.json'
CHECKPOINT_DIR = ROOT / 'checkpoints/japanese_base_v2/chat30m_bpe1024_assistant_only'
LOG_DIR = ROOT / 'logs/japanese_base_v2/chat30m_bpe1024_assistant_only'
SEED = 42
BATCH_SIZE = 8
UPDATES = 50
LEARNING_RATE = 0.0001
GRADIENT_CLIPPING = 1.0
TEST_QUESTIONS = [
    'アルゴリズムとは何ですか。',
    '天気と気候の違いは何ですか。',
    '友人が手伝ってくれました。何と伝えるとよいですか。',
    'なぜ海の水は塩辛いのですか。',
    'コンピュータのCPUは何をする部品ですか。',
]


def masked_loss(model, batch, reduction='mean'):
    x, y, target_mask = batch
    logits = model(x)
    losses = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                             y.reshape(-1), reduction='none').view_as(y)
    if not target_mask.any():
        raise ValueError('Batch contains no assistant response targets')
    selected = losses.masked_select(target_mask)
    return selected.sum() if reduction == 'sum' else selected.mean()


def generate_answers(model, tokenizer, device):
    model.eval()
    results = []
    for question in TEST_QUESTIONS:
        prompt = format_prompt(question)
        torch.manual_seed(SEED)
        if device == 'cuda':
            torch.cuda.manual_seed_all(SEED)
        full_text = generate(model, tokenizer, prompt, max_new_tokens=96,
                             temperature=0.8, top_k=40)
        if not full_text.startswith(prompt):
            raise ValueError('Chat generation did not preserve its prompt')
        raw_continuation = full_text[len(prompt):]
        answer = raw_continuation.split('<|user|>', 1)[0].split('<|assistant|>', 1)[0].strip()
        answer = next((line.strip() for line in answer.splitlines() if line.strip()), '')
        results.append({
            'question': question, 'prompt': prompt, 'raw_continuation': raw_continuation,
            'answer': answer, 'temperature': 0.8, 'top_k': 40,
            'max_new_tokens': 96, 'seed': SEED, 'utf8_safe': '\ufffd' not in full_text,
        })
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the 50-update trial')
    checkpoint_path = CHECKPOINT_DIR / 'step_50.pt'
    if checkpoint_path.exists():
        raise FileExistsError(f'Refusing to overwrite {checkpoint_path}')

    base_state = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    if base_state['tokenizer_type'] != 'bpe':
        raise ValueError('Base checkpoint must use Byte-BPE')
    tokenizer = BPETokenizer.from_dict(base_state['tokenizer_config'])
    dataset_tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    if tokenizer.vocab_size != 1024 or tokenizer.to_dict() != dataset_tokenizer.to_dict():
        raise ValueError('Base checkpoint and instruction data must use the same Byte-BPE 1024')
    config = ModelConfig(**base_state['model_config'])
    if (config.vocab_size, config.embedding_dim, config.num_heads, config.num_layers,
            config.block_size, config.attention_impl) != (1024, 640, 8, 6, 128, 'custom'):
        raise ValueError('Unexpected Base model configuration')
    markers = {}
    for marker in ('<|user|>', '<|assistant|>'):
        ids = tokenizer.encode(marker)
        if tokenizer.decode(ids) != marker:
            raise ValueError(f'Chat marker does not round-trip: {marker}')
        markers[marker] = ids

    if not TRAIN_PATH.exists() or not VALIDATION_PATH.exists() or not SPLIT_REPORT_PATH.exists():
        raise FileNotFoundError('Create the deterministic train/validation split first')
    train_examples = load_instruction_examples(TRAIN_PATH, tokenizer, config.block_size)
    validation_examples = load_instruction_examples(VALIDATION_PATH, tokenizer, config.block_size)
    if len(train_examples) + len(validation_examples) != 108:
        raise ValueError('Expected all 108 source instructions in the split')
    max_length = max(len(x['token_ids']) for x in train_examples + validation_examples)
    if max_length > config.block_size:
        raise ValueError('An instruction exceeds context size')
    model = TinyGPT(config)
    model.load_state_dict(base_state['model_state_dict'])
    del base_state
    model.train()

    # One CPU batch checks the actual chat loader and masked response loss before GPU training.
    cpu_batch = collate_instruction_batch(train_examples, list(range(min(BATCH_SIZE, len(train_examples)))), 'cpu')
    cpu_loss = masked_loss(model, cpu_batch)
    if not math.isfinite(cpu_loss.item()):
        raise FloatingPointError('Nonfinite CPU chat loss')
    cpu_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    cpu_check_loss = cpu_loss.item()
    del cpu_batch, cpu_loss
    print(f'CPU batch forward/backward passed; loss={cpu_check_loss:.6f}', flush=True)

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model.to(device='cuda', dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError('Chat tuning requires FP32 model parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    batch_generator = torch.Generator().manual_seed(SEED)

    def evaluate_dataset(dataset):
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(dataset), BATCH_SIZE):
                ids = list(range(start, min(start + BATCH_SIZE, len(dataset))))
                batch = collate_instruction_batch(dataset, ids, 'cuda')
                loss_sum = masked_loss(model, batch, reduction='sum')
                count = int(batch[2].sum().item())
                total_loss += loss_sum.item()
                total_tokens += count
        model.train(was_training)
        value = total_loss / total_tokens
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite chat evaluation loss')
        return value

    torch.cuda.reset_peak_memory_stats()
    answers_before = generate_answers(model, tokenizer, 'cuda')
    history = [{'update': 0, 'train_loss': evaluate_dataset(train_examples),
                'validation_loss': evaluate_dataset(validation_examples)}]
    model.train()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    for step in range(1, UPDATES + 1):
        indices = torch.randint(len(train_examples), (BATCH_SIZE,), generator=batch_generator)
        batch = collate_instruction_batch(train_examples, indices.tolist(), 'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model, batch)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Nonfinite chat loss at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING, error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0:
            metrics = {'update': step, 'train_loss': evaluate_dataset(train_examples),
                       'validation_loss': evaluate_dataset(validation_examples)}
            history.append(metrics)
            print(f"update={step} train={metrics['train_loss']:.6f} val={metrics['validation_loss']:.6f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    answers_after = generate_answers(model, tokenizer, 'cuda')

    checkpoint_config = {
        'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'seed': SEED,
        'dtype': 'float32', 'amp': False, 'tf32': False,
        'gradient_clipping': GRADIENT_CLIPPING, 'updates': UPDATES,
        'base_checkpoint': str(BASE_CHECKPOINT), 'training_objective': 'assistant-answer-only',
        'chat_markers_added_to_tokenizer': False,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_config': config.to_dict(), 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'tokenizer_type': 'bpe',
        'tokenizer_config': tokenizer.to_dict(), 'step': UPDATES,
        'base_checkpoint': str(BASE_CHECKPOINT), 'train_path': str(TRAIN_PATH),
        'validation_path': str(VALIDATION_PATH), 'split_report_path': str(SPLIT_REPORT_PATH),
        'training_config': checkpoint_config,
        'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
        'batch_generator_state': batch_generator.get_state(),
        'loss_history': history,
    }
    temporary = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    metrics = {
        'base_checkpoint': str(BASE_CHECKPOINT), 'chat_checkpoint': str(checkpoint_path),
        'train_path': str(TRAIN_PATH), 'validation_path': str(VALIDATION_PATH),
        'instruction_count': len(train_examples) + len(validation_examples),
        'train_count': len(train_examples), 'validation_count': len(validation_examples),
        'split_report': json.loads(SPLIT_REPORT_PATH.read_text(encoding='utf-8')),
        'context128_max_token_length': max_length,
        'category_counts': {}, 'tokenizer_unchanged': True,
        'special_token_ids_added': 0, 'markers_as_plain_text': markers,
        'model_config': config.to_dict(), 'training_config': checkpoint_config,
        'cpu_forward_backward_loss': cpu_check_loss,
        'loss_history': history,
        'loss_decreased': history[-1]['train_loss'] < history[0]['train_loss'],
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
        'training_seconds': elapsed,
        'generation': {'settings': {'temperature': 0.8, 'top_k': 40,
                                    'max_new_tokens': 96, 'seed': SEED},
                       'before': answers_before, 'after': answers_after},
    }
    data = [json.loads(line) for path in (TRAIN_PATH, VALIDATION_PATH)
            for line in path.read_text(encoding='utf-8').splitlines()]
    for row in data:
        metrics['category_counts'][row['category']] = metrics['category_counts'].get(row['category'], 0) + 1
    metrics_path = LOG_DIR / 'trial_50_metrics.json'
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Instructions={len(train_examples) + len(validation_examples)}; train={len(train_examples)}; "
          f"validation={len(validation_examples)}; max_tokens={max_length}; "
          f"loss0 train={history[0]['train_loss']:.6f} val={history[0]['validation_loss']:.6f}; "
          f"loss50 train={history[-1]['train_loss']:.6f} val={history[-1]['validation_loss']:.6f}", flush=True)
    print(f'Training={elapsed:.3f}s; peak allocated={metrics["peak_vram_allocated_mib"]:.3f} MiB; '
          f'peak reserved={metrics["peak_vram_reserved_mib"]:.3f} MiB', flush=True)
    print(json.dumps(metrics['generation'], ensure_ascii=False, indent=2), flush=True)
    print(f'Metrics: {metrics_path}', flush=True)


if __name__ == '__main__':
    main()
