import argparse
import math

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from data import ROOT
from model import ModelConfig, TinyGPT
from tokenizer import CharacterTokenizer
from utf8_sampling import UTF8TokenFilter


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = ModelConfig(**checkpoint['model_config'])
    tokenizer_type = checkpoint.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer.from_dict(checkpoint['tokenizer_config'])
    elif tokenizer_type == 'char':
        chars = checkpoint['tokenizer_config']['chars'] if 'tokenizer_config' in checkpoint else checkpoint['chars']
        tokenizer = CharacterTokenizer(chars)
    else:
        raise ValueError(f'Unknown checkpoint tokenizer type: {tokenizer_type!r}')
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError('Checkpoint vocabulary size does not match its model')
    model = TinyGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=1.0, top_k=None):
    if not prompt or max_new_tokens < 0:
        raise ValueError('Use a nonempty prompt and nonnegative token count')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and greater than 0')
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError('top_k must be a positive integer or None')
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    vocab = getattr(tokenizer, 'vocab', [])
    utf8_filter = UTF8TokenFilter(vocab, device) if vocab and all(isinstance(piece, bytes) for piece in vocab) else None
    utf8_state = 0
    if utf8_filter is not None:
        for token_id in prompt_ids:
            utf8_state = utf8_filter.advance(utf8_state, token_id)
        if utf8_state != 0:
            raise ValueError('Encoded prompt must end on a complete UTF-8 character')
    for step in range(max_new_tokens):
        logits = model(tokens[:, -model.config.block_size:])[:, -1, :] / temperature
        if utf8_filter is not None:
            allowed = utf8_filter.mask(utf8_state, max_new_tokens - step - 1)
            logits = logits.masked_fill(~allowed.unsqueeze(0), float('-inf'))
        if top_k is not None:
            # Limit k to the vocabulary and keep exactly k candidates, even at ties.
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = torch.full_like(logits, float('-inf')).scatter(-1, indices, values)
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if utf8_filter is not None:
            utf8_state = utf8_filter.advance(utf8_state, next_token.item())
        tokens = torch.cat((tokens, next_token), dim=1)
    if utf8_filter is not None:
        return b''.join(vocab[token_id] for token_id in tokens[0].tolist()).decode('utf-8', errors='strict')
    return tokenizer.decode(tokens[0].tolist())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate text from a Shakespeare checkpoint')
    parser.add_argument('--checkpoint', type=str, default=str(ROOT / 'checkpoints/shakespeare/best.pt'))
    parser.add_argument('--prompt', default='First Citizen:\n')
    parser.add_argument('--max-new-tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=1.0, help='Positive sampling temperature (default: 1.0)')
    parser.add_argument('--top-k', type=int, default=None, help='Sample from the top k tokens (positive integer; omitted: disabled)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and greater than 0')
    if args.top_k is not None and args.top_k <= 0:
        parser.error('--top-k must be a positive integer')
    return args


def main():
    args = parse_args()
    model, tokenizer = load_model(args.checkpoint, args.device)
    print(generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k))


if __name__ == '__main__':
    main()
