#!/usr/bin/env python3
"""自作日本語AI 60M v1.0 — local inference only."""
import argparse
import codecs
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import torch
from bpe_tokenizer import BPETokenizer
from model import ModelConfig, TinyGPT
from utf8_sampling import UTF8TokenFilter

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'logs/japanese_base_v3/base48k_to96k_sft1100_20261010/sft/results.json'
EOS = 3


def safe_print(*values, file=None, **kwargs):
    """Escape only unencodable characters, including lone surrogates."""
    stream = sys.stdout if file is None else file
    encoding = getattr(stream, 'encoding', None) or 'utf-8'
    text = kwargs.pop('sep', ' ').join(str(v) for v in values)
    text = text.encode('utf-8', errors='backslashreplace').decode('utf-8')
    text = text.encode(encoding, errors='backslashreplace').decode(encoding)
    print(text, file=stream, **kwargs)


def decode_output(raw):
    """Preserve valid UTF-8, replace corrupt interior bytes, hold partial tail."""
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    repaired = False
    try:
        text = decoder.decode(raw, final=False)
    except UnicodeDecodeError:
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        text = decoder.decode(raw, final=False)
        repaired = True
    return text, repaired or bool(decoder.getstate()[0])


@dataclass
class Answer:
    text: str
    reason: str
    tokens: list
    detail: str = ''
    unicode_repaired: bool = False


class LocalChat:
    def __init__(self, device='auto'):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDAが使用できません。--device cpu を指定してください。')
        self.device = torch.device(device)
        metadata = json.loads(RESULTS.read_text(encoding='utf-8'))
        self.checkpoint = Path(metadata['checkpoint'])
        cp = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        if cp['update'] != 150 or cp['sentinel_id'] != EOS:
            raise ValueError('想定したupdate150のcheckpointではありません。')
        if abs(cp['validation_loss'] - metadata['best_validation_loss']) > 1e-8:
            raise ValueError('checkpointとresults.jsonのValidation Lossが一致しません。')
        self.tokenizer_checkpoint = Path(cp['base_checkpoint'])
        base = torch.load(self.tokenizer_checkpoint, map_location='cpu', weights_only=False)
        self.tokenizer = BPETokenizer.from_dict(base['tokenizer_config'])
        del base
        cfg = ModelConfig(**cp['model_config'])
        if cfg.block_size != 128 or cfg.vocab_size != self.tokenizer.vocab_size:
            raise ValueError('contextまたはtokenizer設定が一致しません。')
        self.context = cfg.block_size
        self.model = TinyGPT(cfg)
        self.model.load_state_dict(cp['model_state_dict'])
        del cp
        self.model.to(self.device).eval()
        self.utf8 = UTF8TokenFilter(self.tokenizer.vocab, self.device)
        if self.tokenizer.encode('\x03') != [EOS]:
            raise ValueError('終了tokenが一致しません。')

    def prompt_ids(self, question):
        if any(0xD800 <= ord(c) <= 0xDFFF for c in question):
            raise ValueError('入力に不正なUnicode文字（サロゲート）が含まれています。端末の文字コードをUTF-8にして再入力してください。入力はモデルに渡していません。')
        if not question.strip():
            raise ValueError('質問を入力してください。')
        if any(t in question for t in ('<|user|>', '<|assistant|>')) or any(ord(c) < 32 and c not in '\n\t' for c in question):
            raise ValueError('質問に会話区切りまたは制御文字は使用できません。')
        prompt = f'<|user|>\n{question}\n<|assistant|>\n'
        ids = self.tokenizer.encode(prompt)
        if self.tokenizer.decode(ids) != prompt:
            raise ValueError('質問を正しくtoken化できませんでした。')
        if len(ids) >= self.context:
            raise ValueError(f'質問が長すぎます（prompt {len(ids)}token、上限127）。短くして再入力してください。質問は切り捨てていません。')
        return ids

    @staticmethod
    def repeating(ids):
        # Stop only substantial consecutive loops; never replace the answer.
        return any(len(ids) >= size * 3 and ids[-size:] == ids[-2*size:-size] == ids[-3*size:-2*size]
                   for size in range(4, 25))

    @torch.inference_mode()
    def answer(self, question, max_new_tokens=64):
        if max_new_tokens < 1:
            raise ValueError('生成上限は1以上にしてください。')
        prefix = self.prompt_ids(question)
        budget = min(max_new_tokens, self.context - len(prefix))
        ids, state = [], 0
        reason = 'context_limit' if budget < max_new_tokens else 'token_limit'
        detail = ''
        try:
            for step in range(budget):
                x = torch.tensor([prefix + ids], dtype=torch.long, device=self.device)
                logits = self.model(x)[0, -1].float()
                logits.masked_fill_(~self.utf8.mask(state, budget-step-1), float('-inf'))
                if not torch.isfinite(logits).any():
                    raise RuntimeError('生成可能なtokenがありません。')
                token = int(logits.argmax())
                ids.append(token)
                state = self.utf8.advance(state, token)
                if token == EOS:
                    reason = 'eos'
                    break
                if state == 0 and self.repeating(ids):
                    reason = 'repetition'
                    break
        except Exception as exc:
            reason, detail = 'error', f'{type(exc).__name__}: {exc}'
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        visible = ids[:-1] if ids[-1:] == [EOS] else ids
        text, repaired = decode_output(b''.join(self.tokenizer.vocab[t] for t in visible))
        return Answer(text.strip(), reason, ids, detail, repaired)


def show(answer):
    safe_print('AI: ' + answer.text if answer.text else 'AI: （空の回答でした）')
    notes = {'token_limit': '生成token上限で停止しました。',
             'context_limit': 'context上限で停止しました。質問は切り捨てていません。',
             'repetition': '反復を検出したため生成を停止しました。',
             'error': '生成中にエラーが発生しました。' + answer.detail}
    if answer.reason in notes:
        safe_print('［' + notes[answer.reason] + '］', file=sys.stderr)
    if answer.unicode_repaired:
        safe_print('［不正なUTF-8 byteを置換、または未完成の末尾byteを除外しました。］', file=sys.stderr)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('1以上を指定してください。')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description='自作日本語AI 60M v1.0（ローカル・独立した短文QA）')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto', help='実行先（既定: auto）')
    parser.add_argument('--max-new-tokens', type=positive, default=64, help='生成token上限（既定: 64、context残量まで）')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', help='質問を一度だけ処理して終了')
    mode.add_argument('--batch', type=Path, help='UTF-8の1行1問ファイルを順に実行し、結果を保存')
    args = parser.parse_args(argv)
    questions = None
    if args.batch is not None:
        try:
            questions = [line.strip() for line in args.batch.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            if not questions:
                raise ValueError('質問ファイルが空です。')
        except (OSError, UnicodeError, ValueError) as exc:
            safe_print(f'質問ファイルを読み込めませんでした: {exc}', file=sys.stderr)
            return 2
    safe_print('自作日本語AI 60M v1.0')
    load_started = time.perf_counter()
    try:
        chat = LocalChat(args.device)
    except Exception as exc:
        safe_print(f'モデルの読み込みに失敗しました: {exc}', file=sys.stderr)
        return 1
    safe_print('モデルを読み込みました。')
    if questions is not None:
        from chat_batch import run_batch
        return run_batch(chat, questions, args.batch, ROOT/'logs/chat_batch20',
                         args.max_new_tokens, safe_print, time.perf_counter()-load_started)
    if args.once is not None:
        try:
            result = chat.answer(args.once, args.max_new_tokens)
            show(result)
            return 1 if result.reason == 'error' else 0
        except ValueError as exc:
            safe_print(str(exc), file=sys.stderr)
            return 2
    safe_print('終了: exit または quit')
    try:
        while True:
            try:
                safe_print('\nあなた: ', end='', flush=True)
                question = sys.stdin.readline()
                if question == '':
                    break
                question = question.rstrip('\r\n')
            except UnicodeDecodeError:
                safe_print('入力をデコードできませんでした。端末の文字コードをUTF-8にして再入力してください。', file=sys.stderr)
                continue
            except EOFError:
                break
            if question.strip().lower() in ('exit', 'quit'):
                break
            if not question.strip():
                continue
            try:
                show(chat.answer(question, args.max_new_tokens))
            except ValueError as exc:
                safe_print(str(exc), file=sys.stderr)
    except KeyboardInterrupt:
        safe_print()
    safe_print('終了しました。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
