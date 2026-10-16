"""Run with .venv/bin/python tests/smoke_chat.py. No training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chat import LocalChat


def run():
    started = time.time()
    metadata = json.loads((ROOT/'logs/japanese_base_v3/base48k_to96k_sft1100_20261010/sft/results.json').read_text())
    paths = [Path(metadata['checkpoint']), Path(metadata['base_checkpoint'])]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    chat = LocalChat('auto')
    checks = {'checkpoint_load': True, 'device': str(chat.device), 'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        assert chat.device.type == 'cuda'
        checks['gpu'] = torch.cuda.get_device_name(0)
    tok = chat.tokenizer
    sample = 'こんにちは。日本語の質問です。'
    assert tok.decode(tok.encode(sample)) == sample
    checks['tokenizer_roundtrip'] = True
    records = []
    model_identity = id(chat.model)
    for q in ['さいきんどう', 'こんにちは', '1+1はいくつ？', '日本の首都は？', '800円の30%引きはいくら？', '空は一般に何色ですか。']:
        answer = chat.answer(q)
        assert answer.reason != 'error', answer.detail
        assert '\ufffd' not in answer.text
        assert len(chat.prompt_ids(q)) + len(answer.tokens) <= 128
        records.append({'question': q, 'answer': answer.text, 'stop_reason': answer.reason, 'generated_tokens': len(answer.tokens)})
    assert id(chat.model) == model_identity
    assert any(r['stop_reason'] == 'eos' for r in records)
    checks['real_eos_stop'] = True
    checks['single_loaded_model_reused'] = True
    try:
        chat.answer('日本語の長い質問です。' * 1000)
        raise AssertionError('long question accepted')
    except ValueError as exc:
        assert '長すぎ' in str(exc)
    checks['long_input_rejected_without_truncation'] = True
    # Isolated failure/empty-output checks, using a stub only inside this test.
    real_model = chat.model
    class Broken:
        def __call__(self, x):
            raise RuntimeError('smoke injected error')
    chat.model = Broken()
    assert chat.answer('こんにちは').reason == 'error'
    class Empty:
        def __call__(self, x):
            z = torch.full((1, x.shape[1], 1024), -100., device=x.device)
            z[0, -1, 3] = 100.
            return z
    chat.model = Empty()
    empty = chat.answer('こんにちは')
    assert empty.text == '' and empty.reason == 'eos'
    chat.model = real_model
    assert chat.answer('こんにちは', 1).reason != 'error'
    checks['generation_error_and_empty_answer_recovery'] = True
    assert LocalChat.repeating([10,11,12,13]*3)
    checks['repetition_detector'] = True
    # Free GPU memory before fresh CLI process checks.
    del real_model, chat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    commands = [
        (['--help'], None, 0),
        (['--device', 'cpu', '--max-new-tokens', '64', '--once', 'さいきんどう'], None, 0),
        ([], '長い質問です。'*1000+'\nさいきんどう\nこんにちは\nexit\n', 0),
        (['--device','cpu'], 'さいきんどう\nこんにちは\nquit\n', 0),
    ]
    cli = []
    for options, stdin, expected in commands:
        result = subprocess.run([sys.executable, str(ROOT/'chat.py'), *options], input=stdin, text=True, capture_output=True, timeout=180, cwd=ROOT)
        assert result.returncode == expected, result.stderr
        assert 'UnicodeEncodeError' not in result.stderr and 'surrogates not allowed' not in result.stderr
        result.stdout.encode('utf-8', 'strict')
        if stdin is not None:
            assert '終了しました。' in result.stdout
        if '--once' in options or (stdin and 'こんにちは' in stdin):
            assert 'AI:' in result.stdout
        cli.append({'options':options, 'returncode':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr})
    checks['cli_help_once_cpu_exit_quit_long_input_recovery'] = True
    assert before == hashes()
    checks['checkpoints_unchanged'] = True
    out = ROOT/'logs/chat_v1_unicode_smoke.json'
    out.write_text(json.dumps({'checks':checks,'real_model_responses':records,'cli':cli,'checkpoint_hashes':before,'elapsed_seconds':time.time()-started},ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks':checks,'responses':records,'saved':str(out)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run()
