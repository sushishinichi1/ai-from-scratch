"""Check the 10M budget, legacy loading, and isolation from 5M artifacts."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from bpe_tokenizer import BPETokenizer
from data import ROOT
from generate import load_model
from model import ModelConfig, ModelConfig10M, TinyGPT
from train_shakespeare import (
    BATCH_SIZE, LEARNING_RATE, STANDARD_BPE_TOKENIZER, main, output_paths,
    parse_args, validate_checkpoint,
)


class Model10MTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer.load(STANDARD_BPE_TOKENIZER)

    def test_parameter_budget_and_preserved_5m_model(self):
        larger = TinyGPT(ModelConfig10M(self.tokenizer.vocab_size))
        smaller = TinyGPT(ModelConfig(self.tokenizer.vocab_size, attention_impl='custom'))
        count = sum(parameter.numel() for parameter in larger.parameters())
        self.assertGreaterEqual(count, 9_000_000)
        self.assertLessEqual(count, 11_000_000)
        self.assertEqual(sum(parameter.numel() for parameter in smaller.parameters()), 5_034_496)
        self.assertEqual(smaller.config.num_layers, 6)
        self.assertEqual(larger.config.vocab_size, self.tokenizer.vocab_size)
        self.assertEqual(larger.config.num_layers, 12)
        for name in ('block_size', 'embedding_dim', 'num_heads', 'attention_impl'):
            self.assertEqual(getattr(larger.config, name), getattr(smaller.config, name))

    def test_existing_generation_loader_restores_10m_checkpoint(self):
        model = TinyGPT(ModelConfig10M(self.tokenizer.vocab_size)).eval()
        tokens = torch.tensor([self.tokenizer.encode('First Citizen:')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save({
                'model_config': model.config.to_dict(),
                'model_state_dict': model.state_dict(),
                'tokenizer_type': 'bpe',
                'tokenizer_config': self.tokenizer.to_dict(),
            }, path)
            restored, tokenizer = load_model(path, 'cpu')
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        self.assertEqual(restored.config.num_layers, 12)
        with torch.no_grad():
            torch.testing.assert_close(restored(tokens), model(tokens))

    def test_cli_defaults_fix_standard_bpe_and_10m_update_target(self):
        args = parse_args(['--model-size', '10m'])
        self.assertEqual(args.bpe_tokenizer, ROOT / 'bpe_512.json')
        self.assertEqual((args.tokenizer, args.attention_impl, args.max_updates), ('bpe', 'custom', 1000))
        self.assertEqual(parse_args([]).model_size, '5m')
        self.assertEqual(parse_args([]).max_updates, 5000)
        invalid = [
            ['--tokenizer', 'char'],
            ['--attention-impl', 'torch'],
            ['--bpe-tokenizer', 'bpe_1024.json'],
        ]
        for options in invalid:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['--model-size', '10m', *options])

    def test_10m_checkpoint_and_log_directories_are_separate(self):
        with patch('train_shakespeare.time.time_ns', return_value=123):
            small_dir, small_log = output_paths('custom', 'bpe', 512)
            large_dir, large_log = output_paths('custom', 'bpe', 512, model_size='10m')
        self.assertEqual(small_dir, ROOT / 'checkpoints/shakespeare/custom_bpe_512')
        self.assertEqual(large_dir, ROOT / 'checkpoints/shakespeare/10m/custom_bpe_512')
        self.assertNotIn(small_dir, large_dir.parents)
        self.assertNotEqual(large_log.parent, small_log.parent)
        self.assertEqual(large_log.parent, ROOT / 'logs/10m')

    def test_5m_checkpoint_cannot_resume_as_10m(self):
        hashes = {'shakespeare_train.txt': 'train', 'shakespeare_val.txt': 'validation'}
        checkpoint = {
            'tokenizer_type': 'bpe',
            'tokenizer_config': self.tokenizer.to_dict(),
            'model_config': ModelConfig(self.tokenizer.vocab_size, attention_impl='custom').to_dict(),
            'training_config': {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE},
            'data_hash': hashes,
        }
        with self.assertRaisesRegex(ValueError, 'model configuration does not match'):
            validate_checkpoint(checkpoint, ModelConfig10M(self.tokenizer.vocab_size), self.tokenizer, hashes)

    def test_cpu_failure_prevents_gpu_transfer_and_optimizer_creation(self):
        tokens = torch.arange(256, dtype=torch.long) % self.tokenizer.vocab_size
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('sys.argv', ['train_shakespeare.py', '--model-size', '10m']):
                with patch('train_shakespeare.torch.cuda.is_available', return_value=True), \
                        patch('train_shakespeare.load_shakespeare', return_value=(self.tokenizer, tokens, tokens)), \
                        patch('train_shakespeare.output_paths', return_value=(root / 'checkpoints', root / 'log.csv')), \
                        patch('train_shakespeare.check_on_cpu', side_effect=FloatingPointError('CPU check failed')), \
                        patch.object(TinyGPT, 'to', side_effect=AssertionError('GPU transfer must not run')), \
                        patch('train_shakespeare.torch.optim.AdamW', side_effect=AssertionError('Optimizer must not run')):
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(FloatingPointError, 'CPU check failed'):
                        main()
            self.assertFalse((root / 'checkpoints').exists())
            self.assertFalse((root / 'log.csv').exists())


if __name__ == '__main__':
    unittest.main()
