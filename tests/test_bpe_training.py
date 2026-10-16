"""Regression checks for tokenizer selection and checkpoint compatibility."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from bpe_tokenizer import BPETokenizer
from data import load_shakespeare
from generate import load_model
from model import ModelConfig, TinyGPT
from tokenizer import CharacterTokenizer
from train_shakespeare import (
    BATCH_SIZE, LEARNING_RATE, load_checkpoint, main, output_paths,
    parse_args, save_checkpoint, validate_checkpoint,
)


class TokenizerTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_text = 'banana bandana\n' * 4
        self.val_text = 'bandana\nbanana\n' * 2
        (self.root / 'shakespeare_train.txt').write_text(self.train_text, encoding='utf-8')
        (self.root / 'shakespeare_val.txt').write_text(self.val_text, encoding='utf-8')
        self.char = CharacterTokenizer.from_text(self.train_text + self.val_text)
        self.bpe = BPETokenizer().train(self.train_text + self.val_text, 16)
        self.bpe.save(self.root / 'bpe_tokenizer.json')
        self.data_hash = {'shakespeare_train.txt': 'train', 'shakespeare_val.txt': 'validation'}

    def checkpoint(self, tokenizer, attention_impl='custom'):
        config = ModelConfig(tokenizer.vocab_size, block_size=8, embedding_dim=16,
                             num_heads=4, num_layers=1, attention_impl=attention_impl)
        model = TinyGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        path = self.root / 'checkpoint.pt'
        save_checkpoint(path, model, optimizer, tokenizer, 0, float('inf'), self.data_hash)
        return path, load_checkpoint(path), model, optimizer

    def test_default_character_and_bpe_encode_both_splits(self):
        tokenizer, train, val = load_shakespeare(self.root)
        self.assertIsInstance(tokenizer, CharacterTokenizer)
        self.assertEqual(len(train), len(self.train_text))
        self.assertEqual(tokenizer.decode(val.tolist()), self.val_text)
        with patch.object(BPETokenizer, 'train', side_effect=AssertionError('Must load JSON')):
            tokenizer, bpe_train, bpe_val = load_shakespeare(self.root, tokenizer_type='bpe')
        self.assertEqual(tokenizer.decode(bpe_train.tolist()), self.train_text)
        self.assertEqual(tokenizer.decode(bpe_val.tolist()), self.val_text)
        self.assertLess(len(bpe_train), len(train))
        for tokens in (bpe_train, bpe_val):
            self.assertEqual(tokens.dtype, torch.long)
            self.assertEqual(tokens.ndim, 1)
            self.assertGreaterEqual(tokens.min().item(), 0)
            self.assertLess(tokens.max().item(), tokenizer.vocab_size)

    def test_cli_defaults_and_separate_outputs(self):
        self.assertEqual(parse_args([]).tokenizer, 'bpe')
        self.assertEqual(parse_args([]).attention_impl, 'custom')
        self.assertEqual(parse_args([]).bpe_tokenizer.name, 'bpe_512.json')
        legacy_args = parse_args(['--tokenizer', 'char', '--attention-impl', 'torch'])
        self.assertEqual((legacy_args.tokenizer, legacy_args.attention_impl), ('char', 'torch'))
        args = parse_args(['--tokenizer', 'bpe', '--attention-impl', 'custom', '--cpu-check'])
        self.assertTrue(args.cpu_check)
        self.assertEqual((args.tokenizer, args.attention_impl), ('bpe', 'custom'))
        paths = []
        logs = []
        with patch('train_shakespeare.time.time_ns', return_value=123):
            for attention in ('torch', 'custom'):
                for tokenizer in ('char', 'bpe'):
                    directory, log = output_paths(attention, tokenizer)
                    self.assertEqual(directory.name, f'{attention}_{tokenizer}')
                    self.assertIn(f'_{attention}_{tokenizer}_', log.name)
                    paths.append(directory)
                    logs.append(log)
        self.assertEqual(len(set(paths)), 4)
        self.assertEqual(len(set(logs)), 4)

    def test_model_and_training_defaults_are_preserved(self):
        config = ModelConfig(self.bpe.vocab_size, attention_impl='custom')
        self.assertEqual((config.block_size, config.embedding_dim, config.num_heads, config.num_layers),
                         (128, 256, 8, 6))
        self.assertEqual(BATCH_SIZE, 16)
        self.assertEqual(LEARNING_RATE, 0.0005)

    def test_char_and_bpe_checkpoint_restore_without_external_json(self):
        (self.root / 'bpe_tokenizer.json').unlink()
        for tokenizer, kind in [(self.char, 'char'), (self.bpe, 'bpe')]:
            with self.subTest(tokenizer=kind):
                path, checkpoint, model, optimizer = self.checkpoint(tokenizer)
                self.assertEqual(checkpoint['tokenizer_type'], kind)
                validate_checkpoint(checkpoint, model.config, tokenizer, self.data_hash)
                restored_model, restored_tokenizer = load_model(path, 'cpu')
                self.assertEqual(restored_tokenizer.encode(self.train_text), tokenizer.encode(self.train_text))
                self.assertEqual(restored_tokenizer.decode(restored_tokenizer.encode(self.val_text)), self.val_text)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored_model.state_dict()[name]))
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if kind == 'bpe':
                    self.assertEqual(checkpoint['tokenizer_config'], self.bpe.to_dict())
                    self.assertEqual(restored_tokenizer.merges, self.bpe.merges)
                else:
                    self.assertEqual(checkpoint['chars'], self.char.chars)

    def test_tokenizer_type_mismatch_is_rejected_in_both_directions(self):
        for saved, requested in [(self.char, self.bpe), (self.bpe, self.char)]:
            with self.subTest(saved=type(saved).__name__):
                _, checkpoint, model, _ = self.checkpoint(saved)
                with self.assertRaisesRegex(ValueError, 'Checkpoint tokenizer mismatch'):
                    validate_checkpoint(checkpoint, model.config, requested, self.data_hash)

    def test_same_size_bpe_with_different_vocabulary_is_rejected(self):
        _, checkpoint, model, _ = self.checkpoint(self.bpe)
        other = BPETokenizer().train((self.train_text + self.val_text)[::-1], self.bpe.vocab_size)
        self.assertEqual(other.vocab_size, self.bpe.vocab_size)
        self.assertNotEqual(other.to_dict(), self.bpe.to_dict())
        with self.assertRaisesRegex(ValueError, 'bpe tokenizer configuration mismatch'):
            validate_checkpoint(checkpoint, model.config, other, self.data_hash)

    def test_different_merge_order_is_rejected(self):
        _, checkpoint, model, _ = self.checkpoint(self.bpe)
        checkpoint = deepcopy(checkpoint)
        checkpoint['tokenizer_config']['merges'].reverse()
        with self.assertRaisesRegex(ValueError, 'merge order differs'):
            validate_checkpoint(checkpoint, model.config, self.bpe, self.data_hash)

    def test_legacy_character_checkpoint_is_compatible(self):
        path, checkpoint, model, _ = self.checkpoint(self.char)
        del checkpoint['tokenizer_type']
        del checkpoint['tokenizer_config']
        validate_checkpoint(checkpoint, model.config, self.char, self.data_hash)
        torch.save(checkpoint, path)
        _, tokenizer = load_model(path, 'cpu')
        self.assertEqual(tokenizer.chars, self.char.chars)
        with self.assertRaisesRegex(ValueError, 'tokenizer mismatch'):
            validate_checkpoint(checkpoint, model.config, self.bpe, self.data_hash)

    def test_attention_and_dataset_mismatch_checks_are_preserved(self):
        _, checkpoint, model, _ = self.checkpoint(self.bpe)
        config = ModelConfig(**{**model.config.to_dict(), 'attention_impl': 'torch'})
        with self.assertRaisesRegex(ValueError, 'attention_impl mismatch'):
            validate_checkpoint(checkpoint, config, self.bpe, self.data_hash)
        with self.assertRaisesRegex(ValueError, 'Dataset differs'):
            validate_checkpoint(checkpoint, model.config, self.bpe, {'changed': 'dataset'})

    def test_resume_rejects_tokenizer_before_encoding(self):
        path, _, _, _ = self.checkpoint(self.char)
        with patch('sys.argv', ['train_shakespeare.py', '--cpu-check', '--tokenizer', 'bpe', '--resume', str(path)]):
            with patch('train_shakespeare.load_shakespeare', side_effect=AssertionError('Must fail before encoding')):
                with self.assertRaisesRegex(ValueError, "checkpoint='char', requested='bpe'"):
                    main()


if __name__ == '__main__':
    unittest.main()
