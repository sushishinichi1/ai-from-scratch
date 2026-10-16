"""Verify vocabulary comparisons, file selection, and separation of checkpoints."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bpe_tokenizer import BPETokenizer, main, model_parameter_count, parse_args
from data import load_shakespeare
from model import ModelConfig
from train_shakespeare import output_paths, parse_args as training_args, validate_checkpoint


class BPEVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_text = 'banana bandana\n' * 8
        self.val_text = 'bandana\nbanana\n' * 3
        self.corpus = self.train_text + self.val_text
        for name, text in [('corpus.txt', self.corpus), ('train.txt', self.train_text),
                           ('val.txt', self.val_text), ('shakespeare_train.txt', self.train_text),
                           ('shakespeare_val.txt', self.val_text)]:
            (self.root / name).write_text(text, encoding='utf-8')
        self.legacy = BPETokenizer().train(self.corpus, 12)
        self.legacy_path = self.root / 'bpe_tokenizer.json'
        self.legacy.save(self.legacy_path)

    def cli_input_paths(self):
        return ['--text', str(self.root / 'corpus.txt'),
                '--train-text', str(self.root / 'train.txt'),
                '--val-text', str(self.root / 'val.txt'), '--output-dir', str(self.root)]

    def test_cli_accepts_requested_sizes_and_compare_mode(self):
        self.assertEqual(parse_args([]).vocab_size, 256)
        self.assertIsNone(parse_args([]).output)
        for size in (256, 512, 1024):
            self.assertEqual(parse_args(['--vocab-size', str(size)]).vocab_size, size)
        self.assertTrue(parse_args(['--compare']).compare)
        for options in [['--vocab-size', '0'], ['--compare', '--vocab-size', '512'],
                        ['--compare', '--output', 'shared.json']]:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options)

    def test_single_cli_uses_size_in_filename_and_preserves_legacy_json(self):
        before = self.legacy_path.read_bytes()
        argv = ['bpe_tokenizer.py', '--vocab-size', '16', *self.cli_input_paths()]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(self.legacy_path.read_bytes(), before)
        tokenizer = BPETokenizer.load(self.root / 'bpe_16.json')
        self.assertEqual(tokenizer.vocab_size, 16)
        self.assertEqual(tokenizer.decode(tokenizer.encode(self.train_text)), self.train_text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(self.val_text)), self.val_text)

    def test_compare_writes_separate_jsons_and_measured_report(self):
        before = self.legacy_path.read_bytes()
        argv = ['bpe_tokenizer.py', '--compare', *self.cli_input_paths()]
        # This small corpus runs out of pairs before 256; the report must record actual sizes.
        with patch('sys.argv', argv), patch('bpe_tokenizer.model_parameter_count', return_value=123):
            with redirect_stdout(io.StringIO()):
                main()
        self.assertEqual(self.legacy_path.read_bytes(), before)
        with (self.root / 'bpe_vocab_comparison.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row['requested_vocab_size']) for row in rows], [256, 512, 1024])
        for row in rows:
            tokenizer = BPETokenizer.load(self.root / f'bpe_{row["requested_vocab_size"]}.json')
            count = len(tokenizer.encode(self.train_text))
            self.assertEqual(int(row['train_tokens']), count)
            self.assertAlmostEqual(float(row['compression_ratio']), count / len(self.train_text))
            self.assertEqual(int(row['vocab_size']), tokenizer.vocab_size)
            self.assertEqual(tokenizer.decode(tokenizer.encode(self.val_text)), self.val_text)

    def test_comparison_snapshots_match_independent_training(self):
        text = 'banana bandana\nabracadabra🙂\n' * 8
        sizes = [len(set(text)) + offset for offset in (4, 8, 12)]
        tokenizer = BPETokenizer()
        snapshots = {}
        for ids in tokenizer._train_steps(text, max(sizes)):
            if tokenizer.vocab_size in sizes:
                snapshots[tokenizer.vocab_size] = tokenizer.to_dict()
                self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(set(snapshots), set(sizes))
        for size, state in snapshots.items():
            expected = BPETokenizer().train(text, size)
            self.assertEqual(state, expected.to_dict())
            restored = BPETokenizer.from_dict(state)
            self.assertEqual(restored.encode(text), expected.encode(text))

    def test_data_loader_uses_selected_file_and_preserves_default(self):
        larger = BPETokenizer().train(self.corpus, 20)
        path = self.root / 'alternate.json'
        larger.save(path)
        default, _, _ = load_shakespeare(self.root, tokenizer_type='bpe')
        selected, train, val = load_shakespeare(self.root, tokenizer_type='bpe', bpe_tokenizer_path=path)
        self.assertEqual(default.to_dict(), self.legacy.to_dict())
        self.assertEqual(selected.to_dict(), larger.to_dict())
        self.assertEqual(selected.decode(train.tolist()), self.train_text)
        self.assertEqual(selected.decode(val.tolist()), self.val_text)

    def test_training_cli_selects_file_only_for_bpe(self):
        self.assertEqual(training_args([]).bpe_tokenizer.name, 'bpe_512.json')
        self.assertIsNone(training_args(['--tokenizer', 'char']).bpe_tokenizer)
        args = training_args(['--tokenizer', 'bpe', '--bpe-tokenizer', 'bpe_512.json'])
        self.assertEqual(args.bpe_tokenizer, Path('bpe_512.json'))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training_args(['--tokenizer', 'char', '--bpe-tokenizer', 'bpe_512.json'])

    def test_larger_vocabularies_do_not_share_256_checkpoint_or_log_paths(self):
        with patch('train_shakespeare.time.time_ns', return_value=123):
            for attention in ('torch', 'custom'):
                original = output_paths(attention, 'bpe')
                self.assertEqual(output_paths(attention, 'bpe', 256), original)
                outputs = [output_paths(attention, 'bpe', size) for size in (256, 512, 1024)]
                self.assertEqual(len({directory for directory, _ in outputs}), 3)
                self.assertEqual(len({log for _, log in outputs}), 3)
                self.assertEqual(outputs[1][0].name, f'{attention}_bpe_512')
                self.assertEqual(outputs[2][0].name, f'{attention}_bpe_1024')

    def test_model_counts_with_preserved_architecture(self):
        for size, expected in [(256, 4_903_168), (512, 5_034_496), (1024, 5_297_152)]:
            with self.subTest(vocab_size=size):
                self.assertEqual(model_parameter_count(size), expected)

    def test_resume_rejects_different_vocabulary_size(self):
        larger = BPETokenizer().train(self.corpus, 20)
        checkpoint = {'tokenizer_type': 'bpe', 'tokenizer_config': self.legacy.to_dict()}
        with self.assertRaisesRegex(ValueError, 'bpe tokenizer configuration mismatch'):
            validate_checkpoint(checkpoint, ModelConfig(larger.vocab_size), larger, {})


if __name__ == '__main__':
    unittest.main()
