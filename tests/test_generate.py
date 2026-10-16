"""Check sampling distributions and CPU generation from compatible checkpoints."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from bpe_tokenizer import BPETokenizer
from custom_byte_bpe_tokenizer import CustomByteBPETokenizer
from generate import generate, load_model, main, parse_args
from model import ModelConfig, TinyGPT
from tokenizer import CharacterTokenizer


class FixedLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.config = SimpleNamespace(block_size=2)
        self.lengths = []
        self.grad_enabled = []

    def forward(self, tokens):
        self.lengths.append(tokens.size(1))
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.logits.expand(tokens.size(0), tokens.size(1), -1)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer('abcdef')
        self.logits = [-2, -1, 0, 1, 2, 3]

    def probabilities(self, temperature=1.0, top_k=None, logits=None):
        model = FixedLogitsModel(self.logits if logits is None else logits)
        with patch('generate.torch.multinomial', return_value=torch.tensor([[0]])) as sample:
            generate(model, self.tokenizer, 'a', max_new_tokens=1,
                     temperature=temperature, top_k=top_k)
        return sample.call_args.args[0][0]

    def test_default_and_explicit_temperature_probabilities(self):
        for temperature in (1.0, 0.8):
            with self.subTest(temperature=temperature):
                expected = torch.softmax(torch.tensor(self.logits) / temperature, dim=0)
                torch.testing.assert_close(self.probabilities(temperature), expected)

    def test_top_k_removes_other_candidates_and_normalizes(self):
        probabilities = self.probabilities(temperature=0.8, top_k=2)
        self.assertEqual(torch.count_nonzero(probabilities[:4]).item(), 0)
        expected = torch.softmax(torch.tensor([2.0, 3.0]) / 0.8, dim=0)
        torch.testing.assert_close(probabilities[4:], expected)
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))

    def test_top_k_keeps_exact_candidate_count_at_ties(self):
        probabilities = self.probabilities(top_k=2, logits=[1] * 6)
        self.assertEqual(torch.count_nonzero(probabilities).item(), 2)
        torch.testing.assert_close(probabilities[probabilities > 0], torch.tensor([0.5, 0.5]))

    def test_top_k_larger_than_vocabulary_is_safe(self):
        torch.testing.assert_close(self.probabilities(top_k=40), self.probabilities())

    def test_top_k_one_uses_best_token_with_cropped_context_and_no_gradients(self):
        model = FixedLogitsModel(self.logits)
        result = generate(model, self.tokenizer, 'abc', max_new_tokens=5, temperature=0.8, top_k=1)
        self.assertEqual(result, 'abcfffff')
        self.assertEqual(model.lengths, [2] * 5)
        self.assertFalse(any(model.grad_enabled))
        self.assertIsNone(model.logits.grad)

    def test_omitted_top_k_preserves_default_sampling(self):
        model = FixedLogitsModel(self.logits)
        torch.manual_seed(123)
        default = generate(model, self.tokenizer, 'a', max_new_tokens=12)
        torch.manual_seed(123)
        explicit = generate(model, self.tokenizer, 'a', max_new_tokens=12, temperature=1.0, top_k=None)
        self.assertEqual(default, explicit)

    def test_byte_sampling_allows_partial_characters_and_filters_invalid_candidates_before_top_k(self):
        tokenizer = CustomByteBPETokenizer()
        logits = [-20.0] * 256
        for token_id, value in [(0x80, 100), (0xc0, 90), (0xe3, 10), (0x81, 9)]:
            logits[token_id] = value
        model = FixedLogitsModel(logits)
        self.assertEqual(generate(model, tokenizer, 'A', max_new_tokens=3,
                                  temperature=0.8, top_k=1), 'A\u3000')

    def test_byte_sampling_finishes_within_token_budget(self):
        tokenizer = CustomByteBPETokenizer()
        logits = [-20.0] * 256
        logits[0xe3], logits[ord('B')] = 10, 9
        self.assertEqual(generate(FixedLogitsModel(logits), tokenizer, 'A',
                                  max_new_tokens=1, top_k=1), 'AB')

    def test_merged_continuations_can_complete_character_in_one_token(self):
        tokenizer = CustomByteBPETokenizer()
        tokenizer.vocab.append(b'\x81\x82')
        tokenizer.merges.append((0x81, 0x82, 256))
        logits = [-20.0] * 257
        logits[0xe3], logits[256], logits[ord('B')] = 20, 19, 1
        self.assertEqual(generate(FixedLogitsModel(logits), tokenizer, 'A',
                                  max_new_tokens=2, top_k=1), 'Aあ')

    def test_byte_sampling_rejects_overlong_surrogate_and_out_of_range_sequences(self):
        for invalid in (b'\xc0\x80', b'\xe0\x80\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80'):
            with self.subTest(invalid=invalid):
                tokenizer = CustomByteBPETokenizer()
                tokenizer.vocab.append(invalid)
                logits = [-20.0] * 257
                logits[256], logits[ord('B')] = 10, 9
                self.assertEqual(generate(FixedLogitsModel(logits), tokenizer, 'A',
                                          max_new_tokens=1, top_k=1), 'AB')

    def test_invalid_sampling_parameters_are_rejected(self):
        model = FixedLogitsModel(self.logits)
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, 'temperature'):
                    generate(model, self.tokenizer, 'a', temperature=temperature)
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, 'top_k'):
                    generate(model, self.tokenizer, 'a', top_k=top_k)
        self.assertEqual(model.lengths, [])

    def test_cli_defaults_and_requested_options(self):
        args = parse_args([])
        self.assertEqual(args.temperature, 1.0)
        self.assertIsNone(args.top_k)
        args = parse_args(['--temperature', '0.8', '--top-k', '40', '--device', 'cpu'])
        self.assertEqual((args.temperature, args.top_k, args.device), (0.8, 40, 'cpu'))

    def test_invalid_cli_temperature_fails_before_loading_checkpoint(self):
        with patch('sys.argv', ['generate.py', '--temperature', '0', '--device', 'cpu']):
            with patch('generate.load_model', side_effect=AssertionError('Must reject before loading')):
                with redirect_stderr(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as error:
                        main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn('--temperature must be finite and greater than 0', output.getvalue())

    def test_invalid_cli_top_k_is_rejected(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args(['--top-k', '0'])
        self.assertEqual(error.exception.code, 2)

    def test_cpu_generation_from_legacy_char_new_char_and_bpe_checkpoints(self):
        char = CharacterTokenizer.from_text('banana ')
        bpe = BPETokenizer.from_dict({
            'version': 1, 'vocab': [' ', 'a', 'b', 'n', 'an', 'ban', 'ana'],
            'merges': [[1, 3, 4], [2, 4, 5], [4, 1, 6]],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            for attention in ('torch', 'custom'):
                for kind, tokenizer in [('legacy_char', char), ('char', char), ('bpe', bpe)]:
                    with self.subTest(attention=attention, tokenizer=kind):
                        config = ModelConfig(tokenizer.vocab_size, block_size=4, embedding_dim=16,
                                             num_heads=4, num_layers=1, attention_impl=attention)
                        model = TinyGPT(config)
                        checkpoint = {'model_config': config.to_dict(), 'model_state_dict': model.state_dict()}
                        if kind == 'legacy_char':
                            checkpoint['chars'] = char.chars
                        else:
                            checkpoint['tokenizer_type'] = kind
                            checkpoint['tokenizer_config'] = bpe.to_dict() if kind == 'bpe' else {'chars': char.chars}
                        torch.save(checkpoint, path)
                        restored, tokenizer = load_model(path, 'cpu')
                        self.assertFalse(restored.training)
                        for temperature, top_k in [(1.0, None), (0.8, 40)]:
                            text = generate(restored, tokenizer, 'banana ', max_new_tokens=5,
                                            temperature=temperature, top_k=top_k)
                            self.assertTrue(text.startswith('banana '))
                            self.assertGreater(len(text), len('banana '))
                        self.assertTrue(all(parameter.grad is None for parameter in restored.parameters()))


if __name__ == '__main__':
    unittest.main()
