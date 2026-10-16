"""Unicode boundary and CLI recovery tests; no checkpoint needed."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat import Answer, LocalChat, decode_output, main, safe_print, show
from custom_byte_bpe_tokenizer import CustomByteBPETokenizer


class UnicodeTests(unittest.TestCase):
    def test_byte_bpe_decode_and_reported_encode_failure(self):
        tokenizer = CustomByteBPETokenizer()
        text = 'さいきんどう🙂'
        self.assertEqual(tokenizer.decode(list(text.encode('utf-8'))), text)
        decoded = tokenizer.decode(list(b'\xed\xa0\x80'))
        decoded.encode('utf-8', 'strict')
        with self.assertRaises(UnicodeEncodeError) as caught:
            tokenizer.encode('<|user|>\nさ\udce3\udc81きんどう\n<|assistant|>\n')
        self.assertEqual((caught.exception.start, caught.exception.end), (10, 12))

    def test_valid_japanese_and_four_byte_characters_unchanged(self):
        text = 'さいきんどう、日本語🙂𠮷'
        self.assertEqual(decode_output(text.encode()), (text, False))

    def test_every_utf8_boundary(self):
        for text in ['あ', '🙂', '𠮷']:
            raw = text.encode()
            for split in range(1, len(raw)):
                self.assertEqual(decode_output(b'OK'+raw[:split]), ('OK', True))
            self.assertEqual(decode_output(b'OK'+raw), ('OK'+text, False))

    def test_corrupt_and_surrogate_encoded_bytes(self):
        for bad in [b'\xff', b'\x80', b'\xed\xa0\x80', b'\xf0\x80\x80\x80']:
            text, repaired = decode_output('日本'.encode()+bad+'語'.encode())
            self.assertTrue(repaired)
            self.assertTrue(text.startswith('日本') and text.endswith('語'))
            self.assertNotIn(True, [0xD800 <= ord(c) <= 0xDFFF for c in text])
            text.encode('utf-8', 'strict')

    def test_surrogate_input_rejected_before_tokenization(self):
        chat = LocalChat.__new__(LocalChat)
        for invalid in ['\ud800', '\udfff', '\udce3\udc81']:
            with self.assertRaisesRegex(ValueError, 'サロゲート'):
                chat.prompt_ids('さいきん'+invalid+'どう')

    def test_strict_output_surrogates_and_error_detail(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding='utf-8', errors='strict')
        with patch('sys.stdout', stream), patch('sys.stderr', stream):
            show(Answer('正常な日本語\ud800\udcff🙂', 'error', [], '異常\udfff'))
        stream.flush()
        text = raw.getvalue().decode('utf-8', 'strict')
        self.assertIn('正常な日本語\\ud800\\udcff🙂', text)
        self.assertIn('異常\\udfff', text)

    def test_ascii_terminal_fallback(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding='ascii', errors='strict')
        safe_print('日本語\ud800', file=stream)
        stream.flush()
        self.assertIn(b'\\ud800', raw.getvalue())

    def test_interactive_recovers_from_bad_input_and_output(self):
        class FakeChat:
            def __init__(self, device):
                pass
            def answer(self, question, limit):
                if any(0xD800 <= ord(c) <= 0xDFFF for c in question):
                    LocalChat.prompt_ids(self, question)
                return Answer('モデルの日本語\ud800回答', 'eos', [3])
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding='utf-8', errors='strict')
        with patch('chat.LocalChat', FakeChat), patch('sys.stdin', io.StringIO('不正\udce3\udc81\nさいきんどう\nexit\n')), patch('sys.stdout', stream), patch('sys.stderr', stream):
            self.assertEqual(main([]), 0)
        stream.flush()
        output = raw.getvalue().decode('utf-8', 'strict')
        self.assertIn('サロゲート', output)
        self.assertIn('モデルの日本語\\ud800回答', output)
        self.assertIn('終了しました。', output)


if __name__ == '__main__':
    unittest.main()
