"""Own UTF-8 byte BPE with a 256-byte base vocabulary and native CPU loops."""

import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np

from bpe_tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parent
_LIBRARY = None
_POINTER = ctypes.POINTER(ctypes.c_uint32)


def native_library():
    global _LIBRARY
    if _LIBRARY is None:
        source = ROOT / 'custom_byte_bpe.cpp'
        binary = ROOT / '.build' / 'custom_byte_bpe.so'
        binary.parent.mkdir(exist_ok=True)
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            temporary = binary.with_suffix('.so.tmp')
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC',
                            str(source), '-o', str(temporary)], check=True)
            temporary.replace(binary)
        _LIBRARY = ctypes.CDLL(str(binary))
        _LIBRARY.bbpe_train.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
                                       ctypes.POINTER(_POINTER), ctypes.POINTER(_POINTER),
                                       ctypes.POINTER(ctypes.c_size_t)]
        _LIBRARY.bbpe_train.restype = ctypes.c_int
        _LIBRARY.bbpe_encode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, _POINTER,
                                        ctypes.c_uint32, ctypes.POINTER(_POINTER),
                                        ctypes.POINTER(ctypes.c_size_t)]
        _LIBRARY.bbpe_encode.restype = ctypes.c_int
        _LIBRARY.bbpe_free.argtypes = [_POINTER]
        _LIBRARY.bbpe_error.restype = ctypes.c_char_p
    return _LIBRARY


class CustomByteBPETokenizer(BPETokenizer):
    def __init__(self):
        super().__init__()
        self.vocab = [bytes([value]) for value in range(256)]

    def train_with_ids(self, text, vocab_size=4096):
        raw = text.encode('utf-8')
        library = native_library()
        pairs, encoded, length = _POINTER(), _POINTER(), ctypes.c_size_t()
        if library.bbpe_train(raw, len(raw), vocab_size,
                              ctypes.byref(pairs), ctypes.byref(encoded), ctypes.byref(length)):
            raise RuntimeError(library.bbpe_error().decode())
        try:
            self.vocab = [bytes([value]) for value in range(256)]
            array = np.ctypeslib.as_array(pairs, shape=((vocab_size - 256) * 2,))
            self.merges = []
            for offset in range(vocab_size - 256):
                left, right = map(int, array[offset * 2:offset * 2 + 2])
                new_id = 256 + offset
                self.merges.append((left, right, new_id))
                self.vocab.append(self.vocab[left] + self.vocab[right])
            return np.ctypeslib.as_array(encoded, shape=(length.value,)).copy()
        finally:
            library.bbpe_free(pairs)
            library.bbpe_free(encoded)

    def train(self, text, vocab_size=4096):
        self.train_with_ids(text, vocab_size)
        return self

    def encode_array(self, text):
        raw = text.encode('utf-8')
        pairs = np.asarray([(left, right) for left, right, _ in self.merges], dtype=np.uint32).reshape(-1)
        encoded, length = _POINTER(), ctypes.c_size_t()
        library = native_library()
        if library.bbpe_encode(raw, len(raw), pairs.ctypes.data_as(_POINTER), len(self.merges),
                               ctypes.byref(encoded), ctypes.byref(length)):
            raise RuntimeError(library.bbpe_error().decode())
        try:
            return np.ctypeslib.as_array(encoded, shape=(length.value,)).copy()
        finally:
            library.bbpe_free(encoded)

    def encode(self, text):
        return self.encode_array(text).tolist()

    def decode(self, ids, errors='replace'):
        pieces = []
        for value in ids:
            token_id = int(value)
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f'Invalid byte BPE token ID: {value}')
            pieces.append(self.vocab[token_id])
        return b''.join(pieces).decode('utf-8', errors=errors)

    def to_dict(self):
        return {'version': 3, 'algorithm': 'custom-byte-level-bpe',
                'base_vocab_size': 256, 'boundary': 'newline',
                'vocab_hex': [value.hex() for value in self.vocab],
                'merges': [list(merge) for merge in self.merges]}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or payload.get('version') != 3
                or payload.get('algorithm') != 'custom-byte-level-bpe'
                or payload.get('base_vocab_size') != 256 or payload.get('boundary') != 'newline'):
            raise ValueError('Invalid custom byte BPE format')
        tokenizer = cls()
        for new_id, merge in enumerate(payload['merges'], start=256):
            if (len(merge) != 3 or merge[2] != new_id
                    or any(type(value) is not int for value in merge)
                    or not 0 <= merge[0] < new_id or not 0 <= merge[1] < new_id):
                raise ValueError('Invalid custom byte BPE merge')
            left, right, _ = merge
            tokenizer.merges.append((left, right, new_id))
            tokenizer.vocab.append(tokenizer.vocab[left] + tokenizer.vocab[right])
        if [value.hex() for value in tokenizer.vocab] != payload['vocab_hex']:
            raise ValueError('Custom byte BPE vocabulary mismatch')
        return tokenizer
