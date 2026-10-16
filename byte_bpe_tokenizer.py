"""Lossless byte-level BPE for Unicode corpora with a small vocabulary."""

import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bpe_tokenizer import BPETokenizer


class ByteBPETokenizer(BPETokenizer):
    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size() if self.backend is not None else 0

    def train(self, text, vocab_size):
        if not text or type(vocab_size) is not int or vocab_size < 256:
            raise ValueError('Use nonempty training text and a vocabulary of at least 256')
        self.backend = Tokenizer(models.BPE())
        self.backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        # Train on every line. Line boundaries also bound encoding memory usage.
        self.backend.train_from_iterator(text.splitlines(keepends=True), trainer=trainer)
        return self

    def encode(self, text):
        if self.backend is None:
            raise ValueError('Train or load the tokenizer before encoding')
        ids = []
        for line in text.splitlines(keepends=True):
            ids.extend(self.backend.encode(line, add_special_tokens=False).ids)
        return ids

    def decode(self, ids):
        if self.backend is None:
            raise ValueError('Train or load the tokenizer before decoding')
        return self.backend.decode(ids, skip_special_tokens=False)

    def to_dict(self):
        if self.backend is None:
            raise ValueError('Train or load the tokenizer before saving')
        return {'version': 2, 'algorithm': 'byte-level-bpe',
                'backend': json.loads(self.backend.to_str())}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or payload.get('version') != 2
                or payload.get('algorithm') != 'byte-level-bpe'
                or not isinstance(payload.get('backend'), dict)
                or payload['backend'].get('model', {}).get('type') != 'BPE'):
            raise ValueError('Invalid byte-level BPE configuration')
        return cls(Tokenizer.from_str(json.dumps(payload['backend'])))
