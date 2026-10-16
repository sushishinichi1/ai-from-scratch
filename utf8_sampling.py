"""Filter byte tokens by strict UTF-8 prefix validity and completion budget."""

import codecs

import torch


class UTF8TokenFilter:
    # Prefixes represent the eight distinct UTF-8 continuation requirements.
    PREFIXES = (b'', b'\xc2', b'\xe1', b'\xf1', b'\xe0', b'\xed', b'\xf0', b'\xf4')

    @staticmethod
    def prefix_state(pending):
        if not pending:
            return 0
        if len(pending) == 1 and pending[0] in (0xe0, 0xed, 0xf0, 0xf4):
            return {0xe0: 4, 0xed: 5, 0xf0: 6, 0xf4: 7}[pending[0]]
        length = 2 if pending[0] < 0xe0 else 3 if pending[0] < 0xf0 else 4
        return length - len(pending)

    def __init__(self, vocab, device):
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        transitions = []
        for prefix in self.PREFIXES:
            row = []
            for piece in vocab:
                decoder.reset()
                try:
                    decoder.decode(prefix + piece, final=False)
                    row.append(self.prefix_state(decoder.getstate()[0]))
                except UnicodeDecodeError:
                    row.append(-1)
            transitions.append(row)
        # Shortest token path to a complete character boundary. Merged tokens
        # may finish several continuation bytes in a single sampling step.
        distances = [0] + [float('inf')] * 7
        while True:
            updated = [0] + [min([distances[state]] +
                                [1 + distances[target] for target in transitions[state] if target >= 0])
                             for state in range(1, 8)]
            if updated == distances:
                break
            distances = updated
        self.transitions = transitions
        self.masks = {}
        for state, row in enumerate(transitions):
            self.masks[state, None] = torch.tensor([target >= 0 for target in row], device=device)
            for remaining in range(3):
                self.masks[state, remaining] = torch.tensor(
                    [target >= 0 and distances[target] <= remaining for target in row], device=device)

    def mask(self, state, remaining_tokens):
        # The full 256-byte base vocabulary can finish any valid prefix in at
        # most three tokens. Only the final three steps need a budget check.
        budget = remaining_tokens if remaining_tokens < 3 else None
        return self.masks[state, budget]

    def advance(self, state, token_id):
        following = self.transitions[state][token_id]
        if following < 0:
            raise ValueError('Token would produce an invalid UTF-8 prefix')
        return following
