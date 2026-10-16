class CharacterTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        if not self.chars or len(set(self.chars)) != len(self.chars):
            raise ValueError('Vocabulary must contain unique characters')
        self.stoi = {char: index for index, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f'Characters outside vocabulary: {sorted(unknown)!r}')
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        return ''.join(self.chars[index] for index in ids)
