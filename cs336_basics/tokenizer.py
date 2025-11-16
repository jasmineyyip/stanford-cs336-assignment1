"""BPE Tokenizer implementation."""

def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    """Train BPE tokenizer - to be implemented."""
    raise NotImplementedError("BPE training not yet implemented")


class Tokenizer:
    """BPE Tokenizer - to be implemented."""
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        raise NotImplementedError("Tokenizer not yet implemented")