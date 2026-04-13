from tuneml.tokenizers.ITokenizer import ITokenizer
from transformers import AutoTokenizer

class Flan5Tokenizer(ITokenizer):
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return self.tokenizer.vocab_size

    def __call__(self, text: str, **kwargs):
        return self.tokenizer(text, **kwargs)
