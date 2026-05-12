# ——————————————————————————————————————————————————————————————
# Imports
from typing import List
from miditok import REMI, TokenizerConfig, Structured
from tuneml.tokenizers.ITokenizer import ITokenizer

# ——————————————————————————————————————————————————————————————
# MidiTokenizer class
class MidiTokenizer(ITokenizer):
    """
    A class that tokenizes MIDI files into a sequence of tokens based on the REMI representation. 
    It allows for configurable parameters such as pitch range, beat resolution, number of velocities, and the inclusion of special tokens, chords, rests, tempos, time signatures, and program changes. 
    The tokenizer can be used to convert MIDI files into token sequences that can be fed into machine learning models for music generation or analysis.

    **Example Usage:**

    .. code-block:: python
    
        from tuneml.tokenizers.MidiTokenizer import MidiTokenizer

        # Initialize the tokenizer with default parameters
        tokenizer = MidiTokenizer()

        # Tokenize a MIDI file
        tokens = tokenizer("path/to/midi/file.mid")

        # Convert to MIDI and save it
        generated_midi = tokenizer(tokens)
        generated_midi.dump_midi("path/to/generated/file.mid")
    
    """
    def __init__(
        self,
        params= {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
            "special_tokens": [
                "PAD",      # Padding token for sequences shorter than max length
                "BOS",      # Beginning of sequence token
                "EOS",      # End of sequence token
                "MASK"      # Mask token for masked language modeling tasks
            ],
            "use_chords": True,
            "use_rests": False,
            "use_tempos": True,
            "use_time_signatures": True,
            "use_programs": True,
            "num_tempos": 32,  # number of tempo bins
            "tempo_range": (40, 250),  # (min, max)
        }
    ) -> None:
        # Initialize the tokenizer configuration
        config = TokenizerConfig(**params)

        # Creates the tokenizer
        self.tokenizer = Structured(config)

        # Build the vocabulary
        vocab_size = max(self.tokenizer.vocab.values(), default=-1) + 1
        self.vocab: List[str] = [""] * vocab_size
        for token, idx in self.tokenizer.vocab.items():
            self.vocab[idx] = token

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, new_vocab: List[str]):
        if not isinstance(new_vocab, list):
            raise ValueError("Vocabulary must be a list of tokens.")
        self._vocab = new_vocab
        
    def get_idx(self, token: str) -> int:
        """
        Get the index of a token in the vocabulary.
        
        :param token: The token to look up.
        :type token: str
        
        :return: The index of the token in the vocabulary.
        :rtype: int
        """
        if token not in self.tokenizer.vocab:
            raise ValueError(f"Token '{token}' not found in vocabulary.")
        return self.tokenizer.vocab[token]

    def __len__(self):
        return len(self.vocab)

    def __call__(
        self,
        file_path: str = None,
        max_length: int = 2048
    ) -> List[int]:
        """
        Convert a MIDI file to a list of tokens based on the vocabulary.
        
        :param file_path: Path to the MIDI file to be tokenized.
        :type file_path: str
        :param max_length: Maximum length of the output token list. If the number of tokens exceeds this length, it will be truncated. Default is 2048.
        :type max_length: int
        
        :return: A list of token indices.
        :rtype: List[int]
        """
        if file_path is None:
            raise ValueError("file_path must be provided.")

        # Tokenize the MIDI file
        tokens = self.tokenizer(file_path)

        # Convert tokens to indices using the provided vocabulary
        return tokens.ids[:max_length]
    
    def tokens_to_midi(
        self,
        tokens: List[int],
        store_path: str = None
    ) -> None:
        """
        Translates a set of indices in the Oore et. al, 2018 vocabulary into a midi file

        :param tokens: list of indices in vocab to be translated into a midi file
        :type tokens: list or torch.Tensor
        :param store_path: optional path to save the generated midi file; if None, the midi file will not be saved
        :type store_path: str or None
        
        :returns: None
        :rtype: None
        """
        midi_obj = self.tokenizer(tokens)
        if store_path is not None:
            midi_obj.dump_midi(store_path)
        
