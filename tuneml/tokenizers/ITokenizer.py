from abc import ABC

class ITokenizer(ABC):
    """
    Interface for tokenizers for converting input data into tokens
    """
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)