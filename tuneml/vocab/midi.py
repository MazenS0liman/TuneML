

class MidiVocab:
    """
    Vocabulary described in Oore et. al, 2018

    Possible MIDI events being considered:
        128 note_on events
        128 note_off events
        125 time_shift events #time_shift = 1: 8 ms
        32  velocity events

    Total midi events = 413

    Indices in the vocabulary:
    v[       0] = '<pad>'
    v[  1..128] = note_on
    v[129..256] = note_off
    v[257..381] = time_shift
    v[382..413] = velocity
    v[414..415] = '<start>', '<end>'
    """
    NOTE_ON_EVENTS = 128
    NOTE_OFF_EVENTS = 128
    NOTE_EVENTS = NOTE_ON_EVENTS + NOTE_OFF_EVENTS
    TIME_SHIFT_EVENTS = 125
    VELOCITY_EVENTS = 32
    
    VOCAB_SIZE = 1 + NOTE_EVENTS + TIME_SHIFT_EVENTS + VELOCITY_EVENTS + 2 # +2 for start and end tokens
    PAD_TOKEN = 0
    START_TOKEN = VOCAB_SIZE - 2
    END_TOKEN = VOCAB_SIZE - 1
    
    LTH = 1000 # max time shift in ms
    DIV = LTH // TIME_SHIFT_EVENTS # time shift step in ms
    BIN_STEP = 128 // VELOCITY_EVENTS # velocity step

    def __init__(self):
        self.vocab = ['<pad>'] + \
                [f'note_on_{i}' for i in range(self.NOTE_ON_EVENTS)] + \
                [f'note_off_{i}' for i in range(self.NOTE_OFF_EVENTS)] + \
                [f'time_shift_{i}' for i in range(1, self.TIME_SHIFT_EVENTS + 1)] + \
                [f'velocity_{i}' for i in range(self.VELOCITY_EVENTS)] + \
                ['<start>', '<end>']
    
    def __len__(self):
        return self.VOCAB_SIZE
    
    def __getitem__(self, index):
        if index < 0 or index >= self.VOCAB_SIZE:
            raise IndexError(f"Index {index} out of range for vocabulary of size {self.VOCAB_SIZE}")
        return self.vocab[index]

    def __contains__(self, event):
        return event in self.vocab

    def index(self, event):
        return self.vocab.index(event)
