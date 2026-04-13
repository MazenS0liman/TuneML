import mido
from typing import List, Tuple
from torch import LongTensor
from miditok import REMI, TokenizerConfig

from tuneml.tokenizers.ITokenizer import ITokenizer
from tuneml.vocab.midi import MidiVocab

class MidiTokenizer(ITokenizer):

    def __init__(
        self,
        params= {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 32,
            "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
            "use_chords": True,
            "use_rests": False,
            "use_tempos": True,
            "use_time_signatures": False,
            "use_programs": True,
            "num_tempos": 32,  # number of tempo bins
            "tempo_range": (40, 250),  # (min, max)
        }
    ):
        # Initialize the tokenizer configuration
        config = TokenizerConfig(**params)

        # Creates the tokenizer
        self.tokenizer = REMI(config)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.tokenizer.vocab)

    def events_to_indices(
        self, 
        events, 
        vocab=None
    ) -> List[int]:
        """
        Convert a list of MIDI events to a list of indices based on the vocabulary.
        
        Args:
            events (list[str]): List of MIDI events to be converted to indices.
            vocab (dict): Vocabulary mapping event names to indices. If None, a default vocabulary will be used.

        Returns:
            list[int]: List of indices corresponding to the input MIDI events.
        """
        if vocab is None:
            vocab = self.vocab
        indices = []
        for event in events:
            if event in vocab:
                indices.append(vocab.index(event))
            else:
                raise ValueError(f"Event '{event}' not found in vocabulary.")
        return indices

    def indices_to_events(
        self,
        indices,
        vocab=None
    ) -> List[str]:
        """
        Convert a list of indices back to MIDI events based on the vocabulary.
        
        Args:
            indices (list[int]): List of indices to be converted to MIDI events.
            vocab (dict): Vocabulary mapping event names to indices. If None, a default vocabulary will be used.
        
        Returns:
            list[str]: List of MIDI events corresponding to the input indices.
        """
        if vocab is None:
            vocab = self.vocab
        events = []
        for index in indices:
            if 0 <= index < len(vocab):
                events.append(vocab[index])
            else:
                raise ValueError(f"Index '{index}' is out of bounds for the vocabulary.")
        return events

    def velocity_to_bin(
        self,
        velocity,
        step=MidiVocab.BIN_STEP
    ) -> int:
        """
        Convert a velocity value (0-127) to a bin index (0-31).
        
        Args:
            velocity (int): Velocity value to be converted to a bin index. Must be in the range [0, 127].
            step (int): Step size for binning velocity values. Must be a divisor of 128. Default is 4.
            
        Returns:
            int: Bin index corresponding to the input velocity value.
        """
        if 128 % step != 0:
            raise ValueError("Step must be a divisor of 128.")
        if velocity < 0 or velocity > 127:
            raise ValueError("Velocity must be in the range [0, 127].")
        return velocity // step

    def time_cutter(
        self,
        time, 
        lth=MidiVocab.LTH, 
        div=MidiVocab.DIV
    ) -> List[int]:
        """
        Convert a time value (in ms) to a bin index based on the specified step.
        
        Args:
            time (int): Time value in milliseconds to be converted to a bin index. Must be non-negative.
            lth (int): Maximum time shift in milliseconds. Must be positive and divisible by div. Default is 1000 ms.
            div (int): Time shift step in milliseconds. Must be a positive divisor of lth. Default is 8 ms.
            
        Returns:
            list[int]: List of time shift bin indices corresponding to the input time value.
        """
        def round_(a):
            """
            Custom rounding function for consistent rounding of 0.5 to greater integer
            """
            b = a // 1
            decimal_digits = a % 1
            adder = 1 if decimal_digits >= 0.5 else 0
            return int(b + adder)
        
        if lth % div != 0:
            raise ValueError("lth must be divisible by div")

        time_shifts = []

        for _ in range(time // lth):
            time_shifts.append(round_(lth / div))
        leftover_time_shift = round_((time % lth) / div)
        time_shifts.append(leftover_time_shift) if leftover_time_shift > 0 else None

        return time_shifts

    def time_to_events(
        self,
        delta_time,
        vocab=None
    ) -> Tuple[List[str], List[int]]:
        """
        Convert a time value (in ms) to a list of time_shift events based on the vocabulary.
        
        Args:
            delta_time (int): Time value in milliseconds to be converted to time_shift events.
            vocab (dict): Vocabulary mapping event names to indices. If None, a default vocabulary will be used.
        
        Returns:
            tuple[list[str], list[int]]: A tuple containing the list of time_shift events and their corresponding indices.
        """
        if vocab is None:
            vocab = self.vocab
        time_shifts = self.time_cutter(delta_time)
        events = [f'time_shift_{ts}' for ts in time_shifts]
        indices = self.events_to_indices(events, vocab)
        return events, indices

    def __call__(
        self,
        fname=None,
        max_length=2048, 
        vocab=None
    ) -> Tuple[List[str], List[int]]:
        """
        Convert a MIDI file to a list of tokens based on the vocabulary.
        
        Args:
            fname (str): Path to the MIDI file.
            max_length (int): Maximum length of the output token list. If the number of tokens exceeds this length, it will be truncated. Default is 2048.
            vocab (dict): Vocabulary mapping event names to indices. If None, a default vocabulary will be used.
        
        Returns:
            tuple[list[str], list[int]]: A tuple containing the list of events and their corresponding indices.
        """
        if vocab is None:
            vocab = self.vocab
        if fname is not None:
            midi_obj = mido.MidiFile(fname)
        else:
            raise ValueError("fname must be provided.")
        
        delta_time = 0      # time between midi events (note_on, note_off) to be translated into midi vocab
        events = []         # list of events in midi vocab
        indices = []        # list of indices in midi vocab
        pedal_events = {}   # dict to keep track of pedal events, which affect the duration of notes but are not considered important enough to be included in the midi vocab
        pedal_flag = False  # flag to indicate whether the pedal is currently pressed; affects duration of notes
        
        tempo = 0           # tempo of midi file
        # translate midi file to event list
        for track in midi_obj.tracks:
            for msg in track:
                # increase delta_time by msg time for all messages
                delta_time += msg.time

                # meta events are irrelevant
                if msg.is_meta:
                    if (msg.type == 'set_tempo') and (tempo == 0):
                        tempo = msg.tempo
                    continue
                
                # process by message type
                t = msg.type
                vel = 0         # velocity
                
                if t == "note_on":
                    idx = msg.note + 1
                    vel = self.velocity_to_bin(msg.velocity)
                
                elif t == "note_off":
                    note = msg.note
                    
                    if pedal_flag:
                        if note not in pedal_events:
                            pedal_events[note] = 0    
                        pedal_events[note] += 1
                        continue
                    else:
                        idx = MidiVocab.NOTE_ON_EVENTS + note + 1
                
                elif t == "control_change":
                    if msg.control == 64:
                        if msg.value >= 64:
                            pedal_flag = True
                        else:
                            pedal_flag = False
                            # add events for all notes in pedal_events, then clear pedal_events
                            for note, count in pedal_events.items():
                                idx = MidiVocab.NOTE_ON_EVENTS + note + 1
                                for _ in range(count):
                                    events.append(vocab[idx])
                                    indices.append(idx)
                            # reset pedal_events after processing
                            pedal_events.clear()
                    
                    # to prevent adding more events to output lists, continue
                    continue
                
                # if it's not a type of msg we care about, continue to avoid adding to output lists
                else:
                    continue
                
                # process delta_time into events and indices in vocab
                event_list, index_list = self.time_to_events(delta_time, vocab)
                events.extend(event_list)
                indices.extend(index_list)
                
                # append velocity event if note_on
                if t == "note_on":
                    vel_idx = MidiVocab.NOTE_EVENTS + MidiVocab.TIME_SHIFT_EVENTS + vel + 1
                    events.append(self.vocab[vel_idx])
                    indices.append(vel_idx)
                
                events.append(vocab[idx])
                indices.append(idx)
                
                # reset delta_time after processing
                delta_time = 0
                
        return LongTensor(indices[:max_length])
