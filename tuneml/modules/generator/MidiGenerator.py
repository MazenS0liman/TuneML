# ——————————————————————————————————————————————————————————————
# Imports
import os
import mido
from midi2audio import FluidSynth
from typing import Union, List

import torch
import torch.nn.functional as F

from tuneml.vocab.midi import MidiVocab
from tuneml.core.utils import generate_causal_mask
from tuneml.tokenizers.TextTokenizer import Flan5Tokenizer
from tuneml.models.transformer.midi.MidiTransformer import MidiTransformer
from tuneml.models.transformer.midi.MidiTransformerTrainer import MidiTransformerHparams

# ——————————————————————————————————————————————————————————————
# Midi Generator Class
class MidiGenerator:
    """
    Module for generating MIDI sequences from text prompts using a trained MidiTransformer model.
    The generator loads a trained model checkpoint, processes input text prompts, and generates MIDI token sequences that can be converted to MIDI files.
    """
    def __init__(
        self, 
        model_ckpt_path: str,
        device: str = "cuda"
    ) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            print("CUDA is not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"Using device: {device}")
        
        self.device = device         
        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        self.hparams = ckpt["hparams"]
        midi_vocab_size = self.hparams.get("vocab_size", self.hparams.get("midi_vocab_size"))
        if midi_vocab_size is None:
            raise KeyError("Expected 'vocab_size' or 'midi_vocab_size' in checkpoint hparams")

        self.model = MidiTransformer(
            d_model=self.hparams["d_model"],
            num_layers=self.hparams["num_layers"],
            num_heads=self.hparams["num_heads"],
            d_ff=self.hparams["d_ff"],
            max_abs_position=self.hparams["max_abs_position"],
            midi_vocab_size=midi_vocab_size,
            text_vocab_size=self.hparams["text_vocab_size"],
            bias=self.hparams["bias"],
            dropout=self.hparams["dropout"],
            layernorm_eps=self.hparams["layernorm_eps"],
        ).to(self.device)

        # Support both direct and wrapped checkpoints.
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def bin_to_velocity(
        self,
        bin_index: int,
        step: int = MidiVocab.BIN_STEP
    ) -> int:
        """
        Convert a bin index (0-31) back to a velocity value (0-127).

        :param bin_index: Index of the velocity bin to convert. Must be in the range [0, 128 // step - 1].
        :type bin_index: int
        :param step: Step size for binning velocity values. Must be a divisor of
                        128. Default is MidiVocab.BIN_STEP (e.g., 4).
        :type step: int
        
        :returns: Velocity value corresponding to the input bin index.
        :rtype: int
        """
        if not (0 <= bin_index * step <= 127):
            raise ValueError(f"Bin index must be in the range [0, {128 // step - 1}].")
        
        return int(bin_index * step + step / 2)
        
    def tokens_to_midi(
        self,
        tokens: Union[List[int], torch.Tensor],
        fname: str,
        tempo=500000,
        save_path: str = None
    ) -> mido.MidiFile:
        """
        Translates a set of indices in the Oore et. al, 2018 vocabulary into a midi file

        :param tokens: list of indices in vocab to be translated into a midi file
        :type tokens: list or torch.Tensor
        :param fname: name for single track of midi file returned
        :type fname: str
        :param tempo: tempo of midi file returned in µs / beat, tempo_in_µs_per_beat = 60 * 10e6 / tempo_in_bpm
        :type tempo: int
        :param save_path: optional path to save the generated midi file; if None, the midi file will not be saved
        :type save_path: str or None
        
        :returns: a single-track piano midi file translated from the input vocab
        :rtype: mido.MidiFile
        """
        if tokens is None:
            raise ValueError("Input tokens cannot be None")

        # check tokens is ints, assuming 1d list
        if isinstance(tokens, list):
            if not all(isinstance(i, int) for i in tokens):
                raise ValueError("All tokens must be int type")
        elif isinstance(tokens, torch.Tensor):
            if not all(isinstance(i.item(), int) for i in tokens):
                raise ValueError("All tokens must be int type")
        else:
            raise ValueError("Tokens must be a list or torch.Tensor")

        # set up midi file
        mid = mido.MidiFile()
        meta_track = mido.MidiTrack()
        track = mido.MidiTrack()

        # meta messages; meta time is 0 everywhere to prevent delay in playing notes
        meta_track.append(mido.MetaMessage("track_name").copy(name=fname, time=0))
        meta_track.append(mido.MetaMessage("smpte_offset"))
        # assume time_signature is 4/4
        time_sig = mido.MetaMessage("time_signature")
        time_sig = time_sig.copy(numerator=4, denominator=4, time=0)
        meta_track.append(time_sig)
        # assume key_signature is C
        key_sig = mido.MetaMessage("key_signature", time=0)
        meta_track.append(key_sig)
        # assume tempo is constant at input tempo
        set_tempo = mido.MetaMessage("set_tempo")
        set_tempo = set_tempo.copy(tempo=tempo, time=0)
        meta_track.append(set_tempo)
        # end of meta track
        end = mido.MetaMessage("end_of_track").copy(time=0)
        meta_track.append(end)

        # set up the piano; default channel is 0 everywhere; program=0 -> piano
        program = mido.Message("program_change", channel=0, program=0, time=0)
        track.append(program)
        # dummy pedal off message; control should be < 64
        cc = mido.Message("control_change", time=0)
        track.append(cc)

        # things needed for conversion
        delta_time = 0
        vel = 0

        # reconstruct the performance
        for idx in tokens:
            # if torch tensor, get item
            try:
                idx = idx.item()
            except AttributeError:
                pass
            # if pad token, continue
            if idx <= 0:
                continue
            # adjust idx to ignore pad token
            idx = idx - 1

            # note messages
            if 0 <= idx < MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS:
                # note on event
                if 0 <= idx < MidiVocab.NOTE_ON_EVENTS:
                    note = idx
                    t = "note_on"
                    v = vel  # get velocity from previous event
                # note off event
                else:
                    note = idx - MidiVocab.NOTE_ON_EVENTS
                    t = "note_off"
                    v = 127

                # create note message and append to track
                msg = mido.Message(t)
                msg = msg.copy(note=note, velocity=v, time=delta_time)
                track.append(msg)

                # reinitialize delta_time and velocity to handle subsequent notes
                delta_time = 0
                vel = 0

            # time shift event
            elif MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS <= idx < MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS + MidiVocab.TIME_SHIFT_EVENTS:
                # find cut time in range (1, time_shift_events)
                cut_time = idx - (MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS - 1)
                # scale cut_time by DIV (from vocabulary) to find time in ms; add to delta_time
                delta_time += cut_time * MidiVocab.DIV

            # velocity event
            elif MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS + MidiVocab.TIME_SHIFT_EVENTS <= idx < MidiVocab.NOTE_EVENTS + MidiVocab.TIME_SHIFT_EVENTS + MidiVocab.VELOCITY_EVENTS:
                # get velocity for next note_on in range (0, 127)
                vel = self.bin_to_velocity(idx - (MidiVocab.NOTE_ON_EVENTS + MidiVocab.NOTE_OFF_EVENTS + MidiVocab.TIME_SHIFT_EVENTS))

        # end the track
        end = mido.MetaMessage("end_of_track").copy(time=0)
        track.append(end)

        # append finalized track and return midi file
        mid.tracks.append(meta_track)
        mid.tracks.append(track)
        
        # save midi file if save_path is provided
        if save_path is not None:
            mid.save(os.path.join(save_path, fname + ".mid"))
        
        return mid
    
    def play(
        self,
        midi_path: str,
        soundfont_path: str = './soundfonts/The Ultimate SoundFont Pack/Ultimate Guitar Kit 2.SF2',
        output_wav_path: str = './output.wav',
        loop: bool = False
    ):
        """
        Plays a MIDI file using FluidSynth and the specified SoundFont.
        
        :param midi_path: Path to the MIDI file to be played.
        :type midi_path: str
        :param soundfont_path: Path to the SoundFont (.sf2) file to use for rendering the MIDI.
        :type soundfont_path: str
        :param output_wav_path: Path where the rendered WAV file will be saved.
        :type output_wav_path: str
        :param loop: If True, the MIDI will loop indefinitely on Windows.
        :type loop: bool
        
        :returns: Path to the rendered WAV file.
        :rtype: str
        """
        # Initialize FluidSynth with the SoundFont
        fs = FluidSynth(
            soundfont_path,
            sample_rate=22050
        )

        # Convert MIDI to WAV
        fs.midi_to_audio(midi_path, output_wav_path)

        # On Windows, play once by default (loop=False). Use stop_midi() to force stop.
        try:
            import winsound

            flags = winsound.SND_FILENAME | winsound.SND_ASYNC
            if loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(output_wav_path, flags)
        except Exception:
            # If playback backend is unavailable, keep the rendered WAV for manual playback.
            pass

        return output_wav_path

    def play_midi(
        self,
        midi_path: str,
        soundfont_path: str = './soundfonts/The Ultimate SoundFont Pack/Ultimate Guitar Kit 2.SF2',
        output_wav_path: str = './output.wav',
        loop: bool = False
    ):
        return self.play(midi_path, soundfont_path, output_wav_path, loop)

    def stop_midi(self):
        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:
            pass

    def __call__(
        self,
        text,
        max_midi_length=MidiTransformerHparams.MAX_MIDI_LEN,
        max_text_length=MidiTransformerHparams.MAX_TEXT_LEN,
        temperature=1.0,
        top_k=0,
        top_p=0.0
    ):
        """
        Generates a MIDI sequence from a text prompt using the trained model

        :param text: input text prompt to condition the MIDI generation on
        :type text: str
        :param max_midi_length: maximum length of the generated MIDI sequence
        :type max_midi_length: int
        :param max_text_length: maximum length of the input text sequence
        :type max_text_length: int
        :param temperature: sampling temperature for controlling randomness of generation
        :type temperature: float
        :param top_k: if > 0, only sample from the top_k most likely next tokens at each step
        :type top_k: int
        :param top_p: if > 0, only sample from the smallest set of most likely next tokens whose cumulative probability exceeds top_p at each step
        :type top_p: float
        
        :returns: List of generated MIDI tokens.
        :rtype: list[int]
        """        
        with torch.no_grad():
            self.model.eval()
            # --- Tokenize and encode the input text ---
            text_tokenizer = Flan5Tokenizer()
            text_tokens = text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0).tolist()  # (T_text,)
            text_tokens = text_tokens[:max_text_length]
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T_text)
            text_padding_mask = text_tokens.eq(MidiVocab.PAD_TOKEN)

            # --- Autoregressive generation loop ---
            generated = [MidiVocab.START_TOKEN]
            for _ in range(max(1, max_midi_length - 1)):
                midi_in = torch.tensor(generated, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T_midi)
                tgt_mask = generate_causal_mask(midi_in.shape[1], device=self.device)
                predictions = self.model(text_tokens, midi_in, tgt_mask=tgt_mask, text_padding_mask=text_padding_mask)
                next_token_logits = predictions[:, -1, :] / temperature

                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_values[:, -1]] = float('-inf')

                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')

                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()
                generated.append(next_token)

                if next_token == MidiVocab.END_TOKEN:
                    break

        # Remove special tokens from the returned sequence for downstream MIDI conversion.
        return [
            token
            for token in generated
            if token not in (MidiVocab.START_TOKEN, MidiVocab.END_TOKEN)
        ]
