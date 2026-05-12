# ——————————————————————————————————————————————————————————————
# Imports
import os
import threading
from midi2audio import FluidSynth
from typing import Union, List

import torch
import torch.nn.functional as F

from tuneml.core.utils import generate_causal_mask
from tuneml.tokenizers import Flan5Tokenizer, MidiTokenizer
from tuneml.models.transformer.MidiTransformer import MidiTransformer
from tuneml.trainers.MidiTransformerTrainer import MidiTransformerHparams

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
        self.midi_tokenizer = MidiTokenizer()
        self.text_tokenizer = Flan5Tokenizer()

        self.model = MidiTransformer(
            d_model=self.hparams["d_model"],
            num_layers=self.hparams["num_layers"],
            num_heads=self.hparams["num_heads"],
            d_ff=self.hparams["d_ff"],
            max_midi_len=self.hparams["max_midi_len"],
            max_text_len=self.hparams["max_text_len"],
            midi_vocab_size=self.midi_tokenizer.vocab_size,
            text_vocab_size=self.text_tokenizer.vocab_size,
            bias=self.hparams["bias"],
            dropout=self.hparams["dropout"],
            layernorm_eps=self.hparams["layernorm_eps"],
        ).to(self.device)

        # Support both direct and wrapped checkpoints.
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # Precompute token IDs that should never be sampled at generation time.
        self._banned_token_ids = self._resolve_banned_token_ids()

    def _resolve_banned_token_ids(self):
        banned_ids = set()
        for token_name in ("PAD_None", "MASK_None", "PAD", "MASK"):
            try:
                banned_ids.add(self.midi_tokenizer.get_idx(token_name))
            except ValueError:
                continue
        return sorted(banned_ids)
        
    def tokens_to_midi(
        self,
        tokens: Union[List[int], torch.Tensor],
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
        # Initialize the MIDI tokenizer
        midi_tokenizer = MidiTokenizer()
        
        # Create Midi file if not 
        if store_path and not os.path.exists(store_path):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        # Convert token indices to MIDI tokens and then to a MIDI file
        midi_tokens = [midi_tokenizer.vocab[idx] for idx in tokens]
        midi_tokenizer.tokens_to_midi(midi_tokens, store_path=store_path)
    
    def play(
        self,
        midi_path: str,
        soundfont_path: str = './soundfonts/The Ultimate SoundFont Pack/Ultimate Guitar Kit 2.SF2',
        output_wav_path: str = './output.wav',
        loop: bool = False,
        render_timeout_seconds: int = 30
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
        :param render_timeout_seconds: Maximum time to wait for MIDI-to-WAV rendering. Use 0 or less to disable timeout.
        :type render_timeout_seconds: int
        
        :returns: Path to the rendered WAV file.
        :rtype: str
        """
        # Initialize FluidSynth with the SoundFont
        fs = FluidSynth(
            soundfont_path,
            sample_rate=22050
        )

        # Convert MIDI to WAV with a timeout guard so notebook execution cannot hang indefinitely.
        render_error = []

        def _render_midi_to_audio() -> None:
            try:
                fs.midi_to_audio(midi_path, output_wav_path)
            except Exception as exc:
                render_error.append(exc)

        render_thread = threading.Thread(target=_render_midi_to_audio, daemon=True)
        render_thread.start()

        if render_timeout_seconds > 0:
            render_thread.join(timeout=render_timeout_seconds)
            if render_thread.is_alive():
                return None
        else:
            render_thread.join()

        if render_error:
            raise render_error[0]

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
        loop: bool = False,
        render_timeout_seconds: int = 30
    ):
        return self.play(
            midi_path,
            soundfont_path,
            output_wav_path,
            loop,
            render_timeout_seconds,
        )

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
            text_tokens = self.text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0).tolist()  # (T_text,)
            text_tokens = text_tokens[:max_text_length]
            text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T_text)
            text_padding_mask = text_tokens.eq(0)

            # --- Autoregressive generation loop ---
            generated = [self.midi_tokenizer.get_idx("BOS_None")]
            for _ in range(max(1, max_midi_length - 1)):
                midi_in = torch.tensor(generated, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T_midi)
                tgt_mask = generate_causal_mask(midi_in.shape[1], device=self.device)
                predictions = self.model(text_tokens, midi_in, tgt_mask=tgt_mask, text_padding_mask=text_padding_mask)
                next_token_logits = predictions[:, -1, :] / temperature

                # Ban non-musical special tokens that degrade output quality.
                if self._banned_token_ids:
                    next_token_logits[:, self._banned_token_ids] = float('-inf')

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

                # Guard against pathological filtering where every token is masked out.
                if torch.isneginf(next_token_logits).all():
                    eos_id = self.midi_tokenizer.get_idx("EOS_None")
                    next_token_logits[:, eos_id] = 0.0

                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()
                generated.append(next_token)

                if next_token == self.midi_tokenizer.get_idx("EOS_None"):
                    break

        # Remove special tokens from the returned sequence for downstream MIDI conversion.
        return [
            token
            for token in generated
            if token not in (self.midi_tokenizer.get_idx("BOS_None"), self.midi_tokenizer.get_idx("EOS_None"))
        ]
