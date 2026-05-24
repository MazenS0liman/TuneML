# ——————————————————————————————————————————————————————————————
# Imports
import os
import threading
from typing import Union, List, Optional
from midi2audio import FluidSynth

import torch
import torch.nn.functional as F

from tuneml.core.utils import generate_causal_mask
from tuneml.tokenizers import Flan5Tokenizer, MidiTokenizer
from tuneml.models.transformer.MidiTransformer import MidiTransformer
from tuneml.models.transformer.MidiGPT import MidiGPT
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
        self.model_ckpt_path = model_ckpt_path
        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        self.hparams = ckpt.get("hparams", {})
        self.midi_tokenizer = MidiTokenizer()
        self.text_tokenizer = Flan5Tokenizer()

        # Precompute token IDs that should never be sampled at generation time.
        self._banned_token_ids = self._resolve_banned_token_ids()

    def _build_transformer_from_ckpt(self, ckpt):
        hparams = ckpt["hparams"]
        model = MidiTransformer(
            d_model=hparams["d_model"],
            num_layers=hparams["num_layers"],
            num_heads=hparams["num_heads"],
            d_ff=hparams["d_ff"],
            max_midi_len=hparams["max_midi_len"],
            max_text_len=hparams["max_text_len"],
            midi_vocab_size=self.midi_tokenizer.vocab_size,
            text_vocab_size=self.text_tokenizer.vocab_size,
            bias=hparams["bias"],
            dropout=hparams["dropout"],
            layernorm_eps=hparams["layernorm_eps"],
        ).to(self.device)

        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def _build_gpt_from_ckpt(self, ckpt):
        hparams = ckpt["hparams"]
        model = MidiGPT(
            d_model=hparams["d_model"],
            num_layers=hparams["num_layers"],
            num_heads=hparams["num_heads"],
            d_ff=hparams["d_ff"],
            max_midi_len=hparams["max_midi_len"],
            midi_vocab_size=self.midi_tokenizer.vocab_size,
            bias=hparams["bias"],
            dropout=hparams["dropout"],
            layernorm_eps=hparams["layernorm_eps"],
            pad_token_id=0,
        ).to(self.device)

        state_dict = ckpt.get("model_state_dict", None)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _is_empty_text(text) -> bool:
        if text is None:
            return True
        if isinstance(text, str):
            return text.strip() == ""
        if isinstance(text, (list, tuple)):
            return len(text) == 0
        return False

    def _sample_next_token(
        self, 
        next_token_logits, 
        top_k=0, 
        top_p=0.0
    ) -> int:
        # Ban non-musical special tokens that degrade output quality.
        if self._banned_token_ids:
            next_token_logits[:, self._banned_token_ids] = float('-inf')

        # Apply top-k and top-p filtering.
        if top_k > 0:
            top_k = min(top_k, next_token_logits.shape[-1])
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < top_k_values[:, -1:]] = float('-inf')

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            remove_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[remove_mask] = float('-inf')

        # Guard against pathological filtering where every token is masked out.
        if torch.isneginf(next_token_logits).all():
            eos_id = self.midi_tokenizer.get_idx("EOS_None")
            next_token_logits[:, eos_id] = 0.0

        return torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()

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
        Translates a set of indices into a midi file

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
        text=None,
        midi_file_path: Optional[str] = None,
        max_midi_length=MidiTransformerHparams.MAX_MIDI_LEN,
        max_text_length=MidiTransformerHparams.MAX_TEXT_LEN,
        temperature=1.0,
        top_k=0,
        top_p=0.0
    ) -> List[int]:
        """
        Generates a MIDI sequence from a text prompt or continues from a MIDI file prompt.

        :param text: input text prompt to condition the MIDI generation on
        :type text: str or None
        :param midi_file_path: optional path to a MIDI file used as a token prompt for continuation
        :type midi_file_path: str or None
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
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if midi_file_path is not None and not os.path.isfile(midi_file_path):
            raise FileNotFoundError(f"MIDI prompt file not found: {midi_file_path}")

        # A MIDI prompt implies decoder-only continuation behavior.
        use_gpt = midi_file_path is not None or self._is_empty_text(text)

        with torch.no_grad():
            ckpt = torch.load(self.model_ckpt_path, map_location=self.device)
            if use_gpt:
                self.model: MidiGPT = self._build_gpt_from_ckpt(ckpt)
            else:
                self.model: MidiTransformer = self._build_transformer_from_ckpt(ckpt)

            self.model.eval()

            bos_id = self.midi_tokenizer.get_idx("BOS_None")
            eos_id = self.midi_tokenizer.get_idx("EOS_None")

            if midi_file_path is not None:
                prompt_tokens = self.midi_tokenizer(
                    file_path=midi_file_path,
                    max_length=max_midi_length
                )
                # Keep generation open-ended by dropping EOS from the prompt, if present.
                prompt_tokens = [tok for tok in prompt_tokens if tok != eos_id]
                if len(prompt_tokens) == 0:
                    prompt_tokens = [bos_id]
                elif prompt_tokens[0] != bos_id:
                    prompt_tokens = [bos_id] + prompt_tokens
                    if len(prompt_tokens) > max_midi_length // 2:
                        prompt_tokens = prompt_tokens[:max_midi_length//2]
                if len(prompt_tokens) > max_midi_length:
                    prompt_tokens = prompt_tokens[:max_midi_length]
                generated = prompt_tokens
            else:
                generated = [bos_id]

            text_tokens = None
            text_padding_mask = None
            if not use_gpt:
                # --- Tokenize and encode the input text ---
                text_tokens = self.text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0).tolist()
                text_tokens = text_tokens[:max_text_length]
                text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                text_padding_mask = text_tokens.eq(0)

            remaining_steps = max(0, max_midi_length - len(generated))
            for _ in range(remaining_steps):
                midi_in = torch.tensor(generated, dtype=torch.long, device=self.device).unsqueeze(0)
                tgt_mask = generate_causal_mask(midi_in.shape[1], device=self.device)

                if use_gpt:
                    tgt_key_padding_mask = midi_in.eq(0)
                    predictions = self.model(
                        midi_in,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )
                else:
                    predictions = self.model(
                        text_tokens,
                        midi_in,
                        tgt_mask=tgt_mask,
                        text_padding_mask=text_padding_mask
                    )

                next_token_logits = predictions[:, -1, :] / temperature

                next_token = self._sample_next_token(next_token_logits, top_k=top_k, top_p=top_p)
                generated.append(next_token)

                if next_token == eos_id:
                    break

        # Remove special tokens from the returned sequence for downstream MIDI conversion.
        return [
            token
            for token in generated
            if token not in (bos_id, eos_id)
        ]
