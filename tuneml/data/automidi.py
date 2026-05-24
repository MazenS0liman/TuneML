# ——————————————————————————————————————————————————————————————
# Imports
import os
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tuneml.tokenizers import MidiTokenizer

# ——————————————————————————————————————————————————————————————
# Autoregressive MIDI Dataset
class AutoMidiDataset(Dataset):
    """
    Dataset for training on a directory of MIDI files (no JSON metadata required).

    **Description:**
    
        The dataset tokenizes all MIDI files under `midi_dir` and returns token sequences suitable for autoregressive modeling. 
        Each data point is a MIDI token sequence that can be used to train a model to predict the next token in the sequence.
        Because there is no text metadata, this dataset focuses solely on the MIDI token sequences.
        
    **Args:**

        - `midi_dir` (str): Path to the directory containing MIDI files. The dataset will recursively search for files ending with .mid or .midi.
        - `n_samples` (int, optional): Maximum number of MIDI files to process. If None, all MIDI files in the directory will be used.
        - `max_midi_len` (int, optional): Maximum length of MIDI token sequences. Longer sequences will be truncated. Default is 2048.
        - `batch_size` (int, optional): Number of MIDI files to process in parallel when tokenizing. Default is 32.
        - `num_workers` (int, optional): Number of worker threads to use for parallel tokenization. Default is 4.
    
    """

    def __init__(
        self,
        datapath,
        n_samples=None,
        max_midi_len=2048,
        batch_size: int = 32,
        num_workers: int = 4,
        save_path: str = "./train.pt"
    ):
        self.max_midi_len = max_midi_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        if datapath.endswith(".pt"):
            data = torch.load(datapath)
        else:
            if not os.path.isdir(datapath):
                raise FileNotFoundError(f"Directory not found at {datapath}")

            data = self._process_data(
                datapath,
                n_samples=n_samples,
                batch_size=batch_size,
                max_midi_len=max_midi_len,
                num_workers=num_workers,
            )
            # save processed data for future use
            torch.save(data, save_path)

        self.midi = data["midi_tokens"]

        if n_samples is not None:
            self.midi = self.midi[:n_samples]

    def __len__(self):
        return self.midi.shape[0]

    def __getitem__(self, index):
        midi = self.midi[index]

        midi_non_pad = int(midi.ne(0).sum().item())

        midi_len = max(2, min(midi_non_pad if midi_non_pad > 0 else 2, self.max_midi_len))

        return midi[:midi_len]

    def _collect_midi_paths(self, midi_dir: str) -> List[str]:
        midi_paths: List[str] = []
        for root, _, files in os.walk(midi_dir):
            for file_name in files:
                if file_name.lower().endswith((".mid", ".midi")):
                    midi_paths.append(os.path.join(root, file_name))
        midi_paths.sort()
        return midi_paths

    def _process_data(
        self,
        midi_dir: str,
        n_samples: int = None,
        batch_size: int = 32,
        max_midi_len: int = 2048,
        num_workers: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a directory of MIDI files and returns padded token tensors.

        Structure:
            {
                "midi_tokens": Tensor (N, T_midi),
            }
        """
        midi_tokenizer = MidiTokenizer()
        midi_list = []

        midi_paths = self._collect_midi_paths(midi_dir)
        if n_samples is not None:
            midi_paths = midi_paths[:n_samples]

        if len(midi_paths) == 0:
            raise ValueError(
                f"No MIDI files were found under {midi_dir}. "
                "Expected files ending with .mid or .midi"
            )

        def tokenize_midi_file(fname: str):
            return midi_tokenizer(file_path=fname, max_length=max_midi_len)

        def flush_batch(batch_paths: List[str], error_context: str) -> None:
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    midi_tokens = list(executor.map(tokenize_midi_file, batch_paths))
                midi_list.extend(midi_tokens)
            except Exception as e:
                first_path = batch_paths[0] if batch_paths else "N/A"
                print(f"MIDI Path: {first_path}")
                print(f"Error processing {error_context}: {e}")

        batch_paths = []
        for i, midi_path in enumerate(midi_paths):
            batch_paths.append(midi_path)
            if len(batch_paths) < batch_size:
                continue

            flush_batch(batch_paths, f"batch ending at file #{i}")
            batch_paths = []

        if batch_paths:
            flush_batch(batch_paths, "final batch")

        if len(midi_list) == 0:
            raise ValueError(
                f"No valid MIDI samples were parsed from {midi_dir}. "
                "Check MIDI file validity and tokenizer settings."
            )

        midi_tensors: List[torch.Tensor] = [torch.as_tensor(tokens, dtype=torch.long).flatten() for tokens in midi_list]
        midi_padded: torch.Tensor = pad_sequence(midi_tensors, batch_first=True, padding_value=0)

        processed_data = {
            "midi_tokens": midi_padded,
        }

        print(f"MIDI shape: {midi_padded.shape}")

        return processed_data
