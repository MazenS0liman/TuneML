# ——————————————————————————————————————————————————————————————
# Imports
import os
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tuneml.tokenizers import MidiTokenizer, Flan5Tokenizer

# ——————————————————————————————————————————————————————————————
# MidiCap Dataset
class MidiCapDataset(Dataset):
    """
    Dataset for training a model to generate MIDI data from text captions. 
    Each data point consists of a text caption, a corresponding MIDI file, and the tempo of the MIDI file. 
    The dataset is processed and saved as PyTorch tensors for efficient loading during training.
    """
    def __init__(
        self,
        datapath,
        n_samples=None,
        max_text_len=256,
        max_midi_len=2048,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.max_text_len = max_text_len
        self.max_midi_len = max_midi_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Metadata file not found at {datapath}")
        elif datapath.endswith(".pt"):
            data = torch.load(datapath)
        else:
            data = self._process_data(
                datapath,
                n_samples=n_samples,
                batch_size=batch_size,
                max_text_len=max_text_len,
                max_midi_len=max_midi_len,
                num_workers=num_workers,
            )
            # save processed data for future use
            torch.save(data, datapath.replace(".json", ".pt"))
                
        self.midi = data["midi_tokens"]
        self.text = data["text_tokens"]
        
        if n_samples is not None:
            self.midi = self.midi[:n_samples]
            self.text = self.text[:n_samples]
            
        assert self.midi.shape[0] == self.text.shape[0], "Number of MIDI samples must match number of text samples"

    def __len__(self):
        return self.midi.shape[0]
    
    def __getitem__(self, index):
        text = self.text[index]
        midi = self.midi[index]
        
        text_non_pad = int(text.ne(0).sum().item())
        midi_non_pad = int(midi.ne(0).sum().item())

        text_len = max(1, min(text_non_pad if text_non_pad > 0 else 1, self.max_text_len))
        midi_len = max(2, min(midi_non_pad if midi_non_pad > 0 else 2, self.max_midi_len))

        return text[:text_len], midi[:midi_len]
        
    def _process_data(
        self,
        metadata_path: str,
        n_samples: int = None,
        batch_size: int = 32,
        max_text_len: int = 256,
        max_midi_len: int = 2048,
        num_workers: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes Text to MIDI dataset and saves it as PyTorch tensors.
        
        Structure:
            {
                "midi_tokens": Tensor (N, T_midi),
                "text_tokens": Tensor (N, T_text),
                "tempo": Tensor (N,)
            }
        """
        midi_tokenizer = MidiTokenizer()
        text_tokenizer = Flan5Tokenizer()
        midi_list = []
        text_list = []

        def process_batch(batch_rows):
            captions = [row["caption"] for row in batch_rows]
            midi_paths = [os.path.join("datasets/midicaps", row["location"]) for row in batch_rows]

            def tokenize_midi_file(fname):
                return midi_tokenizer(file_path=fname, max_length=max_midi_len)

            text_tokens = text_tokenizer(
                captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_text_len,
            ).input_ids

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                midi_tokens = list(
                    executor.map(
                        tokenize_midi_file,
                        midi_paths,
                    )
                )

            return text_tokens, midi_tokens

        def flush_batch(batch_rows, error_context):
            try:
                text_tokens, midi_tokens = process_batch(batch_rows)
                text_list.extend(text_tokens)
                midi_list.extend(midi_tokens)
            except Exception as e:
                first_row = batch_rows[0] if batch_rows else {}
                print(f"Caption: {first_row.get('caption', 'N/A')}, MIDI Path: {first_row.get('location', 'N/A')}")
                print(f"Error processing {error_context}: {e}")

        batch_rows = []
        
        with open(metadata_path, 'r') as f:
            for i, line in enumerate(f):
                if n_samples is not None and i >= n_samples:
                    break

                row = {}
                try:
                    row = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
                    continue

                try:
                    batch_rows.append(row)
                    if len(batch_rows) < batch_size:
                        continue

                    flush_batch(batch_rows, f"batch ending at line {i}")
                    batch_rows = []

                except Exception as e:
                    print(f"Caption: {row.get('caption', 'N/A')}, MIDI Path: {row.get('location', 'N/A')}")
                    print(f"Error processing line {i}: {e}")
                    batch_rows = []

        if batch_rows:
            flush_batch(batch_rows, "final batch")

        if len(midi_list) == 0 or len(text_list) == 0:
            raise ValueError(
                f"No valid samples were parsed from {metadata_path}. "
                "Check the metadata format and MIDI file paths."
            )
            
        midi_tensors: List[torch.Tensor] = [torch.as_tensor(tokens, dtype=torch.long).flatten() for tokens in midi_list]
        text_tensors: List[torch.Tensor] = [torch.as_tensor(tokens, dtype=torch.long).flatten() for tokens in text_list]

        midi_padded: torch.Tensor = pad_sequence(midi_tensors, batch_first=True, padding_value=0)
        text_padded: torch.Tensor = pad_sequence(text_tensors, batch_first=True, padding_value=0)

        processed_data = {
            "midi_tokens": midi_padded,
            "text_tokens": text_padded,
        }
        
        print(f"MIDI shape: {midi_padded.shape}")
        print(f"Text shape: {text_padded.shape}")

        return processed_data
