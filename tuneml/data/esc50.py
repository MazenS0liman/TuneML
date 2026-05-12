# ——————————————————————————————————————————————————————————————
# Imports
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tuneml.core.utils import padding, load_audio

# ——————————————————————————————————————————————————————————————
# ESC-50 Dataset
class ESC50Dataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        metadata_file: str, 
        split: str = "train", 
        target_h: int = 128, 
        target_w: int = 128, 
        transform=None
    ):
        super(ESC50Dataset, self).__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.target_h = target_h
        self.target_w = target_w
        self.transform = transform
        
        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] <= 4]
        elif split == "test":
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["filename"])
        label = row["label"]

        # Load the audio file
        waveform, _ = load_audio(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono by averaging channels
        
        # Apply transformations if any
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform
            
        spectrogram = spectrogram.squeeze(0).numpy()  # Remove channel dimension for compatibility with VGG input
        spectrogram = padding(spectrogram, self.target_h, self.target_w)  # Pad to (128, 128) for VGG input
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).float()  # Add channel dimension back for PyTorch

        return spectrogram, label