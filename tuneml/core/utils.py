import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

def transformer_lr_schedule(
    d_model, 
    step_num, 
    warmup_steps=4000
):
    """
    Implements the learning rate schedule from the original Transformer paper (Vaswani et al., 2017).
    
    :param d_model: The dimensionality of the model's embeddings (also known as the hidden dimension).
    :type d_model: int
    :param step_num: The current training step.
    :type step_num: int
    :param warmup_steps: The number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate.
    :type warmup_steps: int

    :return: The learning rate at the current step.
    :rtype: float
    """
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0

    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

    return (d_model ** -0.5) * arg


def generate_causal_mask(
    seq_len: int, 
    device=None
):
    """
    Creates a causal mask with True values above the diagonal.
    
    :param seq_len: The length of the sequence for which to create the mask.
    :type seq_len: int
    :param device: The device on which to create the mask (e.g., 'cpu' or 'cuda').
    :type device: str

    :return: A (seq_len, seq_len) boolean tensor where True values indicate masked positions.
    :rtype: torch.Tensor
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device), 
        diagonal=1
    ).bool()


def padding(array, xx, yy):
    """
    Pads the input array to the target shape (xx, yy) with zeros.
    
    :param array: The input array to be padded.
    :type array: np.ndarray
    :param xx: The target number of rows.
    :type xx: int
    :param yy: The target number of columns.
    :type yy: int
    
    :return: The padded array with shape (xx, yy).
    :rtype: np.ndarray
    """
    h = array.shape[0]
    w = array.shape[1]
    
    # Trim if larger than target
    if h > xx:
        array = array[:xx, :]
        h = xx
    if w > yy:
        array = array[:, :yy]
        w = yy

    # Pad if smaller than target
    a = (xx - h) // 2
    aa = (xx - h) - a
    b = (yy - w) // 2
    bb = (yy - w) - b
    
    return np.pad(array, ((a, aa), (b, bb)), mode='constant', constant_values=0)


def load_audio(file_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples it to the target sample rate.
    
    :param file_path: The path to the audio file.
    :type file_path: str
    :param target_sr: The target sample rate for resampling (default is 22050).
    :type target_sr: int
    
    :return: A tuple containing the audio waveform as a numpy array and the sample rate.
    :rtype: Tuple[np.ndarray, int]
    """
    waveform, sr = sf.read(file_path)
    
    # Resample if the original sample rate is different from the target sample rate
    if sr != target_sr:
        waveform = librosa.resample(waveform.astype(float), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Convert to PyTorch tensor        
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Handle mono and stereo audio
    if waveform.ndim == 1:  # Mono audio
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    elif waveform.ndim == 2 and waveform.shape[1] == 2:  # Stereo audio
        waveform = waveform.T  # Transpose to (channels, samples)
    else:
        raise ValueError(f"Unsupported audio format with shape {waveform.shape}")
    
    return waveform, sr
