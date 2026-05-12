
import os
from typing import List, Tuple, Union

import torch
import torchaudio.transforms as T

from tuneml.core.utils import load_audio, padding
from tuneml.models.vgg.MiniVGG import MiniVGG


class AudioClassifier:
	"""
	Inference module for audio classification using a trained MiniVGG model.
	"""

	def __init__(
		self,
		checkpoint_path: str,
		device: str = "cuda",
		target_h: int = 128,
		target_w: int = 128,
		sample_rate: int = 22050,
	) -> None:
		if not torch.cuda.is_available() and device == "cuda":
			print("CUDA is not available, falling back to CPU")
			device = "cpu"
		else:
			print(f"Using device: {device}")

		self.device = device
		self.target_h = target_h
		self.target_w = target_w
		self.sample_rate = sample_rate

		self.transform = T.MelSpectrogram(
			sample_rate=self.sample_rate,
			n_fft=2048,
			hop_length=512,
			n_mels=128,
			f_min=0,
			f_max=self.sample_rate // 2,
		)
		self.to_db = T.AmplitudeToDB()

		ckpt = torch.load(checkpoint_path, map_location=self.device)
		self.class_to_idx = ckpt.get("class_to_idx")

		if self.class_to_idx:
			num_classes = len(self.class_to_idx)
			self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
		else:
			num_classes = ckpt.get("num_classes", 50)
			self.idx_to_class = {idx: str(idx) for idx in range(num_classes)}

		self.model = MiniVGG(num_classes=num_classes, target_h=self.target_h, target_w=self.target_w).to(self.device)
		state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
		if state_dict is None:
			raise ValueError("Checkpoint must contain 'model_state_dict' or 'state_dict'.")
		self.model.load_state_dict(state_dict)
		self.model.eval()

	def _preprocess_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
		"""
		Converts a waveform tensor into the MiniVGG input shape: (1, 1, H, W).
		"""
		if waveform.ndim != 2:
			raise ValueError(f"waveform must have shape (channels, samples), got {waveform.shape}")

		if waveform.shape[0] > 1:
			waveform = torch.mean(waveform, dim=0, keepdim=True)

		mel = self.transform(waveform)
		mel_db = self.to_db(mel)

		spec = mel_db.squeeze(0).cpu().numpy()
		spec = padding(spec, self.target_h, self.target_w)

		tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
		return tensor.to(self.device)

	def preprocess_file(self, audio_path: str) -> torch.Tensor:
		"""
		Loads and preprocesses an audio file into model-ready input.
		"""
		if not os.path.exists(audio_path):
			raise FileNotFoundError(f"Audio file not found: {audio_path}")
		waveform, _ = load_audio(audio_path, target_sr=self.sample_rate)
		return self._preprocess_waveform(waveform)

	def predict_proba(self, audio: Union[str, torch.Tensor]) -> torch.Tensor:
		"""
		Returns class probabilities for an audio file path or waveform tensor.
		"""
		if isinstance(audio, str):
			x = self.preprocess_file(audio)
		elif isinstance(audio, torch.Tensor):
			x = self._preprocess_waveform(audio)
		else:
			raise TypeError("audio must be either a file path (str) or waveform tensor")

		with torch.no_grad():
			logits = self.model(x)
			probs = torch.softmax(logits, dim=1)
		return probs.squeeze(0).cpu()

	def predict(self, audio: Union[str, torch.Tensor], top_k: int = 1) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
		"""
		Returns top-k predictions as (class_name, probability).
		"""
		probs = self.predict_proba(audio)
		top_k = max(1, min(top_k, probs.shape[0]))
		values, indices = torch.topk(probs, k=top_k)

		output = [
			(self.idx_to_class.get(int(idx), str(int(idx))), float(prob))
			for prob, idx in zip(values.tolist(), indices.tolist())
		]
		return output[0] if top_k == 1 else output

	def __call__(self, audio: Union[str, torch.Tensor], top_k: int = 1):
		return self.predict(audio, top_k=top_k)