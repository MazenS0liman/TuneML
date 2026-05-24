# ——————————————————————————————————————————————————————————————
# Imports
import os
import time
import copy
import mlflow
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tuneml.core.utils import transformer_lr_schedule, generate_causal_mask
from tuneml.data.automidi import AutoMidiDataset
from tuneml.tokenizers import MidiTokenizer
from tuneml.models.transformer.MidiGPT import MidiGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————————————————————————————————————————————————————
# Loss and training functions
def loss_fn(
	prediction,
	target,
	pad_token_id: int = 0,
	criterion=F.cross_entropy
):
	"""
	Calculates masked token-level loss, ignoring pad positions.

	:param prediction: logits in shape (B, vocab_size, T)
	:type prediction: torch.Tensor
 
	:param target: token targets in shape (B, T)
	:type target: torch.Tensor
 
	:param criterion: per-token loss function, default is cross-entropy
	:type criterion: callable, optional
 
	:return: average loss per non-pad token
	:rtype: torch.Tensor
	"""
	# use ignore_index to make per-token criterion skip padding positions reliably
	_loss = criterion(prediction, target, reduction="none", ignore_index=pad_token_id)

	mask = target.ne(torch.tensor(pad_token_id, device=target.device))
	mask = mask.to(_loss.dtype)
	_loss = _loss * mask

	denom = torch.sum(mask)
	if denom.item() == 0:
		return torch.sum(_loss)
	return torch.sum(_loss) / denom


def collate_fn(batch):
	"""
	Collate function to pad MIDI token sequences and create input-target pairs for autoregressive training.
 
	:param batch: List of MIDI token sequences from the dataset
	:type batch: list
	"""
	if len(batch) == 0:
		return None

	midi_seqs = [item for item in batch if item is not None and len(item) > 1]

	if len(midi_seqs) == 0:
		return None

	midi_seqs = [torch.tensor(seq, dtype=torch.long) for seq in midi_seqs]
	midi_seqs_padded = pad_sequence(midi_seqs, batch_first=True, padding_value=0)

	midi_in = midi_seqs_padded[:, :-1]
	midi_tar = midi_seqs_padded[:, 1:]

	return midi_in, midi_tar


def train_step(
	model: MidiGPT,
	opt,
	sched,
	midi_in,
	midi_tar
):
	"""
	Executes one optimization step.
	"""
	tgt_mask = generate_causal_mask(midi_in.shape[1], device=midi_in.device)
	tgt_key_padding_mask = midi_in.eq(model.pad_token_id)

	predictions = model(
		midi_in,
		tgt_mask=tgt_mask,
		tgt_key_padding_mask=tgt_key_padding_mask
	)

	opt.zero_grad()
	loss = loss_fn(predictions.transpose(-1, -2), midi_tar)

	# detect NaN/inf in loss before backward
	if torch.isnan(loss) or torch.isinf(loss):
		print("Detected NaN/inf loss before backward. Inspecting outputs...")
		# basic diagnostics
		with torch.no_grad():
			p = predictions
			print(f"predictions min/max: {p.min().item()}/{p.max().item()}")
			for name, param in model.named_parameters():
				if torch.isnan(param).any() or torch.isinf(param).any():
					print(f"NaN/inf in param: {name}")
		raise RuntimeError("NaN/inf loss encountered")

	loss.backward()
	# gradient clipping to prevent explosion
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
	opt.step()
	sched.step()

	return float(loss)


def val_step(
	model: MidiGPT,
	midi_in,
	midi_tar
):
	"""
	Executes one evaluation step.
	"""
	tgt_mask = generate_causal_mask(midi_in.shape[1], device=midi_in.device)
	tgt_key_padding_mask = midi_in.eq(model.pad_token_id)
	predictions = model(
		midi_in,
		tgt_mask=tgt_mask,
		tgt_key_padding_mask=tgt_key_padding_mask
	)
	loss = loss_fn(predictions.transpose(-1, -2), midi_tar)
	if torch.isnan(loss) or torch.isinf(loss):
		print("Detected NaN/inf loss during validation. Inspecting outputs...")
		with torch.no_grad():
			p = predictions
			print(f"val predictions min/max: {p.min().item()}/{p.max().item()}")
			for name, param in model.named_parameters():
				if torch.isnan(param).any() or torch.isinf(param).any():
					print(f"NaN/inf in param: {name}")
		raise RuntimeError("NaN/inf validation loss encountered")

	return float(loss)

# ——————————————————————————————————————————————————————————————
# Midi GPT Hyperparameters
class MidiGPTHparams:
	"""
	Hyperparameters for MidiGPT model and training.
	"""
	D_MODEL = 512
	NUM_LAYERS = 4
	NUM_HEADS = 8
	D_FF = 2048
	MAX_MIDI_LEN = 2048
	BIAS = True
	DROPOUT = 0.1
	LAYERNORM_EPS = 1e-5

	def __init__(
		self,
		d_model=D_MODEL,
		num_layers=NUM_LAYERS,
		num_heads=NUM_HEADS,
		d_ff=D_FF,
		max_midi_len=MAX_MIDI_LEN,
		bias=BIAS,
		dropout=DROPOUT,
		layernorm_eps=LAYERNORM_EPS
	):
		self.d_model = d_model
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.max_midi_len = max_midi_len
		self.bias = bias
		self.dropout = dropout
		self.layernorm_eps = layernorm_eps

# ——————————————————————————————————————————————————————————————
# Midi GPT Trainer
class MidiGPTTrainer:
	"""
	Trainer for decoder-only MidiGPT.
	"""
	def __init__(
		self,
		hparams=None,
		datapath="./data/processed_dataset.pt",
		batch_size=32,
		warmup_steps=4000,
		ckpt_path="midi_gpt_ckpt.pt",
		load_from_checkpoint=False,
		n_samples=None,
		mlflow_enabled=False,
		mlflow_log_model=False,
	):
		self.midi_tokenizer = MidiTokenizer()

		self.datapath = datapath
		self.batch_size = batch_size
		self.n_samples = n_samples

		if hparams is None:
			hparams = MidiGPTHparams()
			hparams = vars(hparams)
		else:
			if isinstance(hparams, dict):
				hparams_obj = MidiGPTHparams(**hparams)
				hparams = vars(hparams_obj)
			else:
				raise ValueError("hparams must be a dict or None")

		ds = AutoMidiDataset(
			datapath=datapath,
			n_samples=n_samples,
			max_midi_len=hparams["max_midi_len"],
			batch_size=batch_size,
			num_workers=4
		)
		if len(ds) < 2:
			raise ValueError(f"Need at least 2 valid samples to train, found {len(ds)}")

		train_len = max(1, int(0.8 * len(ds)))
		val_len = len(ds) - train_len
		if val_len == 0:
			val_len = 1
			train_len = len(ds) - 1

		self.train_ds, self.val_ds = torch.utils.data.random_split(ds, [train_len, val_len])

		print(
			f"There are {len(ds)} samples in the data, "
			f"{len(self.train_ds)} training samples and {len(self.val_ds)} validation samples"
		)

		self.train_dl = DataLoader(
			self.train_ds,
			batch_size=self.batch_size,
			shuffle=True,
			collate_fn=collate_fn
		)
		self.val_dl = DataLoader(
			self.val_ds,
			batch_size=self.batch_size,
			shuffle=False,
			collate_fn=collate_fn
		)

		self.model = MidiGPT(
			d_model=hparams["d_model"],
			num_layers=hparams["num_layers"],
			num_heads=hparams["num_heads"],
			d_ff=hparams["d_ff"],
			max_midi_len=hparams["max_midi_len"],
			midi_vocab_size=self.midi_tokenizer.vocab_size,
			bias=hparams["bias"],
			dropout=hparams["dropout"],
			layernorm_eps=hparams["layernorm_eps"],
			pad_token_id=0
		).to(device)
		self.hparams = hparams

		self.warmup_steps = warmup_steps
		self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
		self.scheduler = optim.lr_scheduler.LambdaLR(
			self.optimizer,
			lambda x: transformer_lr_schedule(self.hparams["d_model"], x, self.warmup_steps)
		)

		self.ckpt_path = ckpt_path
		self.train_losses = []
		self.val_losses = []
		self.best_val_loss = float("inf")
		self.best_state_dict = None
		self.best_ckpt_path = os.path.splitext(self.ckpt_path)[0] + "_best.pt"
		self.mlflow_enabled = bool(mlflow_enabled)
		self.mlflow_log_model = bool(mlflow_log_model)

		if load_from_checkpoint and os.path.isfile(self.ckpt_path):
			self.load()

	def save(self, ckpt_path=None):
		"""
		Saves a training checkpoint.
		"""
		if ckpt_path is not None:
			self.ckpt_path = ckpt_path

		ckpt = {
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"scheduler_state_dict": self.scheduler.state_dict(),
			"train_losses": self.train_losses,
			"validation_losses": self.val_losses,
			"warmup_steps": self.warmup_steps,
			"hparams": self.hparams,
			"batch_size": self.batch_size,
			"n_samples": self.n_samples,
			"midi_vocab_size": self.midi_tokenizer.vocab_size,
		}

		torch.save(ckpt, self.ckpt_path)
		return

	def load(self, ckpt_path=None):
		"""
		Loads a training checkpoint.
		"""
		if ckpt_path is not None:
			self.ckpt_path = ckpt_path

		ckpt = torch.load(self.ckpt_path)

		self.hparams = ckpt["hparams"]
		self.model = MidiGPT(
			d_model=self.hparams["d_model"],
			num_layers=self.hparams["num_layers"],
			num_heads=self.hparams["num_heads"],
			d_ff=self.hparams["d_ff"],
			max_midi_len=self.hparams["max_midi_len"],
			midi_vocab_size=ckpt["midi_vocab_size"],
			bias=self.hparams["bias"],
			dropout=self.hparams["dropout"],
			layernorm_eps=self.hparams["layernorm_eps"],
			pad_token_id=0
		).to(device)

		print("Loading the model...", end="")
		print(self.model.load_state_dict(ckpt["model_state_dict"], strict=False))

		self.warmup_steps = ckpt["warmup_steps"]
		self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
		self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
		self.scheduler = optim.lr_scheduler.LambdaLR(
			self.optimizer,
			lambda x: transformer_lr_schedule(self.hparams["d_model"], x, self.warmup_steps)
		)
		self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

		self.train_losses = ckpt["train_losses"]
		self.val_losses = ckpt["validation_losses"]

		return

	def fit(self, epochs):
		"""
		Trains MidiGPT for the requested number of epochs.
		"""
		train_losses = []
		val_losses = []
		start = time.time()

		print("Beginning training...")
		print(time.strftime("%Y-%m-%d %H:%M"))
		torch.set_float32_matmul_precision("high")

		if self.mlflow_enabled:
			mlflow.log_params({
				"batch_size": self.batch_size,
				"warmup_steps": self.warmup_steps,
				"epochs": epochs,
				"n_samples": self.n_samples,
				**self.hparams,
			})

		try:
			epoch_iterator = tqdm(range(epochs), desc=f"Training Epoch: 0/{epochs}")
			for epoch in epoch_iterator:
				epoch_iterator.set_description(f"Training Epoch: {epoch + 1}/{epochs}")
				train_epoch_losses = []
				val_epoch_losses = []

				self.model.train()
				for batch in self.train_dl:
					if batch is None:
						continue
					train_midi_in, train_midi_tar = [x.to(device) for x in batch]
					loss = train_step(
						self.model,
						self.optimizer,
						self.scheduler,
						train_midi_in,
						train_midi_tar
					)
					train_epoch_losses.append(loss)

				self.model.eval()
				with torch.no_grad():
					for batch in self.val_dl:
						if batch is None:
							continue
						val_midi_in, val_midi_tar = [x.to(device) for x in batch]
						loss = val_step(self.model, val_midi_in, val_midi_tar)
						val_epoch_losses.append(loss)

				if len(train_epoch_losses) == 0 or len(val_epoch_losses) == 0:
					raise RuntimeError(
						"No valid batches were produced. Check dataset preprocessing and sequence lengths."
					)

				train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
				val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

				self.train_losses.append(train_mean)
				train_losses.append(train_mean)
				self.val_losses.append(val_mean)
				val_losses.append(val_mean)

				print(
					f"Epoch: {epoch + 1} - Time: {round(time.time() - start, 2)} seconds - "
					f"Train Loss: {train_losses[-1]} - Val Loss: {val_losses[-1]}"
				)
				start = time.time()

				if self.mlflow_enabled:
					mlflow.log_metrics(
						{
							"train_loss": train_mean,
							"val_loss": val_mean,
							"lr": self.optimizer.param_groups[0]["lr"],
						},
						step=epoch + 1,
					)

				is_best = val_mean < self.best_val_loss
				if is_best:
					self.best_val_loss = val_mean
					self.best_state_dict = copy.deepcopy(self.model.state_dict())

					if self.mlflow_enabled:
						mlflow.log_metric("best_val_loss", self.best_val_loss, step=epoch + 1)
						if self.mlflow_log_model:
							try:
								mlflow.pytorch.log_model(self.model, name="best_midi_gpt")
							except Exception as e:
								print(f"Warning: MLflow model logging failed: {e}. Continuing training.")

		except KeyboardInterrupt:
			pass

		if self.best_state_dict is not None:
			self.model.load_state_dict(self.best_state_dict)

		print("Checkpointing...")
		self.save()
		print("Done")
		print(time.strftime("%Y-%m-%d %H:%M"))

		return train_losses, val_losses
