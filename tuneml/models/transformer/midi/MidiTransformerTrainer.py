import os
import time
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tuneml.core.utils import transformer_lr_schedule, generate_causal_mask
from tuneml.data.midi import MidiCapDataset
from tuneml.vocab.midi import MidiVocab
from tuneml.models.transformer.midi.MidiTransformer import MidiTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(
    prediction, 
    target,
    criterion=F.cross_entropy
):
    """
    Calculate the loss between the model's predictions and the target MIDI tokens, while masking out the padding tokens.

    :param prediction: The output of the model, expected to be of shape (batch_size, seq_len, vocab_size) after transposing
    :type prediction: torch.Tensor
    :param target: The true MIDI token indices, expected to be of shape (batch_size, seq_len)
    :type target: torch.Tensor
    :param criterion: The loss function to use, default is cross-entropy loss. It should be able to handle the shape of prediction and target.
    :type criterion: function

    :return: The average loss per non-padding token in the batch
    :rtype: torch.Tensor
    """
    mask = torch.ne(target, torch.zeros_like(target))           # ones where target is 0
    _loss = criterion(prediction, target, reduction='none')     # loss before masking

    # multiply mask to loss elementwise to zero out pad positions
    mask = mask.to(_loss.dtype)
    _loss *= mask

    # output is average over the number of values that were not masked
    return torch.sum(_loss) / torch.sum(mask)


def collate_fn(batch):
    """
    Collate function to be used with the DataLoader for the MidiCapDataset. This function takes a batch of samples from the dataset,
    truncates and pads the text and MIDI token sequences, and prepares the input and target tensors for the model. 
    It also creates the necessary padding masks for the text input.
    """
    
    text_batch, midi_batch = zip(*batch)

    # truncate sequences that are too long
    text_batch = [t[:MidiTransformerHparams.MAX_TEXT_LEN] for t in text_batch]
    midi_batch = [m[:MidiTransformerHparams.MAX_MIDI_LEN] for m in midi_batch]

    # pad sequences to max length in batch with pad token, and stack into tensors
    text_tokens = pad_sequence(text_batch, batch_first=True, padding_value=MidiVocab.PAD_TOKEN)
    midi_tokens = pad_sequence(midi_batch, batch_first=True, padding_value=MidiVocab.PAD_TOKEN)

    # if the midi sequence is only 1 token long, add a pad token to avoid issues with teacher forcing and loss calculation
    if midi_tokens.shape[1] < 2:
        midi_tokens = F.pad(midi_tokens, (0, 1), value=MidiVocab.PAD_TOKEN)

    # --- Teacher forcing ---
    midi_in = midi_tokens[:, :-1]
    midi_tar = midi_tokens[:, 1:]

    # --- Padding mask ---
    text_padding_mask = text_tokens.eq(MidiVocab.PAD_TOKEN)

    return text_tokens, text_padding_mask, midi_in, midi_tar


def train_step(
    model: MidiTransformer, 
    opt, 
    sched, 
    text_tokens, 
    text_padding_mask, 
    midi_in, 
    midi_tar
):
    """
    Computes loss and backward pass for a single training step of the model
    
    :param model: The MidiTransformer model to be trained
    :type model: MidiTransformer
    :param opt: The optimizer initialized with the model's parameters
    :type opt: torch.optim.Optimizer
    :param sched: The learning rate scheduler properly initialized with the optimizer
    :type sched: torch.optim.lr_scheduler._LRScheduler
    :param text_tokens: The tokenized text input batch, expected to be of shape (batch_size, text_seq_len)
    :type text_tokens: torch.Tensor
    :param text_padding_mask: The encoder padding mask for the text tokens, expected to be of shape (batch_size, text_seq_len) with True values indicating padding positions
    :type text_padding_mask: torch.Tensor
    :param midi_in: The decoder input MIDI tokens (shifted right), expected to be of shape (batch_size, midi_seq_len - 1)
    :type midi_in: torch.Tensor
    :param midi_tar: The decoder target MIDI tokens, expected to be of shape (batch_size, midi_seq_len - 1)
    :type midi_tar: torch.Tensor
    
    :return: The loss value before the backward pass, averaged over the non-padding tokens in the batch
    :rtype: float
    """
    tgt_mask = generate_causal_mask(midi_in.shape[1], device=midi_in.device)
    # forward pass
    predictions = model(text_tokens, midi_in, tgt_mask=tgt_mask, text_padding_mask=text_padding_mask)

    # backward pass
    opt.zero_grad()
    loss = loss_fn(predictions.transpose(-1, -2), midi_tar)
    loss.backward()
    opt.step()
    sched.step()

    return float(loss)


def val_step(
    model: MidiTransformer, 
    text_tokens, 
    text_padding_mask, 
    midi_in, 
    midi_tar
):
    """
    Computes loss for a single evaluation / validation step of the model

    :param model: MidiTransformer model to evaluate
    :type model: MidiTransformer
    :param text_tokens: tokenized text input batch
    :type text_tokens: torch.Tensor
    :param text_padding_mask: encoder padding mask for text tokens
    :type text_padding_mask: torch.Tensor
    :param midi_in: decoder input MIDI tokens (shifted right)
    :type midi_in: torch.Tensor
    :param midi_tar: decoder target MIDI tokens
    :type midi_tar: torch.Tensor

    :return: The loss value for the evaluation step
    :rtype: float
    """
    tgt_mask = generate_causal_mask(midi_in.shape[1], device=midi_in.device)
    predictions = model(text_tokens, midi_in, tgt_mask=tgt_mask, text_padding_mask=text_padding_mask)
    loss = loss_fn(predictions.transpose(-1, -2), midi_tar)
    return float(loss)


class MidiTransformerHparams:
    """
    Hyperparameters for the MidiTransformer model and training process. 
    This class is a simple wrapper around a dictionary to allow for easy saving and loading of hyperparameters.
    """
    # Default hyperparameters for the model and training process
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 2048
    MAX_ABS_POSITION = 2048
    MIDI_VOCAB_SIZE = 512
    TEXT_VOCAB_SIZE = 512
    MAX_MIDI_LEN = 2048
    MAX_TEXT_LEN = 256
    BIAS = False
    DROPOUT = 0.1
    LAYERNORM_EPS = 1e-5
    
    def __init__(
        self,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_abs_position=MAX_ABS_POSITION,
        midi_vocab_size=MIDI_VOCAB_SIZE,
        text_vocab_size=TEXT_VOCAB_SIZE,
        max_midi_len=MAX_MIDI_LEN,
        max_text_len=MAX_TEXT_LEN,
        bias=BIAS,
        dropout=DROPOUT,
        layernorm_eps=LAYERNORM_EPS
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_abs_position = max_abs_position
        self.midi_vocab_size = midi_vocab_size
        self.text_vocab_size = text_vocab_size
        self.max_midi_len = max_midi_len
        self.max_text_len = max_text_len
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps

class MidiTransformerTrainer:
    """
    Trainer for the MidiTransformer model.

    This class handles loading the dataset, creating the model, and running the training loop. 
    It also includes methods for saving and loading checkpoints, and for computing the learning rate schedule.
    """
    def __init__(
        self, 
        hparams = None,
        datapath="../data/midicaps/processed_dataset.pt",
        batch_size=32,
        warmup_steps=4000,
        ckpt_path="midi_transformer_ckpt.pt", 
        load_from_checkpoint=False, 
        n_samples=None
    ):
        # get the data
        self.datapath = datapath
        self.batch_size = batch_size
        self.n_samples = n_samples
        data = self._load_dataset(datapath, n_samples=n_samples)

        if len(data) < 2:
            raise ValueError(f"Need at least 2 valid samples to train, found {len(data)}")
        
        if hparams is None:
            hparams = MidiTransformerHparams()
            hparams = vars(hparams)
        else:
            # if hparams is passed in as a dict, convert to MidiTransformerHparams object and back to dict to fill in any missing hyperparameters with defaults
            if isinstance(hparams, dict):
                hparams_obj = MidiTransformerHparams(**hparams)
                hparams = vars(hparams_obj)
            else:
                raise ValueError("hparams must be a dict or None")
            

        # Keep positional encoding bounded to avoid pathological memory use
        if hparams["max_abs_position"] > 0:
            hparams["max_abs_position"] = min(hparams["max_abs_position"], MidiTransformerHparams.MAX_MIDI_LEN)

        # train / validation split: 80 / 20
        train_len = max(1, round(len(data) * 0.8))
        train_data = data[:train_len]
        val_data = data[train_len:]
        if len(val_data) == 0:
            val_data = train_data[:1]

        print(f"There are {len(data)} samples in the data, {len(train_data)} training samples and {len(val_data)} validation samples")

        self.train_ds = MidiCapDataset(
            datapath,
            n_samples=self.n_samples,
            max_text_len=MidiTransformerHparams.MAX_TEXT_LEN,
            max_midi_len=hparams["max_abs_position"]
        )

        train_len = max(1, int(0.8 * len(self.train_ds)))
        val_len = len(self.train_ds) - train_len

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.train_ds, [train_len, val_len]
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
        
        # create model
        self.model = MidiTransformer(
            d_model=hparams["d_model"],
            num_layers=hparams["num_layers"],
            num_heads=hparams["num_heads"],
            d_ff=hparams["d_ff"],
            max_abs_position=hparams["max_abs_position"],
            midi_vocab_size=hparams["midi_vocab_size"],
            text_vocab_size=hparams["text_vocab_size"],
            bias=hparams["bias"],
            dropout=hparams["dropout"],
            layernorm_eps=hparams["layernorm_eps"],
            pad_token_id=MidiVocab.PAD_TOKEN
        ).to(device) 
        self.hparams = hparams

        # setup training
        self.warmup_steps = warmup_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )

        # setup checkpointing / saving
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []

        # load checkpoint if necessesary
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()

    @staticmethod
    def _load_dataset(datapath, n_samples=None):
        """
        Loads dataset ONLY from .pt file (tensor format).
        
        Expected format:
        {
            "midi_tokens": Tensor (N, T_midi),
            "text_tokens": Tensor (N, T_text),
            "tempo": Tensor (N,)   # optional
        }
        """

        dataset = torch.load(datapath)

        if not isinstance(dataset, dict):
            raise ValueError("Dataset must be a dict saved as .pt")

        required_keys = {"midi_tokens", "text_tokens"}
        if not required_keys.issubset(dataset.keys()):
            raise ValueError(f"Dataset must contain keys: {required_keys}")

        midi = dataset["midi_tokens"]
        text = dataset["text_tokens"]

        if n_samples is not None:
            midi = midi[:n_samples]
            text = text[:n_samples]

        # Convert to list-of-dicts for compatibility with DataLoader
        samples = [
            {
                "midi_encoding": midi[i],
                "text_encoding": text[i]
            }
            for i in range(midi.shape[0])
        ]

        return samples

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint at ckpt_path

        Args:
            ckpt_path (str, optional): if None, saves the checkpoint at the previously stored self.ckpt_path
                                       else saves the checkpoints at the new passed-in path, and stores this new path at
                                       the member variable self.ckpt_path
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
            "hparams": self.hparams
        }

        torch.save(ckpt, self.ckpt_path)
        return

    def load(self, ckpt_path=None):
        """
        Loads a checkpoint from ckpt_path

        Args:
            ckpt_path (str, optional): if None, loads the checkpoint at the previously stored self.ckpt_path
                                       else loads the checkpoints from the new passed-in path, and stores this new path
                                       at the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path)

        del self.model, self.optimizer, self.scheduler

        # create and load model
        self.model = MidiTransformer(**ckpt["hparams"]).to(device)
        self.hparams = ckpt["hparams"]
        print("Loading the model...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"], strict=False))

        # create and load load optimizer and scheduler
        self.warmup_steps = ckpt["warmup_steps"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # load loss histories
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]

        return

    def fit(self, epochs):
        """
        Training loop to fit the model to the data stored at the passed in datapath. If KeyboardInterrupt at anytime
        during the training loop, and if progresss being printed, this method will save a checkpoint at the 
        passed-in ckpt_path

        Args:
            epochs: number of epochs to train for.

        Returns:
            history of training and validation losses for this training session
        """
        train_losses = []
        val_losses = []
        start = time.time()

        print("Beginning training...")
        print(time.strftime("%Y-%m-%d %H:%M"))
        torch.set_float32_matmul_precision("high") # this speeds up traning

        try:
            for epoch in tqdm(range(epochs), desc=f"Training Epochs {epochs + 1}: ", unit="epoch"):
                train_epoch_losses = []
                val_epoch_losses = []

                self.model.train()
                for batch in self.train_dl:
                    if batch is None:
                        continue
                    train_text, train_text_pad, train_midi_in, train_midi_tar = [x.to(device) for x in batch]
                    loss = train_step(self.model, self.optimizer, self.scheduler, train_text, train_text_pad, train_midi_in, train_midi_tar)
                    train_epoch_losses.append(loss)

                self.model.eval()
                with torch.no_grad():
                    for batch in self.val_dl:
                        if batch is None:
                            continue
                        val_text, val_text_pad, val_midi_in, val_midi_tar = [x.to(device) for x in batch]
                        loss = val_step(self.model, val_text, val_text_pad, val_midi_in, val_midi_tar)
                        val_epoch_losses.append(loss)

                # mean losses for the epoch
                if len(train_epoch_losses) == 0 or len(val_epoch_losses) == 0:
                    raise RuntimeError("No valid batches were produced. Check dataset preprocessing and sequence lengths.")

                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

                # store complete history of losses in member lists and relative history for this session in output lists
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)

                print(f"Epoch: {epoch + 1} - Time: {round(time.time() - start, 2)} seconds - "
                    f"Train Loss: {train_losses[-1]} - Val Loss: {val_losses[-1]}")
                start = time.time()

        except KeyboardInterrupt:
            pass

        print("Checkpointing...")
        self.save()
        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))

        return train_losses, val_losses