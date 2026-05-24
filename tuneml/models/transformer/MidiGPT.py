# ——————————————————————————————————————————————————————————————
# Imports
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tuneml.models.transformer.Transformer import MultiHeadAttention, PointwiseFFN
from tuneml.core.utils import generate_causal_mask

# ——————————————————————————————————————————————————————————————
# Decoder Layer class
class DecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        d_ff,
        bias=True,
        dropout=0.1,
        layernorm_eps=1e-6
    ):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, bias)
        self.ffn = PointwiseFFN(d_model, d_ff, bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, 
        tgt,
        tgt_mask=None,
        tgt_key_padding_mask=None
    ):
        # Build a broadcastable self-attention mask: (B, 1, T, T)
        self_attn_mask = tgt_mask # (1, T, T) or (T, T)
        if self_attn_mask is not None and self_attn_mask.dim() == 2:
            self_attn_mask = self_attn_mask.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
        if tgt_key_padding_mask is not None:
            tgt_pad = tgt_key_padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, T)
            # Combine masks: if self_attn_mask is None, use tgt_pad; otherwise, combine with OR
            self_attn_mask = tgt_pad if self_attn_mask is None else (self_attn_mask | tgt_pad)

        # Self-Attention
        x = self.norm1(tgt) # Pre-LN
        attn_output = self.self_attn(x, x, x, mask=self_attn_mask) # (B, T, D)
        attn_output = self.dropout1(attn_output) # Apply dropout after attention
        z = tgt + attn_output # residual connection

        # Pointwise FFN
        x = self.norm2(z) # Pre-LN
        ffn_output = self.ffn(x) # (B, T, D)
        ffn_output = self.dropout2(ffn_output) # Apply dropout after FFN
        z = z + ffn_output # residual connection
        
        return z

# ——————————————————————————————————————————————————————————————
# MidiGPT class
class MidiGPT(nn.Module):
    """
    An autoregressive transformer model for generating MIDI music. 
    
    **Description:**
    
        The MidiGPT model consists of an embedding layer for MIDI tokens, absolute positional encodings, and a stack of decoder layers. 
        Each decoder layer includes masked multi-head self-attention and a pointwise feedforward network, with residual connections and layer normalization. 
        The model is designed to predict the next token in a sequence given all previous tokens, making it suitable for music generation tasks.
    
    **Args:**
        d_model (int): the dimensionality of the token embeddings and model layers.
        num_layers (int): the number of decoder layers in the transformer.
        num_heads (int): the number of attention heads in each multi-head attention layer.
        d_ff (int): the dimensionality of the feedforward network's hidden layer.
        max_midi_len (int): the maximum length of MIDI token sequences that the model can process.
        midi_vocab_size (int): the size of the MIDI token vocabulary.
        bias (bool): whether to include bias terms in linear layers.
        dropout (float): the dropout rate to apply after attention and feedforward layers.
        layernorm_eps (float): the epsilon value for layer normalization to prevent division by zero.
        pad_token_id (int, optional): the token ID used for padding sequences. Defaults to 0.
        
    **Example Usage:**
    
    .. code-block:: python
        from tuneml.models.transformer.MidiGPT import MidiGPT

        # Initialize the model
        model = MidiGPT(
            d_model=512,
            num_layers=6,
            num_heads=8,
            d_ff=2048,
            max_midi_len=1024,
            midi_vocab_size=128,  # Example vocab size
            bias=True,
            dropout=0.1,
            layernorm_eps=1e-6
        )

        # Example input: batch of MIDI token sequences (B, T)
        midi_tokens = torch.randint(0, 128, (16, 512))  # (batch_size=16, seq_len=512)

        # Forward pass
        logits = model(midi_tokens)  # (B, T, midi_vocab_size)
    
    """
    def __init__(
        self,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_midi_len,
        midi_vocab_size,
        bias,
        dropout,
        layernorm_eps,
        pad_token_id=0
    ):
        super(MidiGPT, self).__init__()

        self.d_model = d_model                  # model dimension
        self.max_midi_len = max_midi_len        # maximum sequence length the model can handle
        self.midi_vocab_size = midi_vocab_size  # size of the MIDI token vocabulary
        self.pad_token_id = pad_token_id        # Embedding layer for MIDI tokens

        self.midi_embedding = nn.Embedding(
            num_embeddings=midi_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )
        self.midi_pos_encoding = self.abs_positional_encoding(max_midi_len, d_model)

        self.input_dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                bias=bias,
                dropout=dropout,
                layernorm_eps=layernorm_eps
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.output_bias = nn.Parameter(torch.zeros(midi_vocab_size))


    def inverse_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project decoder hidden states back to MIDI vocabulary logits using the
        transposed token embedding matrix.
        """
        return F.linear(hidden_states, self.midi_embedding.weight, self.output_bias)


    def abs_positional_encoding(
        self, 
        max_position, 
        d_model
    ):
        """
        Computes sinusoidal absolute positional encodings.
        """
        positions = torch.arange(max_position).float()
        k = torch.arange(d_model).float()
        coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
        angles = positions.view(-1, 1) @ coeffs.view(1, -1)

        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        return angles.unsqueeze(0)


    def forward(
        self, 
        midi_tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] =None, 
        tgt_key_padding_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Takes a batch of MIDI token sequences and returns the logits for the next token prediction at each position.
        
        :param midi_tokens: a tensor of shape (B, T) containing token IDs for each sequence in the batch.
        :type midi_tokens: torch.Tensor
        
        :param tgt_mask: an optional tensor of shape (T, T) representing the causal mask to prevent attention to future tokens.
        :type tgt_mask: torch.Tensor, optional
        
        :param tgt_key_padding_mask: an optional tensor of shape (B, T) indicating which tokens are padding and should be ignored in attention.
        :type tgt_key_padding_mask: torch.Tensor, optional
        
        :returns: a tensor of shape (B, T, midi_vocab_size) containing the logits for the next token prediction at each position in the sequence.
        :rtype: torch.Tensor
        """
        seq_len = midi_tokens.shape[1]
        if seq_len > self.max_midi_len:
            raise ValueError(
                f"Input length {seq_len} exceeds max_midi_len {self.max_midi_len}."
            )

        if tgt_mask is None:
            tgt_mask = generate_causal_mask(seq_len, device=midi_tokens.device)

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = midi_tokens.eq(self.pad_token_id)

        x = self.midi_embedding(midi_tokens) * sqrt(self.d_model)
        pos = self.midi_pos_encoding[:, :seq_len, :].to(x.device)
        x = self.input_dropout(x + pos)

        for layer in self.decoder_layers:
            x = layer(
                tgt=x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        x = self.norm(x)
        return self.inverse_embedding(x)


    @torch.no_grad()
    def generate(
        self,
        midi_tokens,
        max_len=None,
        temperature=1.0,
        top_k=None
    ) -> torch.Tensor:
        """
        Continues each token sequence autoregressively until max length.

        Args:
            midi_tokens (torch.Tensor): initial prompt tokens, shape (B, T0)
            max_len (int, optional): target generated length. Defaults to self.max_midi_len.
            temperature (float): softmax temperature for sampling.
            top_k (int, optional): if set, sample only from top-k logits.

        Returns:
            torch.Tensor: generated tokens, shape (B, T_out)
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if max_len is None:
            max_len = self.max_midi_len
        max_len = min(max_len, self.max_midi_len)

        if midi_tokens.dim() != 2:
            raise ValueError("midi_tokens must have shape (batch_size, seq_len)")

        out_tokens = midi_tokens.clone()

        while out_tokens.shape[1] < max_len:
            input_tokens = out_tokens[:, -self.max_midi_len:]
            logits = self.forward(input_tokens) # (B, T_in, vocab_size)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.shape[-1])
                top_vals, top_idx = torch.topk(next_logits, k=k, dim=-1)
                probs = torch.softmax(top_vals, dim=-1)
                sampled_rel = torch.multinomial(probs, num_samples=1)
                next_token = top_idx.gather(-1, sampled_rel)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                print(f"Sampled next token IDs: {next_token.shape} {next_token}")

            out_tokens = torch.cat([out_tokens, next_token], dim=1)

        return out_tokens

