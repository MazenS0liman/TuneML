# ——————————————————————————————————————————————————————————————
# Imports
from math import sqrt

import torch
import torch.nn as nn

from tuneml.tokenizers.TextTokenizer import Flan5Tokenizer
from tuneml.models.transformer.Transformer import MultiHeadAttention, PointwiseFFN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————————————————————————————————————————————————————
# Decoder Layer class
class MidiDecoderLayer(nn.Module):
    """
    Decoder layer with:
        1. Masked self-attention   (attends to previous MIDI tokens)
        2. Cross-attention         (attends to encoded text memory)
        3. Pointwise FFN
    All using Pre-LN (LayerNorm before sublayer).
    """
    def __init__(
        self, 
        d_model, 
        num_heads, 
        d_ff, 
        bias=True, 
        dropout=0.1, 
        layernorm_eps=1e-6
    ):
        super().__init__()

        self.self_attn  = MultiHeadAttention(d_model, num_heads, bias)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, bias)
        self.ffn        = PointwiseFFN(d_model, d_ff, bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self, 
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False
    ):  
        # Build a broadcastable self-attention mask: (B, 1, T, T)
        self_attn_mask = tgt_mask
        if self_attn_mask is not None and self_attn_mask.dim() == 2:
            self_attn_mask = self_attn_mask.unsqueeze(0).unsqueeze(0)
        if tgt_key_padding_mask is not None:
            tgt_pad = tgt_key_padding_mask.unsqueeze(1).unsqueeze(1)
            self_attn_mask = tgt_pad if self_attn_mask is None else (self_attn_mask | tgt_pad)

        # Build a broadcastable cross-attention mask: (B, 1, T, S)
        cross_attn_mask = memory_mask
        if cross_attn_mask is not None and cross_attn_mask.dim() == 2:
            cross_attn_mask = cross_attn_mask.unsqueeze(0).unsqueeze(0)
        if memory_key_padding_mask is not None:
            mem_pad = memory_key_padding_mask.unsqueeze(1).unsqueeze(1)
            cross_attn_mask = mem_pad if cross_attn_mask is None else (cross_attn_mask | mem_pad)

        # 1. masked self-attention over MIDI tokens
        x = self.norm1(tgt)
        x = tgt + self.drop1(self.self_attn(x, x, x, mask=self_attn_mask)) # (batch_size, seq_len, d_model)

        # 2. cross-attention: MIDI queries attend to text
        y = self.norm2(x)
        y = x + self.drop2(self.cross_attn(y, memory, memory, mask=cross_attn_mask)) # (batch_size, seq_len, d_model)

        # 3. FFN
        z = self.norm3(y)
        z = y + self.drop3(self.ffn(z))

        return z # (batch_size, seq_len, d_model)
    
# ——————————————————————————————————————————————————————————————
# Midi Transformer class
class MidiTransformer(nn.Module):
    """
    Encoder-Decoder Transformer that conditions MIDI generation on text input.
    
    Architecture:
        Encoder: Embeds text tokens -> stack of EncoderLayers
        Decoder: Embeds MIDI tokens -> stack of DecoderLayers cross-attending to encoder output
        Output:  Linear projection -> MIDI vocabulary logits
    """
    def __init__(
        self,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_midi_len,
        max_text_len,
        midi_vocab_size,
        text_vocab_size,
        bias,
        dropout,
        layernorm_eps,
        pad_token_id=0
    ):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_layers (int): number of encoder and decoder layers
            num_heads (int): number of attention heads
            d_ff (int): intermediate dimension of FFN blocks
            max_midi_len (int): maximum length of MIDI token sequences
            max_text_len (int): maximum length of text token sequences
            midi_vocab_size (int): number of tokens in the MIDI vocabulary
            text_vocab_size (int): number of tokens in the text vocabulary (from tokenizer)
            bias (bool): whether Linear layers learn a bias term
            dropout (float): dropout rate
            layernorm_eps (float): epsilon for LayerNorm
            pad_token_id (int): token id used for padding (masked in encoder)
        """
        super(MidiTransformer, self).__init__()

        self.d_model = d_model
        self.midi_vocab_size = midi_vocab_size
        self.text_vocab_size = text_vocab_size
        self.pad_token_id = pad_token_id
        
        # -- Text Embedding --
        self.text_embedding = nn.Embedding(self.text_vocab_size, d_model, padding_idx=pad_token_id)
        self.text_pos_encoding = self.abs_positional_encoding(max_text_len, d_model)
        
        # -- Midi Embedding --
        self.midi_embedding = nn.Embedding(self.midi_vocab_size, d_model)
        self.midi_pos_encoding = self.abs_positional_encoding(max_midi_len, d_model)

        # -- Text Encoder --
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layernorm_eps)
        )

        # -- Midi Decoder --
        self.decoder = nn.TransformerDecoder(
            MidiDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                bias=bias,
                dropout=dropout,
                layernorm_eps=layernorm_eps
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layernorm_eps)
        )

        self.input_dropout = nn.Dropout(dropout)
        self.final = nn.Linear(d_model, self.midi_vocab_size)


    def abs_positional_encoding(self, max_position, d_model, n=3):
        """
        Since the transformer does not use recurrence or convolution, we have to deliberately give it positional
        information. Though learned relative position embeddings will be added to the model, it is possible that absolute
        position encoding will aid it in predicting next tokens.

        Args:
            max_position (int): maximum position for which to calculate positional encoding
            d_model (int): Transformer hidden dimension size
            n (int): number of dimensions to which to broadcast output

        Returns:
            sinusoidal absolute positional encoding of shape d_model for max_position positions
        """
        # set of all positions to consider
        positions = torch.arange(max_position).float().to(device)

        # get angles to input to sinusoid functions
        k = torch.arange(d_model).float().to(device)
        coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
        angles = positions.view(-1, 1) @ coeffs.view(1, -1)

        # apply sin to the even indices of angles along the last axis
        angles[:, 0::2] = torch.sin(angles[:, 0::2])

        # apply cos to the odd indices of angles along the last axis
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        return angles.view(*[1 for _ in range(n-2)], max_position, d_model)


    def tokenize_text(
        self,
        text,
        device="cpu"
    ):
        """
        Tokenizes raw text string(s) using the T5 tokenizer and prepares
        tensors and padding mask for the encoder.

        Args:
            text (str or list of str): raw text input(s) to condition generation on
            device: device to move tensors to

        Returns:
            text_tokens (torch.Tensor): token IDs of shape (batch_size, text_seq_len)
            text_padding_mask (torch.Tensor): boolean padding mask of shape (batch_size, text_seq_len)
                                              with True at padding positions.
        """
        text_tokenizer = Flan5Tokenizer()
        tokens = text_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        text_tokens = tokens["input_ids"].to(device) # (batch_size, text_seq_len)
        text_padding_mask = tokens["attention_mask"].to(device) == 0

        return text_tokens, text_padding_mask


    def generate_causal_mask(seq_len, device=None):
        """
        Creates a causal mask with True values above the diagonal.
        
        Args:
            seq_len (int): length of the sequence to generate a mask for
            device: device to create the mask on
            
        Returns:
            A boolean tensor of shape (seq_len, seq_len) where True values indicate masked positions.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), 
            diagonal=1
        ).bool()


    def encode_text(
        self, 
        text_tokens, 
        text_padding_mask=None
    ):
        """
        Encodes text tokens into a context memory tensor.
    
        Args:
            text_tokens (torch.Tensor or str or list of str): either pre-tokenized IDs
                        of shape (batch_size, text_seq_len), or raw text string(s) which
                        will be tokenized automatically via text_to_tokens
            text_padding_mask (torch.Tensor, optional): boolean mask of shape (batch_size, text_seq_len)
                              with True at padding positions. Inferred automatically if text_tokens is a string.
    
        Returns:
            memory (torch.Tensor): encoded text of shape (batch_size, text_seq_len, d_model)
        """
        # auto-tokenize if raw text is passed in
        if isinstance(text_tokens, (str, list)):
            device = next(self.parameters()).device
            text_tokens, text_padding_mask = self.tokenize_text(text_tokens, device=device)
    
        x = self.text_embedding(text_tokens) * sqrt(self.d_model)
        if self.text_pos_encoding is not None:
            x += self.text_pos_encoding[:, :x.shape[1], :]
        x = self.input_dropout(x)
        return self.encoder(x, src_key_padding_mask=text_padding_mask)


    def forward(
        self, 
        text_tokens, 
        midi_tokens, 
        tgt_mask=None, 
        text_padding_mask=None
    ):
        """
        Full forward pass: encode text, decode MIDI tokens conditioned on text memory.
    
        Args:
            text_tokens (torch.Tensor or str or list of str): tokenized text of shape
                        (batch_size, text_seq_len), or raw text string(s) to auto-tokenize
            midi_tokens (torch.Tensor): MIDI token sequence of shape (batch_size, midi_seq_len)
            tgt_mask (torch.Tensor, optional): causal mask of shape (midi_seq_len, midi_seq_len)
                      with 1's at positions to mask. Use generate_causal_mask()
            text_padding_mask (torch.Tensor, optional): boolean padding mask for text tokens,
                              shape (batch_size, text_seq_len). Inferred if text_tokens is a string.
    
        Returns:
            logits (torch.Tensor): MIDI token logits of shape (batch_size, midi_seq_len, midi_vocab_size)
        """
        # encode text into memory — handles raw strings or pre-tokenized tensors
        if isinstance(text_tokens, (str, list)):
            device = midi_tokens.device
            text_tokens, text_padding_mask = self.tokenize_text(text_tokens, device=device)
    
        text_encoding = self.encode_text(text_tokens, text_padding_mask)  # (batch_size, text_seq_len, d_model)
    
        # embed MIDI tokens
        midi_embedding = self.midi_embedding(midi_tokens) * sqrt(self.d_model)  # (batch_size, midi_seq_len, d_model)
        if self.midi_pos_encoding is not None:
            midi_embedding += self.midi_pos_encoding[:, :midi_embedding.shape[1], :]
        midi_embedding = self.input_dropout(midi_embedding)

        # decode MIDI conditioned on text memory
        out = self.decoder(
            tgt=midi_embedding,
            memory=text_encoding,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=text_padding_mask
        )  # (batch_size, midi_seq_len, d_model)
    
        return self.final(out)  # (batch_size, midi_seq_len, midi_vocab_size)

    @torch.no_grad()
    def generate(
        self, 
        text_tokens, 
        start_token_id, 
        max_len=512,
        temperature=1.0, 
        top_k=None, 
        text_padding_mask=None, 
        device="cpu"
    ):
        """
        Autoregressively generates a MIDI token sequence conditioned on text.
    
        Args:
            text_tokens (torch.Tensor or str or list of str): tokenized text of shape
                        (batch_size, text_seq_len), or raw text string(s) to auto-tokenize
            start_token_id (int): <start> token id to seed the decoder
            max_len (int): maximum number of MIDI tokens to generate
            temperature (float): softmax temperature; lower = more deterministic
            top_k (int, optional): if set, restricts sampling to the top-k logits
            text_padding_mask (torch.Tensor, optional): padding mask for text tokens.
                              Inferred automatically if text_tokens is a string.
            device: device to run generation on
    
        Returns:
            generated (torch.Tensor): generated MIDI token ids of shape (batch_size, seq_len)
        """
        self.eval()
    
        # auto-tokenize if raw text is passed in
        if isinstance(text_tokens, (str, list)):
            text_tokens, text_padding_mask = self.tokenize_text(text_tokens, device=device)
    
        batch_size = text_tokens.shape[0]
    
        # encode text once — reused at every decoding step
        memory = self.encode_text(text_tokens, text_padding_mask)
    
        # seed the decoder with the <start> token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
    
        for _ in range(max_len - 1):
            seq_len = generated.shape[1]
            causal_mask = self.generate_causal_mask(seq_len, device=device)
    
            tgt = self.midi_embedding(generated) * sqrt(self.d_model)
            if self.midi_pos_encoding is not None:
                tgt += self.midi_pos_encoding[:, :seq_len, :]
    
            out = self.decoder(tgt, memory, tgt_mask=causal_mask,
                               memory_key_padding_mask=text_padding_mask)
    
            logits = self.final(out[:, -1, :]) / temperature  # (batch_size, midi_vocab_size)
    
            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')
    
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    
        return generated