# ——————————————————————————————————————————————————————————————
# Imports
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——————————————————————————————————————————————————————————————
# Multi-Head Attention class
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Block.
    Computes scaled dot-product attention across num_heads heads.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        bias=True
    ) -> None:
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): Number of attention heads
            bias (bool, optional): If False, Linear layers will not learn an additive bias
        """
        super(MultiHeadAttention, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, not {d_model} and {num_heads}")

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.batch_first = True

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=True)

    @staticmethod
    def split_heads(
        x, 
        num_heads, 
        depth=None
    ):
        """
        Splits the last dimension of x into (num_heads, depth) and transposes.

        Args:
            x: Input tensor of shape (..., L, d_model)
            num_heads (int): Number of attention heads
            depth (int, optional): Depth per head; inferred from x if not provided

        Returns:
            Tensor of shape (..., num_heads, L, depth)
        """
        if depth is None:
            if x.shape[-1] % num_heads != 0:
                raise ValueError(f"Last dim must be divisible by num_heads, not {x.shape[-1]} and {num_heads}")
            depth = x.shape[-1] // num_heads

        x = x.view(*x.shape[:-1], num_heads, depth)  # (..., L, num_heads, depth)
        return x.transpose(-3, -2)                    # (..., num_heads, L, depth)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask=None):
        """
        Computes scaled dot-product attention.

        Args:
            q: Queries  (..., num_heads, seq_len_q, depth)
            k: Keys     (..., num_heads, seq_len_k, depth)
            v: Values   (..., num_heads, seq_len_v, depth)
            mask: Optional boolean mask; positions with True are masked out

        Returns:
            Attention output of shape (..., num_heads, seq_len_q, depth)
        """
        scale = q.shape[-1] ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (..., num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        weights = torch.softmax(scores, dim=-1)                 # (..., num_heads, seq_len_q, seq_len_k)
        return torch.matmul(weights, v)                         # (..., num_heads, seq_len_q, depth)

    def forward(self, q, k, v, mask=None):
        """
        Computes Multi-Head Attention on input tensors Q, K, V.

        Args:
            q: Queries of shape (batch_size, seq_len_q, d_model)
            k: Keys    of shape (batch_size, seq_len_k, d_model)
            v: Values  of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask; positions with 1 are masked out

        Returns:
            Attention output of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = q.shape[0]

        # Project inputs
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        # Split into heads
        q = self.split_heads(q, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, self.num_heads, self.depth)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        attn = self.scaled_dot_product_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth)

        # Merge heads and project output
        attn = attn.transpose(-3, -2)                            # (batch_size, seq_len_q, num_heads, depth)
        attn = attn.reshape(batch_size, -1, self.d_model)        # (batch_size, seq_len_q, d_model)

        return self.wo(attn)                                     # (batch_size, seq_len_q, d_model)
    
    
class PointwiseFFN(nn.Module):
    """
    Fully-connected Feedforward layer that follows the MHA block in each Transformer layer, which is simply a 2 layer
    Dense network with a ReLU in between.
    The first layer expands the dimension from d_model to d_ff, and the second layer projects it back to d_model.
    
    **Description:**
    
        This block is applied independently to each position in the sequence, hence "pointwise". It allows the model to learn
        complex transformations of the features at each position after the attention mechanism has aggregated information across the sequence.
    
    **Args:**
        d_model (int): The input and output dimension of the FFN, which is the same as the Transformer hidden dimension size.
        d_ff (int): The intermediate dimension of the FFN, which is typically larger than d_model to allow for more expressive transformations.
        bias (bool, optional): If set to False, the Linear layers will not learn an additive bias term. Default is True.\
    
    **Example Usage:**
    
    .. code-block:: python
    
        from tuneml.models.transformer.Transformer import PointwiseFFN
        
        d_model = 512
        d_ff = 2048
        ffn = PointwiseFFN(d_model, d_ff)
        x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
        output = ffn(x)  # (batch_size, seq_len, d_model)
    
    """
    def __init__(
        self, 
        d_model, 
        d_ff, 
        bias=True
    ):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            d_ff (int): intermediate dimension of FFN blocks
            bias (bool, optional): if set to False, all Linear layers in the PointWiseFFN block will not learn
                                a bias term; default: True
        """
        super(PointwiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.main = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=bias)
        )
    
    def forward(self, x):
        return self.main(x)
