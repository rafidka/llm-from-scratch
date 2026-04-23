import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for Rotary Positional Embeddings (RoPE).

    RoPE encodes position by rotating pairs of dimensions in Q and K.
    For each pair (2i, 2i+1) at position m, the rotation angle is m * theta_i,
    where theta_i = 1 / (base ** (2i / head_dim)).

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base for computing theta_i (default 10000, same as original Transformer).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        # TODO: Precompute the frequency theta_i for each pair of dimensions.
        # theta_i = 1 / (base ** (2i / head_dim)) for i = 0, 1, ..., head_dim // 2 - 1
        # Then for each position m in [0, max_seq_len), compute the angle m * theta_i.
        # Store cos and sin tensors as buffers (not parameters — they're fixed).
        #
        # Hints:
        # - Use torch.arange and torch.outer to build the angle matrix
        # - Shape should be (max_seq_len, head_dim) — one row per position,
        #   with each pair of columns sharing the same angle
        # - Register as buffers so they move with .to(device) but aren't parameters
        half_head_dim = head_dim // 2
        i = torch.arange(0, half_head_dim)
        theta_i = 1 / (base ** (2 * i / head_dim))
        m = torch.arange(0, max_seq_len)
        angles = torch.outer(m, theta_i)  # [max_seq_len, half_head_dim]
        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
        self.cos = nn.Buffer(cos)
        self.sin = nn.Buffer(sin)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        """Return (cos, sin) tensors for positions 0..seq_len-1.

        Returns:
            cos: shape (seq_len, head_dim)
            sin: shape (seq_len, head_dim)
        """
        cos = self.cos[0:seq_len,]
        sin = self.sin[0:seq_len,]
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to an input tensor.

    For each pair of dimensions (2i, 2i+1), apply the 2D rotation:
        [cos(θ) * x_2i - sin(θ) * x_2i+1,
         sin(θ) * x_2i + cos(θ) * x_2i+1]

    Args:
        x: shape (batch, num_heads, seq_len, head_dim)
        cos: shape (seq_len, head_dim) — will be broadcast to match x
        sin: shape (seq_len, head_dim) — will be broadcast to match x

    Returns:
        Tensor of same shape as x with rotary embeddings applied.
    """
    # TODO:
    # 1. Split x into pairs: x_2i and x_2i+1 for each dimension pair
    #    (Try reshaping to (..., head_dim//2, 2) or using even/odd indexing)
    # 2. Apply the 2D rotation using the cos/sin values
    # 3. Interleave the rotated pairs back into the original dimension order
    #
    # Hint: A clean approach is to construct a "rotated" version of x where
    # each pair (x_2i, x_2i+1) becomes (-x_2i+1, x_2i), then use:
    #   result = x * cos + rotated_x * sin
    # This avoids explicit interleaving.

    *_, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    x_ = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    x_rot = torch.stack((-x_[..., 1], x_[..., 0]), dim=-1).flatten(start_dim=-2)

    return x * cos + x_rot * sin
