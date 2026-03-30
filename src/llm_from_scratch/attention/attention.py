from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    # q/k/v: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    # scores: [..., seq_len, seq_len]
    # attn_weights: [..., seq_len, seq_len]
    # output: [..., seq_len, embed_dim]

    scores = q @ k.transpose(-1, -2)
    if causal:
        seq_len = k.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=q.dtype),
            diagonal=1,
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    embed_dim = k.shape[-1]
    attn_weights = torch.softmax(scores / sqrt(embed_dim), dim=-1)

    output = attn_weights @ v

    if return_attn_weights:
        return output, attn_weights
    return output


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, causal: bool):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.causal = causal

    def forward(self, x: "Tensor") -> "Tensor":
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return scaled_dot_product_attention(q, k, v, self.causal)  # type: ignore


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool):
        super().__init__()

        # Find the head dimension
        if embed_dim % num_heads != 0:
            raise ValueError("num_heads doesn't evenly divide embed_dim")
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.causal = causal

        # Linear projections for W, K, and V.
        # Notice that, on paper, we should create num_heads linear projections, each
        # projecting from embed_dim into head_dim. However, to make the computation more
        # efficient, we just create one big matrix, and then partition the output. See
        # the forward() implementation for more info.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: "Tensor") -> "Tensor":
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, embed_dim = x.shape

        q = self.W_q(x)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # transpose into [batch, num_heads, seq_len, head_dim]

        k = self.W_k(x)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # transpose into [batch, num_heads, seq_len, head_dim]

        v = self.W_v(x)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # transpose into [batch, num_heads, seq_len, head_dim]

        attn = scaled_dot_product_attention(q, k, v, self.causal)  # type: ignore[assignment]
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)  # type: ignore[union-attr]

        return self.W_o(attn)  # output: [batch, seq_len, embed_dim]
