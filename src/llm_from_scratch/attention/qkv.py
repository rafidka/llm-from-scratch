from typing import TYPE_CHECKING

import torch
from torch import nn

from math import sqrt


if TYPE_CHECKING:
    from torch import Tensor


def self_attention(q: "Tensor", k: "Tensor", v: "Tensor"):
    # q/k/v: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    # scores: [..., seq_len, seq_len]
    # attn_weights: [..., seq_len, seq_len]
    # output: [..., seq_len, embed_dim]
    scores = q @ k.transpose(-1, -2)

    embed_dim = k.shape[-1]

    attn_weights = torch.softmax(scores / sqrt(embed_dim), dim=-1)

    output = attn_weights @ v

    return output


def self_causal_attention(q: "Tensor", k: "Tensor", v: "Tensor"):
    # q/k/v: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    # scores: [..., seq_len, seq_len]
    # attn_weights: [..., seq_len, seq_len]
    # output: [..., seq_len, embed_dim]

    embed_dim = k.shape[-1]
    seq_len = k.shape[-2]

    scores = q @ k.transpose(-1, -2)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=q.dtype),
        diagonal=1,
    ).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores / sqrt(embed_dim), dim=-1)

    output = attn_weights @ v

    return output


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: "Tensor") -> "Tensor":
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return self_attention(q, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: "Tensor") -> "Tensor":
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return self_causal_attention(q, k, v)
