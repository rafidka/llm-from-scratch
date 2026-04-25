from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from llm_from_scratch.model.lora import LoRALayer
from llm_from_scratch.model.rope import RotaryEmbedding, apply_rotary_emb


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    attn_mask: Tensor | None = None,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Implementation of the scaled dot product as in the "Attention is All You Need" paper.

    Args:
        q, k v: Well known. Their shapes should be either [seq_len, embed_dim] or
            [<one ore more batch dimensions>, seq_len, embed_dim].
        attn_mask: A mask to apply to the attention scores. It should be of shape
            [seq_len] or [<one ore more batch dimensions>, seq_len]. Notice that if
            there are batch dimensions, they should be the same as the batch dimensions
            of q, k, v.
        causal: Whether to calculate bi-directional attention, or just attend to
            previous tokens.
        return_attn_weights: Whether to return the attention weights as well, or just
            the output (after applying the attention on v.)
    """
    # q/k/v: [<zero or more batch dimensions>, seq_len, embed_dim]

    # scores: [<zero or more batch dimensions>, seq_len, seq_len]
    scores = q @ k.transpose(-1, -2)
    if causal:
        # If the attention is causal, each token only attends to previous tokens.
        # To achieve this, we create a mask to fill out the relevant attention scores
        # to -inf.
        q_seq_len, k_seq_len = q.shape[-2], k.shape[-2]
        mask = torch.triu(
            torch.ones(k_seq_len, k_seq_len, device=q.device, dtype=q.dtype),
            diagonal=1,
        )[-q_seq_len:, :].bool()
        scores = scores.masked_fill(mask, float("-inf"))
    if attn_mask is not None:
        # attention mask is of shape [<zero or more batch dimensions>, seq_len].
        # We want to add a dimension of 1 between the batch dimensions and the seq_len
        # dimension, so it becomes [<zero or more batch dimensions>, 1, seq_len].
        # This way it can be broadcasted to the scores shape as follows:
        # Attention mask shape: [<zero or more batch dimensions>,    1,    seq_len]
        #                                                            ↓
        # Scores shape:         [<zero or more batch dimensions>, seq_len, seq_len]
        scores = scores.masked_fill(
            attn_mask.unsqueeze(-2).bool().logical_not(),
            float("-inf"),
        )

    # attn_weights: [..., seq_len, seq_len]
    embed_dim = k.shape[-1]
    attn_weights = torch.softmax(scores / sqrt(embed_dim), dim=-1)

    # output: [..., seq_len, embed_dim]
    output = attn_weights @ v

    if return_attn_weights:
        return output, attn_weights
    return output


class SingleHeadAttention(nn.Module):
    """
    A wrapper layer around the scaled_dot_product_attention function (single head.)
    """

    def __init__(self, embed_dim: int, causal: bool):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.causal = causal

    def forward(self, x: "Tensor", attn_mask: "Tensor | None" = None) -> "Tensor":
        """
        Forward pass for the single head attention.

        Args:
            x: Input tensor with shape [seq_len, embed_dim] or [batch, seq_len,
                embed_dim].
            attn_mask: Attention mask with shape [seq_len] or [batch, seq_len].

        Returns:
            The output tensor with shape [seq_len, embed_dim] or [batch, seq_len,
                embed_dim].
        """
        q = self.W_q(x)  # shape stays the same.
        k = self.W_k(x)  # shape stays the same.
        v = self.W_v(x)  # shape stays the same.

        # Output shape is [seq_len, embed_dim] or [batch, seq_len, embed_dim].
        return scaled_dot_product_attention(q, k, v, self.causal, attn_mask)  # type: ignore


@dataclass
class MultiHeadAttentionOutput:
    output: Tensor
    kv_cache: tuple[Tensor, Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool,
        use_rope: bool = False,
        max_seq_len: int = 1024,
        num_kv_heads: int | None = None,
    ):
        super().__init__()

        # Find the head dimension
        if embed_dim % num_heads != 0:
            raise ValueError("num_heads doesn't evenly divide embed_dim")
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads should be a multiple of num_kv_heads")
        self.group_size = num_heads // self.num_kv_heads
        self.causal = causal
        self.use_rope = use_rope

        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, max_seq_len) if use_rope else None
        )

        # Linear projections for W, K, and V.
        # Notice that, on paper, we should create num_heads linear projections, each
        # projecting from embed_dim into head_dim. However, to make the computation more
        # efficient, we just create one big matrix, and then partition the output. See
        # the forward() implementation for more info.
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # LoRA stuff.
        self.W_q_lora: LoRALayer | None = None
        self.W_k_lora: LoRALayer | None = None
        self.W_v_lora: LoRALayer | None = None
        self.W_o_lora: LoRALayer | None = None

    def lorafy(self, rank: int, alpha: float, sigma: float = 0.02):
        self.W_q_lora = LoRALayer(self.W_q, rank, alpha, sigma)
        self.W_k_lora = LoRALayer(self.W_k, rank, alpha, sigma)
        self.W_v_lora = LoRALayer(self.W_v, rank, alpha, sigma)
        self.W_o_lora = LoRALayer(self.W_o, rank, alpha, sigma)

    def forward(
        self,
        x: "Tensor",
        attn_mask: "Tensor | None" = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> MultiHeadAttentionOutput:
        """
        Forward pass for the multi head attention.

        Args:
            x: Input tensor with shape [batch, seq_len, embed_dim].
            attn_mask: Attention mask with shape [batch, seq_len].
            kv_cache: KV cache returned from previous calls to forward(). It should be
                of the shape [batch, num_kv_heads, seq_len, embed_dim].

        Returns:
            The output tensor with shape [batch, seq_len, embed_dim].
        """
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, embed_dim = x.shape

        # LoRA handling
        W_q = self.W_q_lora if self.W_q_lora else self.W_q
        W_k = self.W_k_lora if self.W_k_lora else self.W_k
        W_v = self.W_v_lora if self.W_v_lora else self.W_v
        W_o = self.W_o_lora if self.W_o_lora else self.W_o

        # Q/K/V projections
        q = W_q(x)
        k = W_k(x)
        v = W_v(x)

        # Reshape into [batch, seq_len, num_heads, head_dim]
        # Or into [batch, seq_len, num_kv_heads, head_dim] for k/v in case of GQA
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose into [batch, num_heads, seq_len, head_dim].
        # Or into [batch, num_kv_heads, new_tokens, head_dim] for k/v in case of GQA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # transpose into [batch, num_kv_heads, seq_len, head_dim]

        if kv_cache:
            _, _, cache_seq_len, _ = kv_cache[0].shape
        else:
            cache_seq_len = 0
        total_seq_len = cache_seq_len + seq_len

        if self.use_rope and self.rotary_emb:
            cos, sin = self.rotary_emb(seq_len, cache_seq_len)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if kv_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)

        k_cache_ret, v_cache_ret = k, v  # save this to return to user before expansion

        # Expand k and v into [batch, num_heads, seq_len, head_dim]
        # The code below is a trick to achieve the same effect as repeat_interleaved
        # without copying memory.
        k = (
            k.view(batch, self.num_kv_heads, 1, total_seq_len, self.head_dim)
            .expand(-1, -1, self.group_size, -1, -1)
            .reshape(batch, self.num_heads, total_seq_len, self.head_dim)
        )
        v = (
            v.view(batch, self.num_kv_heads, 1, total_seq_len, self.head_dim)
            .expand(-1, -1, self.group_size, -1, -1)
            .reshape(batch, self.num_heads, total_seq_len, self.head_dim)
        )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-2)

        attn = scaled_dot_product_attention(q, k, v, self.causal, attn_mask)  # type: ignore[assignment]
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)  # type: ignore[union-attr]

        # Output shape: [batch, seq_len, embed_dim].
        return MultiHeadAttentionOutput(W_o(attn), (k_cache_ret, v_cache_ret))
