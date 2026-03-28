import torch
from llm_from_scratch.attention.qkv import (
    SingleHeadAttention,
    CausalSelfAttention,
    self_causal_attention,
)

# Test dimensions
batch, seq_len, embed_dim = 2, 4, 8

# Test SingleHeadAttention
x = torch.randn(batch, seq_len, embed_dim)
attn = SingleHeadAttention(embed_dim)
out = attn(x)
print(f"SingleHeadAttention: {x.shape} -> {out.shape}")

# Test CausalSelfAttention
causal_attn = CausalSelfAttention(embed_dim)
out_causal = causal_attn(x)
print(f"CausalSelfAttention: {x.shape} -> {out.shape}")

# Test causal attention masking working correctly.
q = torch.rand(1, 5, 3)
k = torch.rand(1, 5, 3)
v = torch.rand(1, 5, 3)
self_causal_attention(q, k, v)
