import torch

from llm_from_scratch.attention.scaled_dot_product import (
    SingleHeadAttention,
    scaled_dot_product_attention,
)

# Test dimensions
batch, seq_len, embed_dim = 2, 4, 8

# Test SingleHeadAttention
x = torch.randn(batch, seq_len, embed_dim)
attn = SingleHeadAttention(embed_dim, causal=False)
out = attn(x)
print(f"SingleHeadAttention: {x.shape} -> {out.shape}")

# Test CausalSelfAttention
causal_attn = SingleHeadAttention(embed_dim, causal=True)
out_causal = causal_attn(x)
print(f"CausalSelfAttention: {x.shape} -> {out.shape}")

torch.manual_seed(42)
q = torch.rand(3, 5, 3)
k = torch.rand(3, 5, 3)
v = torch.rand(3, 5, 3)

# Test non causal attention mask.
_, attn = scaled_dot_product_attention(q, k, v, causal=False, return_attn_weights=True)
print("Non causal attention mask:")
print(attn)

# Test causal attention mask.
_, attn = scaled_dot_product_attention(q, k, v, causal=True, return_attn_weights=True)
print("Causal attention mask:")
print(attn)

# Test attention mask, without causal mask.
attn_mask = torch.tensor(
    [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ]
)
_, attn = scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    return_attn_weights=True,
)
print("Attention mask, without causal mask:")
print(attn)

# Test attention mask, with causal mask.
_, attn = scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    causal=True,
    return_attn_weights=True,
)
print("Attention mask, with causal mask:")
print(attn)
