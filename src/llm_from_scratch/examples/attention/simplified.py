import torch
from llm_from_scratch.attention.simplified import SimplifiedSelfAttention

x = torch.randn(2, 4, 8)  # batch=2, seq_len=4, embed_dim=8
attn = SimplifiedSelfAttention(embed_dim=8)
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # Should match
