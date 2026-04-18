import torch

from llm_from_scratch.model.base import GPT

vocab_size = 1000
embed_dim = 64
num_heads = 8
num_layers = 4
max_seq_len = 128
dropout = 0.1
batch_size = 4

gpt = GPT(
    vocab_size,
    embed_dim,
    num_heads,
    num_layers,
    max_seq_len,
    dropout,
)
token_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
attn_mask = torch.ones(batch_size, max_seq_len, dtype=torch.int)
out = gpt(token_ids, attn_mask)
print(out.shape)
