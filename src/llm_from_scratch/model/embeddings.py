from typing import TYPE_CHECKING

from torch import nn
import torch

if TYPE_CHECKING:
    from torch import Tensor


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, token_ids: "Tensor") -> "Tensor":
        _batch, seq_len = token_ids.shape
        token_emb = self.embedding(token_ids)

        pos_ids = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.positional(pos_ids)

        return token_emb + pos_emb
