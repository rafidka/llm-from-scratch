from typing import TYPE_CHECKING

from torch import nn
import torch

if TYPE_CHECKING:
    from torch import Tensor


class GPTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        use_rope: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Embedding(max_seq_len, embed_dim) if not use_rope else None

    def forward(self, token_ids: "Tensor", offset: int = 0) -> "Tensor":
        _batch, seq_len = token_ids.shape
        token_emb = self.token(token_ids)

        if not self.use_rope:
            pos_ids = torch.arange(offset, offset + seq_len, device=token_ids.device)
            pos_emb = self.positional(pos_ids)  # type: ignore
            return token_emb + pos_emb
        else:
            return token_emb
