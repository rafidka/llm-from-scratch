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
        # TODO: When use_rope is True, positional embeddings are unnecessary
        # because position information is encoded via rotary embeddings in attention.
        # Either skip creating self.positional, or create it but zero it out.
        self.positional = nn.Embedding(max_seq_len, embed_dim) if not use_rope else None

    def forward(self, token_ids: "Tensor") -> "Tensor":
        _batch, seq_len = token_ids.shape
        token_emb = self.token(token_ids)

        # TODO: If use_rope is True, return only token_emb (no positional embedding).
        # RoPE handles position encoding in the attention layer.
        if not self.use_rope:
            pos_ids = torch.arange(seq_len, device=token_ids.device)
            pos_emb = self.positional(pos_ids)  # type: ignore
            return token_emb + pos_emb
        else:
            return token_emb
