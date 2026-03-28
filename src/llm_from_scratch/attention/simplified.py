from typing import TYPE_CHECKING

from torch import nn, softmax

if TYPE_CHECKING:
    from torch import Tensor


def simplified_self_attention(embeddings: "Tensor"):
    # embeddings: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    # scores: [..., seq_len, seq_len]
    # attn_weights: [..., seq_len, seq_len]
    # output: [..., seq_len, embed_dim]
    scores = embeddings @ embeddings.transpose(-1, -2)

    attn_weights = softmax(scores, dim=-1)

    output = attn_weights @ embeddings

    return output


class SimplifiedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: "Tensor") -> "Tensor":
        # x: [batch, seq_len, embed_dim]
        return simplified_self_attention(x)
