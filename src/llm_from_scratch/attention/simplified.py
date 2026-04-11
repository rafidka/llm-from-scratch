from typing import TYPE_CHECKING

from torch import nn, softmax

if TYPE_CHECKING:
    from torch import Tensor


def simplified_self_attention(embeddings: "Tensor"):
    """
    Simplified dot product attention with no trainable parameters for learning purposes.

    Args:
        embeddings: The embeddings to calculate the attention on. It should be of
            shape [batch, seq_len, embed_dim] or [seq_len, embed_dim]. The attention
            will be calculated across the seq_len dimension.

    Returns:
        The output is of shape [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    """

    # embeddings: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
    # scores: [..., seq_len, seq_len]
    scores = embeddings @ embeddings.transpose(-1, -2)

    # attn_weights: [..., seq_len, seq_len]
    attn_weights = softmax(scores, dim=-1)

    # output: [..., seq_len, embed_dim]
    output = attn_weights @ embeddings

    return output


class SimplifiedSelfAttention(nn.Module):
    """
    A simple wrapper layer around the simplified_self_attention function.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: "Tensor") -> "Tensor":
        # x: [batch, seq_len, embed_dim]
        return simplified_self_attention(x)
