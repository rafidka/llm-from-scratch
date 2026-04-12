from torch import Tensor, nn

from llm_from_scratch.model.base import GPT


class GPTForClassification(GPT):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__(
            vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout
        )
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, token_ids: "Tensor") -> "Tensor":
        out = super().forward(token_ids)
        if out.ndim == 3:
            logits = self.cls_head(out[:, -1, :])  # shape [batch, embed_dim]
        elif out.ndim == 2:
            logits = self.cls_head(out[:, :])  # shape [batch, embed_dim]

        return logits  # shape [batch, num_classes]
