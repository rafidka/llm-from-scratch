from torch import Tensor, nn

from llm_from_scratch.model.base import GPT, GPTOutput


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
        use_gradient_checkpointing: bool = False,
        use_rms_norm: bool = False,
        use_rope: bool = False,
        use_swiglu: bool = False,
        num_kv_threads: int | None = None,
    ):
        super().__init__(
            vocab_size,
            embed_dim,
            num_heads,
            num_layers,
            max_seq_len,
            dropout,
            use_gradient_checkpointing,
            use_rms_norm,
            use_rope,
            use_swiglu,
            num_kv_threads,
        )
        self.cls_head = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(
        self,
        token_ids: "Tensor",
        attn_mask: "Tensor | None" = None,
        kv_caches: list[tuple["Tensor", "Tensor"]] | None = None,
    ) -> GPTOutput:
        ret = super().forward(token_ids, attn_mask, kv_caches)
        out = ret.output
        if out.ndim == 3:
            logits = self.cls_head(out[:, -1, :])  # shape [batch, embed_dim]
        elif out.ndim == 2:
            logits = self.cls_head(out[-1, :])  # shape [embed_dim]

        return GPTOutput(
            logits,  # shape [batch, num_classes],
            kv_caches=ret.kv_caches,
        )
