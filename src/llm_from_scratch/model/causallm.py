from typing import TYPE_CHECKING

import torch
from torch import nn

from llm_from_scratch.model.base import GPT, GPTOutput

if TYPE_CHECKING:
    from torch import Tensor


class GPTForCausalLM(GPT):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float,
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

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying between token embeddings and lm_head
        self.lm_head.weight = self.embeddings.token.weight

    def forward(
        self,
        token_ids: "Tensor",
        attn_mask: "Tensor | None" = None,
        kv_caches: list[tuple["Tensor", "Tensor"]] | None = None,
    ) -> GPTOutput:
        ret = super().forward(token_ids, attn_mask, kv_caches)
        logits = self.lm_head(ret.output)
        return GPTOutput(
            logits,  # shape [batch, seq_len, vocab_size]
            kv_caches=ret.kv_caches,
        )

    @torch.no_grad()
    def generate(
        self,
        token_ids: "Tensor",
        max_new_tokens: int,
        attn_mask: "Tensor | None" = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> "Tensor":
        # token_ids: [batch, seq_len]
        kv_caches: list[tuple["Tensor", "Tensor"]] | None = None
        generated = token_ids

        for step in range(max_new_tokens):
            if step == 0:
                # First step: process the full prompt, populate the KV cache.
                if token_ids.shape[-1] >= self.max_seq_len:
                    token_ids = token_ids[:, -self.max_seq_len :]
                    if attn_mask is not None:
                        attn_mask = attn_mask[:, -self.max_seq_len :]
                ret = self.forward(token_ids, attn_mask)
                kv_caches = ret.kv_caches
                logits = ret.output
                last_logit = logits[:, -1, :]
            else:
                # Subsequent steps: pass only the new token with the KV cache.
                # No need for attn_mask — the cache already has full context.
                # Truncate cache if we've exceeded max_seq_len.
                if kv_caches is not None and kv_caches[0][0].shape[2] >= self.max_seq_len:
                    kv_caches = [
                        (k[:, :, -self.max_seq_len + 1 :, :], v[:, :, -self.max_seq_len + 1 :, :])
                        for k, v in kv_caches
                    ]
                ret = self.forward(next_tokens, kv_caches=kv_caches)
                kv_caches = ret.kv_caches
                logits = ret.output
                last_logit = logits[:, -1, :]

            if temperature == 0:
                next_tokens = last_logit.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_logit / temperature, dim=-1)
                if top_k:
                    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                    next_tokens = top_k_indices.gather(
                        -1,
                        torch.multinomial(top_k_probs, num_samples=1),
                    )
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_tokens), dim=-1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        if eos_token_id is not None:
            after_eos = (generated == eos_token_id).cumsum(dim=1) >= 1
            generated.masked_fill(after_eos, 0)

        return generated
