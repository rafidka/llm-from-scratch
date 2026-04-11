from typing import TYPE_CHECKING

import torch
from torch import nn

from llm_from_scratch.model.base import GPT

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
    ):
        super().__init__(
            vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids: "Tensor") -> "Tensor":
        out = super().forward(token_ids)
        logits = self.lm_head(out)
        return logits  # shape [batch, seq_len, vocab_size]

    def generate(
        self,
        token_ids: "Tensor",
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        # token_ids: [batch, seq_len]
        for _ in range(max_new_tokens):
            if token_ids.shape[-1] >= self.max_seq_len:
                token_ids = token_ids[:, -self.max_seq_len :]

            logits = self.forward(token_ids)
            last_logit = logits[:, -1, :]

            if temperature == 0:
                # Temperature is zero; use greedy sampling.
                next_tokens = last_logit.argmax(dim=-1, keepdim=True)
                token_ids = torch.cat((token_ids, next_tokens), dim=-1)
                continue

            # Apply temperature and softmax to find probs.
            probs = torch.softmax(last_logit / temperature, dim=-1)

            if top_k:
                # top_k specified; apply it.
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

                next_tokens = top_k_indices.gather(
                    -1,  # dim
                    torch.multinomial(top_k_probs, num_samples=1),
                )
                token_ids = torch.cat((token_ids, next_tokens), dim=-1)
            else:
                next_tokens = torch.multinomial(probs, num_samples=1)
                token_ids = torch.cat((token_ids, next_tokens), dim=-1)
        return token_ids
