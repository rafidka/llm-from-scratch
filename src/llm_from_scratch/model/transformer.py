from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import GPT2LMHeadModel

from llm_from_scratch.attention.attention import MultiHeadAttention
from llm_from_scratch.model.embeddings import GPTEmbedding

if TYPE_CHECKING:
    from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "Tensor") -> "Tensor":
        out = x
        out = self.ff1(out)
        out = self.gelu(out)
        out = self.ff2(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, 4 * embed_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "Tensor") -> "Tensor":
        # x: [batch, seq_len, embed_dim]
        out = x
        out = out + self.dropout(self.attn(self.ln1(out)))
        out = out + self.dropout(self.ff(self.ln2(out)))
        return out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()

        # Save model info.
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.embedding = GPTEmbedding(vocab_size, embed_dim, max_seq_len)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.out_ff = nn.Linear(embed_dim, vocab_size, bias=False)

    @classmethod
    def test(cls, vocab_size: int, max_seq_len: int):
        """Create a very small GPT model for testing purposes."""
        return GPT(
            vocab_size=vocab_size,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=max_seq_len,
            dropout=0.0,
        )

    @classmethod
    def gpt2_small(cls, vocab_size: int, max_seq_len: int = 1024):
        """GPT-2 Small: 124M parameters"""
        return GPT(
            vocab_size,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )

    @classmethod
    def gpt2_medium(cls, vocab_size: int, max_seq_len: int = 1024):
        """GPT-2 Medium: 355M parameters"""
        return GPT(
            vocab_size,
            embed_dim=1024,
            num_heads=16,
            num_layers=24,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )

    # TODO Create the following methods:
    # - large()

    @classmethod
    def gpt2_large(cls, vocab_size: int, max_seq_len: int = 1024):
        """GPT-2 Large: 774M parameters"""
        return GPT(
            vocab_size,
            embed_dim=1280,
            num_heads=20,
            num_layers=36,
            max_seq_len=max_seq_len,
            dropout=0.1,
        )

    @classmethod
    def from_pretrained(cls, model_name: str, max_seq_len: int = 1024):
        """
        Load pretrained weights from HuggingFace.

        Supported: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        """

        if model_name not in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Supported: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'."
            )

        # Download and load HuggingFace model
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)  # type: ignore

        # Get config
        config = hf_model.config

        # Create our model with matching architecture
        model = cls(
            vocab_size=config.vocab_size,
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            num_layers=config.n_layer,
            max_seq_len=max_seq_len,
            dropout=0.0,  # No dropout during inference
        )

        # Map weights (you'll implement this)
        model._load_weights(hf_model)

        return model

    def _load_weights(self, hf_model: GPT2LMHeadModel):
        # fmt: off
        with torch.no_grad():
            self.embedding.token.weight.copy_(hf_model.transformer.wte.weight)
            self.embedding.positional.weight.copy_(hf_model.transformer.wpe.weight)
            for i in range(self.num_layers):
                self.transformer_blocks[i].ln1.load_state_dict(hf_model.transformer.h[i].ln_1.state_dict())
                self.transformer_blocks[i].attn.W_q.weight.copy_(hf_model.transformer.h[i].attn.c_attn.weight[:, : self.embed_dim].t())
                self.transformer_blocks[i].attn.W_q.bias.copy_(hf_model.transformer.h[i].attn.c_attn.bias[: self.embed_dim].t())
                self.transformer_blocks[i].attn.W_k.weight.copy_(hf_model.transformer.h[i].attn.c_attn.weight[:, self.embed_dim : 2 * self.embed_dim].t())
                self.transformer_blocks[i].attn.W_k.bias.copy_(hf_model.transformer.h[i].attn.c_attn.bias[self.embed_dim : 2 * self.embed_dim])
                self.transformer_blocks[i].attn.W_v.weight.copy_(hf_model.transformer.h[i].attn.c_attn.weight[:, 2 * self.embed_dim :].t())
                self.transformer_blocks[i].attn.W_v.bias.copy_(hf_model.transformer.h[i].attn.c_attn.bias[2 * self.embed_dim :])
                self.transformer_blocks[i].attn.W_o.weight.copy_(hf_model.transformer.h[i].attn.c_proj.weight.t())
                self.transformer_blocks[i].attn.W_o.bias.copy_(hf_model.transformer.h[i].attn.c_proj.bias)
                self.transformer_blocks[i].ln2.load_state_dict(hf_model.transformer.h[i].ln_2.state_dict())
                self.transformer_blocks[i].ff.ff1.weight.copy_(hf_model.transformer.h[i].mlp.c_fc.weight.t())
                self.transformer_blocks[i].ff.ff1.bias.copy_(hf_model.transformer.h[i].mlp.c_fc.bias)
                self.transformer_blocks[i].ff.ff2.weight.copy_(hf_model.transformer.h[i].mlp.c_proj.weight.t())
                self.transformer_blocks[i].ff.ff2.bias.copy_(hf_model.transformer.h[i].mlp.c_proj.bias)
            self.ln.load_state_dict(hf_model.transformer.ln_f.state_dict())
            self.out_ff.weight.copy_(hf_model.lm_head.weight)
        # fmt: on

    def forward(self, token_ids: "Tensor") -> "Tensor":
        # token_ids: [batch, seq_len]
        if len(token_ids.shape) != 2:
            raise RuntimeError("Expecting token_ids to be of shape (batch, seq_len).")
        out = self.embedding(token_ids)
        out = self.transformer_blocks(out)
        out = self.ln(out)
        logits = self.out_ff(out)
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
