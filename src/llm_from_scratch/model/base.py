from typing import TYPE_CHECKING, cast

from torch import nn
from torch.utils.checkpoint import checkpoint

from llm_from_scratch.attention.scaled_dot_product import MultiHeadAttention
from llm_from_scratch.model.embeddings import GPTEmbeddings
from llm_from_scratch.model.lora import LoRALayer

if TYPE_CHECKING:
    from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # LoRA stuff.
        self.lora_ff1: LoRALayer | None = None
        self.lora_ff2: LoRALayer | None = None

    def lorafy(self, rank: int, alpha: float, sigma: float = 0.02):
        if self.lora_ff1:
            raise RuntimeError("Already LoRAfied")

        self.lora_ff1 = LoRALayer(self.ff1, rank, alpha, sigma)
        self.lora_ff2 = LoRALayer(self.ff2, rank, alpha, sigma)

    def forward(self, x: "Tensor") -> "Tensor":
        out = x
        out = self.lora_ff1(out) if self.lora_ff1 else self.ff1(out)
        out = self.gelu(out)
        out = self.lora_ff2(out) if self.lora_ff2 else self.ff2(out)
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

    def lorafy(self, rank: int, alpha: float, sigma: float = 0.02):
        self.attn.lorafy(rank, alpha, sigma)
        self.ln1.requires_grad_(False)
        self.ff.lorafy(rank, alpha, sigma)
        self.ln2.requires_grad_(False)

    def forward(self, x: "Tensor", attn_mask: "Tensor | None" = None) -> "Tensor":
        # x: [batch, seq_len, embed_dim]
        out = x
        out = out + self.dropout(self.attn(self.ln1(out), attn_mask))
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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # Save model info.
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.embeddings = GPTEmbeddings(vocab_size, embed_dim, max_seq_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)

    @classmethod
    def tiny(
        cls,
        vocab_size: int,
        max_seq_len: int,
        use_gradient_checkpointing: bool = False,
    ):
        """Create a very small GPT model for testing purposes.

        Note: dropout is set to 0.0 to ensure deterministic behavior in tests.
        Use GPT.small/medium/large() for training, which use dropout=0.1.
        """
        return cls(
            vocab_size=vocab_size,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=max_seq_len,
            dropout=0.0,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @classmethod
    def small(
        cls,
        vocab_size: int,
        max_seq_len: int = 1024,
        use_gradient_checkpointing: bool = False,
    ):
        """GPT-2 Small: 124M parameters"""
        return cls(
            vocab_size,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            max_seq_len=max_seq_len,
            dropout=0.1,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @classmethod
    def medium(
        cls,
        vocab_size: int,
        max_seq_len: int = 1024,
        use_gradient_checkpointing: bool = False,
    ):
        """GPT-2 Medium: 355M parameters"""
        return cls(
            vocab_size,
            embed_dim=1024,
            num_heads=16,
            num_layers=24,
            max_seq_len=max_seq_len,
            dropout=0.1,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @classmethod
    def large(
        cls,
        vocab_size: int,
        max_seq_len: int = 1024,
        use_gradient_checkpointing: bool = False,
    ):
        """GPT-2 Large: 774M parameters"""
        return cls(
            vocab_size,
            embed_dim=1280,
            num_heads=20,
            num_layers=36,
            max_seq_len=max_seq_len,
            dropout=0.1,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def lorafy(self, rank: int, alpha: float, sigma: float = 0.02):
        self.embeddings.requires_grad_(False)
        for block in self.transformer_blocks:
            cast(TransformerBlock, block).lorafy(rank, alpha, sigma)
        self.ln.requires_grad_(False)

    def forward(
        self, token_ids: "Tensor", attn_mask: "Tensor | None" = None
    ) -> "Tensor":
        # token_ids: [batch, seq_len]
        if len(token_ids.shape) != 2:
            raise RuntimeError("Expecting token_ids to be of shape (batch, seq_len).")
        out = self.embeddings(token_ids)
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing:
                out = checkpoint(block, out, attn_mask, use_reentrant=False)
            else:
                out = block(out, attn_mask)
        out = self.ln(out)
        return out  # [batch, seq_len, embed_dim]
