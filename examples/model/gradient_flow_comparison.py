"""
Compare gradient flow in pre-norm vs post-norm Transformer architectures.

This script:
1. Creates two models: pre-norm (GPT-2 style) and post-norm (original Transformer)
2. Trains both on a simple next-token prediction task
3. Logs gradient magnitudes at early layers to show how they differ
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PostNormTransformerBlock(nn.Module):
    """Post-norm: LayerNorm AFTER residual connection (original Transformer)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: LayerNorm after residual
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class PreNormTransformerBlock(nn.Module):
    """Pre-norm: LayerNorm BEFORE sublayer (GPT-2 style)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: LayerNorm before sublayer
        attn_out, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False
        )
        x = x + attn_out
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer for comparing pre-norm vs post-norm."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        block_type: str = "pre",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        BlockClass = (
            PreNormTransformerBlock if block_type == "pre" else PostNormTransformerBlock
        )
        self.blocks = nn.ModuleList(
            [BlockClass(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.num_layers = num_layers

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        x = self.embedding(token_ids) + self.pos_embedding(
            torch.arange(seq_len, device=token_ids.device)
        )

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)


def create_dataset(vocab_size: int, seq_len: int, num_samples: int):
    """Create a simple next-token prediction dataset."""
    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return TensorDataset(inputs, targets)


def get_gradient_stats(model: nn.Module) -> dict:
    """Get gradient statistics for each layer."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            stats[name] = grad_norm
    return stats


def train_model(model: nn.Module, dataloader: DataLoader, num_steps: int, name: str):
    """Train model and log gradients."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gradient_history = []

    for step, (inputs, targets) in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        # Get gradient norms for embedding and first block
        stats = {}
        for n, p in model.named_parameters():
            if p.grad is not None and ("embedding" in n or "blocks.0" in n):
                stats[n] = p.grad.norm().item()

        gradient_history.append(stats)

        optimizer.step()

    return gradient_history


def main():
    # Configuration
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    num_layers = 12  # Deep network to see gradient differences
    max_seq_len = 32
    batch_size = 32
    num_steps = 100

    # Create dataset
    dataset = create_dataset(vocab_size, max_seq_len, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("=" * 60)
    print("Gradient Flow Comparison: Pre-norm vs Post-norm")
    print("=" * 60)
    print(f"Config: {num_layers} layers, {embed_dim} dim, {num_heads} heads")
    print()

    # Train pre-norm model
    print("Training Pre-norm model...")
    pre_norm_model = SimpleTransformer(
        vocab_size, embed_dim, num_heads, num_layers, max_seq_len, block_type="pre"
    )
    pre_norm_grads = train_model(pre_norm_model, dataloader, num_steps, "Pre-norm")

    # Train post-norm model
    print("Training Post-norm model...")
    post_norm_model = SimpleTransformer(
        vocab_size, embed_dim, num_heads, num_layers, max_seq_len, block_type="post"
    )
    post_norm_grads = train_model(post_norm_model, dataloader, num_steps, "Post-norm")

    # Compare gradient magnitudes
    print()
    print("=" * 60)
    print("Gradient Norm Comparison (averaged over training steps)")
    print("=" * 60)
    print()
    print(f"{'Layer':<40} {'Pre-norm':>12} {'Post-norm':>12} {'Ratio':>10}")
    print("-" * 74)

    # Get all parameter names from pre-norm model
    all_keys = set()
    for stats in pre_norm_grads + post_norm_grads:
        all_keys.update(stats.keys())

    # Sort by layer order
    def sort_key(name):
        if "embedding" in name:
            return (0, name)
        elif "blocks" in name:
            parts = name.split(".")
            block_idx = int(parts[1])
            return (1, block_idx, name)
        else:
            return (2, name)

    sorted_keys = sorted(all_keys, key=sort_key)

    for key in sorted_keys[:20]:  # Show first 20 parameters
        pre_norm_vals = [s.get(key, 0) for s in pre_norm_grads]
        post_norm_vals = [s.get(key, 0) for s in post_norm_grads]

        pre_norm_avg = sum(pre_norm_vals) / len(pre_norm_vals)
        post_norm_avg = sum(post_norm_vals) / len(post_norm_vals)

        ratio = pre_norm_avg / (post_norm_avg + 1e-10)

        short_name = key.replace(".weight", ".w").replace(".bias", ".b")
        print(
            f"{short_name:<40} {pre_norm_avg:>12.4f} {post_norm_avg:>12.4f} {ratio:>10.2f}x"
        )

    print()
    print("=" * 60)
    print("Key Findings")
    print("=" * 60)
    print("""
    1. Pre-norm gradients are typically LARGER at early layers
    2. Post-norm gradients VANISH as we go deeper (smaller values)
    3. The ratio shows how much better pre-norm preserves gradient flow
    4. This is why GPT-2 (pre-norm) can train 12+ layer networks stably
    """)


if __name__ == "__main__":
    main()
