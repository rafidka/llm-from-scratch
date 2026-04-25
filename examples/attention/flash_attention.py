from itertools import product

import torch
from torch import randn
from torch.nn import functional as F

from llm_from_scratch.attention.scaled_dot_product import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def benchmark(q, k, v, num_iters=50, warmup=10):
    results = {}

    # --- Our implementation ---
    for _ in range(warmup):
        scaled_dot_product_attention(q, k, v, causal=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        scaled_dot_product_attention(q, k, v, causal=True)
    end.record()
    torch.cuda.synchronize()

    results["ours_time"] = start.elapsed_time(end) / num_iters
    results["ours_mem"] = torch.cuda.max_memory_allocated() / 1e9

    # --- PyTorch F.scaled_dot_product_attention ---
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(num_iters):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    end.record()
    torch.cuda.synchronize()

    results["pytorch_time"] = start.elapsed_time(end) / num_iters
    results["pytorch_mem"] = torch.cuda.max_memory_allocated() / 1e9

    # --- flash-attn library ---
    # flash_attn_func expects [batch, seq_len, num_heads, head_dim]
    # (different from PyTorch's [batch, num_heads, seq_len, head_dim])
    if HAS_FLASH_ATTN:
        q_fa = q.transpose(1, 2)
        k_fa = k.transpose(1, 2)
        v_fa = v.transpose(1, 2)

        for _ in range(warmup):
            flash_attn_func(q_fa, k_fa, v_fa, causal=True)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start.record()
        for _ in range(num_iters):
            flash_attn_func(q_fa, k_fa, v_fa, causal=True)
        end.record()
        torch.cuda.synchronize()

        results["fa_time"] = start.elapsed_time(end) / num_iters
        results["fa_mem"] = torch.cuda.max_memory_allocated() / 1e9

        del q_fa, k_fa, v_fa

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA required for Flash Attention benchmark")
        return

    device = torch.device("cuda")

    num_heads = 12
    head_dim = 64
    batch = 8

    headers = [
        f"{'dtype':>16}",
        f"{'seq_len':>8}",
        f"{'ours (ms)':>10}",
        f"{'ours (GB)':>10}",
        f"{'pytorch (ms)':>12}",
        f"{'pytorch (GB)':>12}",
    ]
    if HAS_FLASH_ATTN:
        headers += [f"{'fa-lib (ms)':>12}", f"{'fa-lib (GB)':>12}"]

    print(" | ".join(headers))
    print("-" * (len(" | ".join(headers))))

    for dtype, seq_len in product(
        [torch.bfloat16, torch.float16, torch.float32],
        [512, 1024, 2048, 4096, 8192],
    ):
        if dtype == torch.float32 and seq_len > 4096:
            continue

        q = randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        results = benchmark(q, k, v)

        row = [
            f"{str(dtype):>16}",
            f"{seq_len:>8}",
            f"{results['ours_time']:>10.2f}",
            f"{results['ours_mem']:>10.2f}",
            f"{results['pytorch_time']:>12.2f}",
            f"{results['pytorch_mem']:>12.2f}",
        ]
        if HAS_FLASH_ATTN:
            row += [
                f"{results['fa_time']:>12.2f}",
                f"{results['fa_mem']:>12.2f}",
            ]

        print(" | ".join(row))

        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
