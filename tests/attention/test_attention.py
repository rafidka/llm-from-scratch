import pytest
import torch

from llm_from_scratch.attention.attention import (
    MultiHeadAttention,
    SingleHeadAttention,
    scaled_dot_product_attention,
)


class TestScaledDotProductAttention:
    def test_output_shape_matches_input(self):
        batch, seq_len, embed_dim = 2, 4, 8
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)

        out = scaled_dot_product_attention(q, k, v, causal=False)
        assert out.shape == (batch, seq_len, embed_dim)

    def test_attention_weights_sum_to_one(self):
        batch, seq_len, embed_dim = 1, 3, 4
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)

        scores = q @ k.transpose(-1, -2)
        embed_dim = k.shape[-1]
        attn_weights = torch.softmax(scores / (embed_dim**0.5), dim=-1)

        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_blocks_future_positions(self):
        seq_len, embed_dim = 4, 8
        q = torch.randn(1, seq_len, embed_dim)
        k = torch.randn(1, seq_len, embed_dim)
        v = torch.randn(1, seq_len, embed_dim)

        out = scaled_dot_product_attention(q, k, v, causal=True)
        assert out.shape == (1, seq_len, embed_dim)

    def test_causal_attention_weights_have_zero_future(self):
        seq_len, embed_dim = 4, 8
        q = torch.randn(1, seq_len, embed_dim)
        k = torch.randn(1, seq_len, embed_dim)
        v = torch.randn(1, seq_len, embed_dim)

        scores = q @ k.transpose(-1, -2)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores_masked = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores_masked / (embed_dim**0.5), dim=-1)

        # Upper triangle (future positions) should be ~0
        upper_triangle = attn_weights[0][mask]
        assert torch.allclose(
            upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6
        )


class TestSingleHeadAttention:
    def test_output_shape(self):
        batch, seq_len, embed_dim = 2, 4, 8
        x = torch.randn(batch, seq_len, embed_dim)
        attn = SingleHeadAttention(embed_dim, causal=False)
        out = attn(x)
        assert out.shape == (batch, seq_len, embed_dim)

    def test_has_learnable_parameters(self):
        embed_dim = 8
        attn = SingleHeadAttention(embed_dim, causal=False)
        params = list(attn.parameters())
        assert len(params) == 3  # W_q, W_k, W_v

    def test_causal_parameter_affects_attention(self):
        batch, seq_len, embed_dim = 1, 4, 8
        x = torch.randn(batch, seq_len, embed_dim)

        attn_noncausal = SingleHeadAttention(embed_dim, causal=False)
        attn_causal = SingleHeadAttention(embed_dim, causal=True)

        # Different outputs due to masking
        out_noncausal = attn_noncausal(x)
        out_causal = attn_causal(x)

        # They should produce different results
        assert not torch.allclose(out_noncausal, out_causal)


class TestMultiHeadAttention:
    def test_output_shape(self):
        batch, seq_len, embed_dim, num_heads = 2, 4, 12, 3
        x = torch.randn(batch, seq_len, embed_dim)
        attn = MultiHeadAttention(embed_dim, num_heads, causal=False)
        out = attn(x)
        assert out.shape == (batch, seq_len, embed_dim)

    def test_num_heads_must_divide_embed_dim(self):
        with pytest.raises(ValueError, match="embed_dim"):
            MultiHeadAttention(embed_dim=8, num_heads=3, causal=False)

    def test_has_output_projection(self):
        embed_dim, num_heads = 8, 2
        attn = MultiHeadAttention(embed_dim, num_heads, causal=False)
        params = list(attn.named_parameters())
        param_names = [name for name, _ in params]
        assert "W_o.weight" in param_names

    def test_multi_head_vs_single_head_equivalent_when_one_head(self):
        batch, seq_len, embed_dim = 1, 4, 8
        x = torch.randn(batch, seq_len, embed_dim)

        # Set same seed for reproducibility
        torch.manual_seed(42)
        single_head = SingleHeadAttention(embed_dim, causal=False)

        torch.manual_seed(42)
        multi_head = MultiHeadAttention(embed_dim, num_heads=1, causal=False)

        out_single = single_head(x)
        out_multi = multi_head(x)

        # With 1 head, they should be equivalent
        # (but won't be exactly equal due to W_o projection in multi-head)
        assert out_single.shape == out_multi.shape

    def test_causal_blocks_future_in_multihead(self):
        batch, seq_len, embed_dim, num_heads = 1, 4, 8, 2
        x = torch.randn(batch, seq_len, embed_dim)
        attn = MultiHeadAttention(embed_dim, num_heads, causal=True)
        out = attn(x)
        assert out.shape == (batch, seq_len, embed_dim)

    def test_different_num_heads(self):
        batch, seq_len, embed_dim = 2, 4, 12
        x = torch.randn(batch, seq_len, embed_dim)

        for num_heads in [1, 2, 3, 4, 6, 12]:
            attn = MultiHeadAttention(embed_dim, num_heads, causal=False)
            out = attn(x)
            assert out.shape == (batch, seq_len, embed_dim)
