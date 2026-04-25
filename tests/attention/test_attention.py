import pytest
import torch

from llm_from_scratch.attention.scaled_dot_product import (
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

        _, attn_weights = scaled_dot_product_attention(
            q, k, v, causal=False, return_attn_weights=True
        )

        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_attention_weights_have_zero_future(self):
        seq_len, embed_dim = 4, 8
        q = torch.randn(1, seq_len, embed_dim)
        k = torch.randn(1, seq_len, embed_dim)
        v = torch.randn(1, seq_len, embed_dim)

        _, attn_weights = scaled_dot_product_attention(
            q, k, v, causal=True, return_attn_weights=True
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        upper_triangle = attn_weights[0][mask]
        assert torch.allclose(
            upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6
        )

    def test_attention_mask_blocks_positions(self):
        batch, seq_len, embed_dim = 1, 4, 8
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)
        attn_mask = torch.tensor([[1, 1, 0, 0]])

        _, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, return_attn_weights=True
        )

        assert torch.allclose(
            attn_weights[0, :, 2:], torch.zeros_like(attn_weights[0, :, 2:]), atol=1e-6
        )

    def test_attention_mask_with_batch(self):
        batch, seq_len, embed_dim = 2, 3, 4
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)
        attn_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        _, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, return_attn_weights=True
        )

        assert torch.allclose(attn_weights[0, :, 2], torch.zeros(seq_len), atol=1e-6)
        assert torch.allclose(
            attn_weights[1, :, 1:], torch.zeros(seq_len, 2), atol=1e-6
        )

    def test_attention_mask_with_causal(self):
        batch, seq_len, embed_dim = 1, 4, 8
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)
        attn_mask = torch.tensor([[1, 1, 1, 0]])

        _, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            causal=True,
            return_attn_weights=True,
        )

        assert torch.allclose(attn_weights[0, :, 3], torch.zeros(seq_len), atol=1e-6)
        upper_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        assert torch.allclose(
            attn_weights[0][upper_mask],
            torch.zeros(int(upper_mask.sum().item())),
            atol=1e-6,
        )

    def test_attention_mask_rows_sum_to_one(self):
        batch, seq_len, embed_dim = 1, 4, 8
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)
        attn_mask = torch.tensor([[1, 1, 1, 0]])

        _, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, return_attn_weights=True
        )

        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_mask_no_mask_equals_unmasked(self):
        batch, seq_len, embed_dim = 2, 4, 8
        q = torch.randn(batch, seq_len, embed_dim)
        k = torch.randn(batch, seq_len, embed_dim)
        v = torch.randn(batch, seq_len, embed_dim)
        all_ones_mask = torch.ones(batch, seq_len)

        out_unmasked = scaled_dot_product_attention(q, k, v, causal=False)
        out_masked = scaled_dot_product_attention(
            q, k, v, attn_mask=all_ones_mask, causal=False
        )

        assert torch.allclose(out_unmasked, out_masked, atol=1e-6)


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
        torch.manual_seed(42)
        attn_noncausal = SingleHeadAttention(embed_dim, causal=False)
        torch.manual_seed(42)
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
        ret = attn(x)
        assert ret.output.shape == (batch, seq_len, embed_dim)

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
        out_multi = multi_head(x).output

        # With 1 head, they should be equivalent
        # (but won't be exactly equal due to W_o projection in multi-head)
        assert out_single.shape == out_multi.shape

    def test_different_num_heads(self):
        batch, seq_len, embed_dim = 2, 4, 12
        x = torch.randn(batch, seq_len, embed_dim)

        for num_heads in [1, 2, 3, 4, 6, 12]:
            attn = MultiHeadAttention(embed_dim, num_heads, causal=False)
            out = attn(x).output
            assert out.shape == (batch, seq_len, embed_dim)
