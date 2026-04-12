"""
Utilities for loading pretrained GPT-2 weights from HuggingFace.

This module handles the conversion from HuggingFace's GPT-2 implementation
to our custom GPT implementation. The main challenge is that HuggingFace
uses Conv1D layers (with transposed weight layout) while our implementation
uses standard nn.Linear layers.

The `_copy_linear` helper handles this transpose automatically when needed.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers import GPT2LMHeadModel

from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.model.base import GPT, TransformerBlock
from llm_from_scratch.model.classification import GPTForClassification


SUPPORTED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


@dataclass
class _FakeLinear:
    """
    A minimal container that mimics nn.Linear's weight/bias interface.

    Used by `_extract_qkv` to represent the split Q, K, V projections extracted from
    HuggingFace's fused Conv1D layer. HuggingFace combines Q, K, V into a single Conv1D
    with output dimension 3 * embed_dim, which we split into three separate Linear
    layers.

    Unlike nn.Linear, this only holds weight and bias tensors without implementing the
    forward pass -- it's purely a data container for weight copying.
    """

    weight: torch.Tensor
    bias: torch.Tensor


def load_pretrained_lm(
    model_name: str,
    max_seq_len: int = 1024,
) -> GPTForCausalLM:
    return _load_pretrained(GPTForCausalLM, model_name, max_seq_len)


def load_pretrained_cls(
    model_name: str,
    num_classes: int,
    max_seq_len: int = 1024,
) -> GPTForClassification:
    return _load_pretrained(
        GPTForClassification,
        model_name,
        max_seq_len,
        num_classes=num_classes,
    )


def _load_pretrained(
    cls,
    model_name: str,
    max_seq_len: int = 1024,
    **kwargs,
):
    """
    Load a pretrained GPT model from HuggingFace.

    Args:
        model_name: One of "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        max_seq_len: Maximum sequence length for positional embeddings

    Returns:
        GPT model with weights loaded from HuggingFace

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}."
        )

    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    config = hf_model.config

    model = cls(
        vocab_size=config.vocab_size,
        embed_dim=config.n_embd,
        num_heads=config.n_head,
        num_layers=config.n_layer,
        max_seq_len=max_seq_len,
        dropout=0.0,
        **kwargs,
    )

    load_weights(model, hf_model)
    return model


def _copy_linear(
    dest: nn.Linear,
    src: nn.Linear | _FakeLinear,
    transpose: bool = False,
) -> None:
    """
    Copy weights and bias from src to dest Linear layer.

    Args:
        dest: Destination nn.Linear layer
        src: Source Linear layer or _FakeLinear container
        transpose: If True, transpose src.weight before copying.
            Needed when src is a HuggingFace Conv1D layer, which stores weights in
            [in_features, out_features] format instead of nn.Linear's [out_features,
            in_features] format.
    """
    weight = src.weight.t() if transpose else src.weight
    dest.weight.copy_(weight)
    if dest.bias is not None and src.bias is not None:
        dest.bias.copy_(src.bias)


def _copy_layernorm(dest: nn.LayerNorm, src: nn.LayerNorm) -> None:
    """Copy weights and bias from src to dest LayerNorm layer."""
    dest.load_state_dict(src.state_dict())


def _extract_qkv(
    hf_c_attn: Any,
    embed_dim: int,
) -> tuple[_FakeLinear, _FakeLinear, _FakeLinear]:
    """
    Extract Q, K, V from HuggingFace's fused Conv1D attention projection.

    HuggingFace's GPT-2 combines Q, K, V into a single Conv1D layer with output
    dimension 3 * embed_dim. This function splits it into three separate _FakeLinear
    containers.

    Args:
        hf_c_attn: The fused Conv1D layer from HuggingFace (c_attn)
        embed_dim: The embedding dimension

    Returns:
        Tuple of (_FakeLinear for Q, _FakeLinear for K, _FakeLinear for V)
    """
    w: torch.Tensor = hf_c_attn.weight
    b: torch.Tensor = hf_c_attn.bias

    q = _FakeLinear(weight=w[:, :embed_dim].t(), bias=b[:embed_dim])
    k = _FakeLinear(
        weight=w[:, embed_dim : 2 * embed_dim].t(),
        bias=b[embed_dim : 2 * embed_dim],
    )
    v = _FakeLinear(
        weight=w[:, 2 * embed_dim :].t(),
        bias=b[2 * embed_dim :],
    )
    return q, k, v


def load_weights(model: GPT | GPTForCausalLM, hf_model: GPT2LMHeadModel) -> None:
    """
    Load weights from HuggingFace GPT2LMHeadModel into our GPT model.

    This handles the structural differences between HuggingFace's implementation and
    ours, including:
    - Fused QKV Conv1D → separate Linear layers
    - Conv1D weight transpose
    - Layer naming differences
    """
    hf: Any = hf_model.transformer

    with torch.no_grad():
        model.embedding.token.weight.copy_(hf.wte.weight)
        model.embedding.positional.weight.copy_(hf.wpe.weight)

        for i in range(model.num_layers):
            our_block = cast(TransformerBlock, model.transformer_blocks[i])
            hf_block: Any = hf.h[i]

            q, k, v = _extract_qkv(hf_block.attn.c_attn, model.embed_dim)
            _copy_linear(our_block.attn.W_q, q)
            _copy_linear(our_block.attn.W_k, k)
            _copy_linear(our_block.attn.W_v, v)
            _copy_linear(our_block.attn.W_o, hf_block.attn.c_proj, transpose=True)

            _copy_layernorm(our_block.ln1, hf_block.ln_1)
            _copy_layernorm(our_block.ln2, hf_block.ln_2)

            _copy_linear(our_block.ff.ff1, hf_block.mlp.c_fc, transpose=True)
            _copy_linear(our_block.ff.ff2, hf_block.mlp.c_proj, transpose=True)

        _copy_layernorm(model.ln, hf.ln_f)

        if isinstance(model, GPTForCausalLM):
            model.lm_head.weight.copy_(hf_model.lm_head.weight)
