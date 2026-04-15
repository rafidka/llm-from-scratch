# Examples

Example scripts demonstrating each component of the LLM implementation, organized by course progression.

---

## 1. Tokenization

### `tokenizers/simple.py`
Demonstrates SimpleTokenizer - a basic word-level tokenizer. Shows vocabulary building from a corpus, encoding text to token IDs, and decoding back to text. Handles unknown tokens with `<|unk|>` special token.
```bash
uv run python examples/tokenizers/simple.py
```
*Source: `src/llm_from_scratch/tokenizers/simple.py`*

### `tokenizers/bpe.py`
Implements BPE (Byte Pair Encoding) tokenizer from scratch. Trains on a corpus by iteratively merging the most frequent character pairs. Demonstrates the core BPE algorithm with encode/decode methods.
```bash
uv run python examples/tokenizers/bpe.py
```
*Source: `src/llm_from_scratch/tokenizers/bpe.py`*

### `tokenizers/test_bpe_corpus.py`
Tests BPE tokenizer on a larger corpus ("The Picture of Dorian Gray"). Shows how vocabulary size grows with merge operations and compares tokenization efficiency.
```bash
uv run python examples/tokenizers/test_bpe_corpus.py
```
*Source: `src/llm_from_scratch/tokenizers/bpe.py`*

### `tokenizers/tiktoken_tokenizer.py`
Wrapper for OpenAI's tiktoken library. Loads GPT-2's pretrained tokenizer (50,257 vocab size). Demonstrates production-grade tokenization used in modern LLMs.
```bash
uv run python examples/tokenizers/tiktoken_tokenizer.py
```
*Source: `src/llm_from_scratch/tokenizers/tiktoken_adapter.py`*

### `tokenizers/compare_tokenizers.py`
Compares our BPE tokenizer (174 vocab, trained on small corpus) against GPT-2's pretrained tokenizer (50,257 vocab). Shows efficiency differences: GPT-2 tokenizes more compactly due to larger vocabulary learned from billions of tokens.
```bash
uv run python examples/tokenizers/compare_tokenizers.py
```

### `data/dataset.py`
Demonstrates LLMDataset class for next-token prediction using sliding window approach. Shows how to create PyTorch DataLoader with proper batching for language model training.
```bash
uv run python examples/data/dataset.py
```
*Source: `src/llm_from_scratch/data/dataset.py`, `src/llm_from_scratch/data/loader.py`*

---

## 2. Attention Mechanism

### `attention/simplified.py`
Demonstrates simplified self-attention without learnable parameters. Shows core attention mechanism: query-key similarity via dot product, softmax normalization, and weighted value aggregation. Useful for understanding attention fundamentals.
```bash
uv run python examples/attention/simplified.py
```
*Source: `src/llm_from_scratch/attention/simplified.py`*

### `attention/scaled_dot_product.py`
Implements full attention with trainable Q, K, V projections. Includes single-head attention with causal masking to prevent attending to future tokens (required for autoregressive language models). Also demonstrates the standalone `scaled_dot_product_attention` function.
```bash
uv run python examples/attention/scaled_dot_product.py
```
*Source: `src/llm_from_scratch/attention/scaled_dot_product.py`*

### `attention/visualization_glove.ipynb`
Visualizes attention patterns using raw GloVe embeddings (untrained). Demonstrates that random/untrained Q/K projections produce near-uniform attention distributions - highlighting why learned projections are essential.
```bash
uv run jupyter notebook examples/attention/visualization_glove.ipynb
```

### `attention/visualization_pretrainedgpt2.ipynb`
Loads pretrained GPT-2 from HuggingFace and visualizes learned attention weights. Compares Layer 0 (local/syntactic patterns) vs Layer 11 (semantic/contextual patterns). Shows what attention learns during training.
```bash
uv run jupyter notebook examples/attention/visualization_pretrainedgpt2.ipynb
```

---

## 3. GPT Architecture

### `model/transformer.py`
Full GPT-2 architecture implementation: token embeddings, positional embeddings, transformer blocks with pre-norm, feed-forward networks with GELU activation, and text generation with temperature/top-k sampling.
```bash
uv run python examples/model/transformer.py
```
*Source: `src/llm_from_scratch/model/causallm.py`, `src/llm_from_scratch/model/embeddings.py`*

### `model/transformer_without_topk.py`
Step-by-step demonstration of text generation without top-k sampling. Shows the core loop: forward pass, apply temperature, softmax, sample next token. Useful for understanding generation mechanics.
```bash
uv run python examples/model/transformer_without_topk.py
```

### `model/transformer_with_topk.py`
Step-by-step demonstration of text generation with top-k sampling. Shows how to restrict sampling to top k most likely tokens before applying softmax. Improves generation quality.
```bash
uv run python examples/model/transformer_with_topk.py
```
*Source: `src/llm_from_scratch/model/causallm.py` (see `generate` method)*

### `model/gradient_flow_comparison.py`
Compares gradient flow in pre-norm (GPT-2 style) vs post-norm (original Transformer) architectures. Trains both on a simple task and logs gradient magnitudes at early layers. Demonstrates why pre-norm trains more stably in deep networks (12+ layers). Pre-norm gradients are ~5x larger at early layers.
```bash
uv run python examples/model/gradient_flow_comparison.py
```
*Source: `src/llm_from_scratch/model/causallm.py`*

### `model/classification.py`
Demonstrates GPTForClassification model - a GPT-based architecture adapted for classification tasks. Shows how to create a classification head on top of the GPT backbone and run a forward pass.
```bash
uv run python examples/model/classification.py
```
*Source: `src/llm_from_scratch/model/classification.py`*

---

## 4. Pretrained Weights

### `pretrained/generate.py`
Loads pretrained GPT-2 weights from HuggingFace into our architecture. Maps weight names between HuggingFace's Conv1D format and our nn.Linear format. Demonstrates text generation with identical outputs to HuggingFace's reference implementation (verified with temperature=0 for determinism).
```bash
uv run python examples/pretrained/generate.py
```
*Source: `src/llm_from_scratch/model/pretrained.py`*