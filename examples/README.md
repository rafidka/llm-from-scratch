# Examples

Example scripts demonstrating each component of the LLM implementation, organized by course progression.

---

## 1. Tokenization

### `tokenizers/simple.py`
Demonstrates SimpleTokenizer - a basic word-level tokenizer. Shows vocabulary building from a corpus, encoding text to token IDs, and decoding back to text. Handles unknown tokens with `<|unk|>` special token.
```bash
uv run python -m examples.tokenizers.simple
```
*Source: `src/llm_from_scratch/tokenizers/simple.py`*

### `tokenizers/bpe.py`
Implements BPE (Byte Pair Encoding) tokenizer from scratch. Trains on a corpus by iteratively merging the most frequent character pairs. Demonstrates the core BPE algorithm with encode/decode methods.
```bash
uv run python -m examples.tokenizers.bpe
```
*Source: `src/llm_from_scratch/tokenizers/bpe.py`*

### `tokenizers/test_bpe_corpus.py`
Tests BPE tokenizer on a larger corpus ("The Picture of Dorian Gray"). Shows how vocabulary size grows with merge operations and compares tokenization efficiency.
```bash
uv run python -m examples.tokenizers.test_bpe_corpus
```
*Source: `src/llm_from_scratch/tokenizers/bpe.py`*

### `tokenizers/tiktoken_tokenizer.py`
Wrapper for OpenAI's tiktoken library. Loads GPT-2's pretrained tokenizer (50,257 vocab size). Demonstrates production-grade tokenization used in modern LLMs.
```bash
uv run python -m examples.tokenizers.tiktoken_tokenizer
```
*Source: `src/llm_from_scratch/tokenizers/tiktoken_adapter.py`*

### `tokenizers/compare_tokenizers.py`
Compares our BPE tokenizer (174 vocab, trained on small corpus) against GPT-2's pretrained tokenizer (50,257 vocab). Shows efficiency differences: GPT-2 tokenizes more compactly due to larger vocabulary learned from billions of tokens.
```bash
uv run python -m examples.tokenizers.compare_tokenizers
```

### `data/dataset.py`
Demonstrates LLMDataset class for next-token prediction using sliding window approach. Shows how to create PyTorch DataLoader with proper batching for language model training.
```bash
uv run python -m examples.data.dataset
```
*Source: `src/llm_from_scratch/data/dataset.py`, `src/llm_from_scratch/data/loader.py`*

### `data/dataset_tiny_shakespeare.py`
Downloads and prepares Tiny Shakespeare dataset (~1MB). Creates DataLoader for training language models. Handles dataset download, caching, and tokenization.
```bash
uv run python -m examples.data.dataset_tiny_shakespeare
```
*Source: `src/llm_from_scratch/data/dataset.py`*

---

## 2. Attention Mechanism

### `attention/simplified.py`
Demonstrates simplified self-attention without learnable parameters. Shows core attention mechanism: query-key similarity via dot product, softmax normalization, and weighted value aggregation. Useful for understanding attention fundamentals.
```bash
uv run python -m examples.attention.simplified
```
*Source: `src/llm_from_scratch/attention/simplified.py`*

### `attention/attention.py`
Implements full attention with trainable Q, K, V projections. Includes single-head attention and causal masking to prevent attending to future tokens (required for autoregressive language models).
```bash
uv run python -m examples.attention.attention
```
*Source: `src/llm_from_scratch/attention/qkv.py`, `src/llm_from_scratch/attention/scaled_dot_product.py`*

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
uv run python -m examples.model.transformer
```
*Source: `src/llm_from_scratch/model/transformer.py`, `src/llm_from_scratch/model/embeddings.py`*

### `model/transformer_without_topk.py`
Step-by-step demonstration of text generation without top-k sampling. Shows the core loop: forward pass, apply temperature, softmax, sample next token. Useful for understanding generation mechanics.
```bash
uv run python -m examples.model.transformer_without_topk
```

### `model/transformer_with_topk.py`
Step-by-step demonstration of text generation with top-k sampling. Shows how to restrict sampling to top k most likely tokens before applying softmax. Improves generation quality.
```bash
uv run python -m examples.model.transformer_with_topk
```
*Source: `src/llm_from_scratch/model/transformer.py` (see `generate` method)*

### `model/gradient_flow_comparison.py`
Compares gradient flow in pre-norm (GPT-2 style) vs post-norm (original Transformer) architectures. Trains both on a simple task and logs gradient magnitudes at early layers. Demonstrates why pre-norm trains more stably in deep networks (12+ layers). Pre-norm gradients are ~5x larger at early layers.
```bash
uv run python -m examples.model.gradient_flow_comparison
```
*Source: `src/llm_from_scratch/model/transformer.py`*

---

## 4. Pretraining

### `training/trainer.py`
Training script for Tiny Shakespeare dataset. Implements cross-entropy loss, AdamW optimizer with weight decay, learning rate warmup + cosine decay scheduling, and sample generation after each epoch. Uses GPT-tiny model (~50K params) for local testing.
```bash
uv run python -m examples.training.trainer
```
*Source: `src/llm_from_scratch/training/trainer.py`*

### `training/train_cloud.py`
Cloud GPU training script with checkpointing. Configured for GPT-2 Small (124M params) and Medium (355M params) sizes. Supports Wikitext-103 dataset and saves model checkpoints for resumable training.
```bash
uv run python -m examples.training.train_cloud
```
*Source: `src/llm_from_scratch/training/trainer.py`*

### `training/evaluation.py`
Evaluates perplexity on held-out data (Wikitext-103 validation split). Perplexity = exp(cross-entropy loss), lower is better. Compares our implementation against HuggingFace's GPT-2 as a baseline. Shows that context length significantly affects perplexity (1024 tokens → ~22-30 perplexity for GPT-2).
```bash
uv run python -m examples.training.evaluation
```
*Source: `src/llm_from_scratch/training/evaluate.py`*

---

## 5. Pretrained Weights

### `pretrained/generate.py`
Loads pretrained GPT-2 weights from HuggingFace into our architecture. Maps weight names between HuggingFace's Conv1D format and our nn.Linear format. Demonstrates text generation with identical outputs to HuggingFace's reference implementation (verified with temperature=0 for determinism).
```bash
uv run python -m examples.pretrained.generate
```
*Source: `src/llm_from_scratch/model/pretrained.py`*