# Progress Log

<!-- Session entries are added by the agent at the end of each working session. -->

## Session 1 — 2026-03-19 — Tokenization (Part 1)

### What we covered
- Discussed the tokenization spectrum: character-level → word-level → subword (BPE)
- Explained the regex pattern used for splitting text into tokens
- Built a `SimpleTokenizer` class with encode/decode functionality
- Added special tokens (`<|unk|>` and `<|endoftext|>`) for handling unknown tokens and document boundaries
- Discussed BPE algorithm conceptually with a worked example

### Key learnings
- Tokenization is the bridge between raw text and the integer sequences an LLM operates on
- `re.split()` with a capturing group preserves delimiters in the output
- Word-level tokenizers suffer from closed vocabulary, vocabulary explosion, and lack of morphological awareness — motivating subword approaches like BPE
- BPE iteratively merges the most frequent adjacent pair, building up from characters to subwords

### Code written
- `src/llm_from_scratch/tokens.py` — `SimpleTokenizer` class with `_tokenize`, `encode`, `decode`, special token support
- `examples/tokens.py` — Example script demonstrating the tokenizer

### PLAN.md items completed
- [x] Understand the tokenization spectrum: character → word → subword (BPE)
- [x] Build a simple tokenizer from scratch (vocabulary building, encode/decode)

### Open questions
- BPE implementation is next — `BPETokenizer` class with training (merge loop) and encoding (replay merges)

## Session 2 — 2026-03-20 to 2026-03-28 — Tokenization (Part 2)

### What we covered
- Implemented `BPETokenizer` class with BPE training algorithm
- Built the merge loop: count pairs, find most frequent, merge everywhere, store merge rule
- Implemented `encode`: pre-tokenize, split words into characters, replay merges in order
- Implemented `decode`: map IDs back to tokens
- Added special tokens (`<|unk|>` and ``) with reserved IDs
- Fixed space handling: changed regex to capture tokens with optional leading space
- Tested on "The Picture of Dorian Gray" corpus with 100 merges

### Key learnings
- BPE stores merge rules in order — encoding must replay them in the same order
- Space handling: GPT-2 style regex `r"( ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+)"` captures spaces as part of following token
- Vocabulary must include initial characters (from corpus) plus merged tokens plus special tokens
- `set(corpus)` gives non-deterministic iteration order — vocab IDs vary across runs (minor issue)

### Code written
- `src/llm_from_scratch/tokenizers/bpe.py` — `BPETokenizer` class with `_train`, `_create_initial_mapping`, `_tokenize`, `_merge_loop`, `_apply_merge`, `encode`, `decode`
- `src/llm_from_scratch/tokenizers/simple.py` — moved `SimpleTokenizer` here from `tokens.py`
- `examples/tokenizers/bpe.py` — Example script for BPE tokenizer
- `examples/tokenizers/test_bpe_corpus.py` — Test script with larger corpus

### PLAN.md items completed
- [x] **Deep dive**: Implement BPE merge algorithm by hand

### Open questions
- Vocabulary ordering is non-deterministic (minor) — could sort initial chars for reproducibility
- Next: Use `tiktoken` with GPT-2's tokenizer, then build `Dataset` and `DataLoader`

## Session 3 — 2026-03-28 — Tokenization (Part 3)

### What we covered
- Implemented `TiktokenTokenizer` wrapper class for OpenAI's tiktoken library
- Loaded GPT-2's pretrained tokenizer (50,257 vocab size)
- Compared our BPE tokenizer (174 vocab) against GPT-2's tokenizer
- Observed efficiency differences: GPT-2 tokenizes "Hello world!" in 3 tokens vs our 8 tokens

### Key learnings
- `tiktoken.get_encoding("gpt2")` loads GPT-2's pretrained BPE tokenizer
- GPT-2's massive vocabulary (50,257) learned from billions of tokens makes it far more efficient than our toy tokenizer
- Production tokenizers like tiktoken use byte-level BPE and sophisticated regex patterns for robustness
- The same interface (`encode`, `decode`) allows swapping between tokenizers easily

### Code written
- `src/llm_from_scratch/tokenizers/tiktoken_adapter.py` — `TiktokenTokenizer` wrapper class
- `examples/tokenizers/compare_tokenizers.py` — Comparison script for GPT-2 vs our BPE

### PLAN.md items completed
- [x] Use `tiktoken` with GPT-2's tokenizer (`cl100k_base` / `gpt2` encoding)

### Open questions
- None — Phase 1 complete

## Session 4 — 2026-03-28 — Tokenization (Part 4)

### What we covered
- Built `LLMDataset` class for next-token prediction with sliding window approach
- Implemented `__len__` and `__getitem__` methods
- Created `create_dataloader` function wrapping PyTorch's DataLoader
- Verified batching produces correct shapes `(batch_size, max_length)`

### Key learnings
- Sliding window: for tokens `[t1, t2, ..., tn]`, create input-target pairs where target is shifted by 1
- `__len__` formula: `max(floor((len(tokens) - max_length - 1) / stride) + 1, 0)`
- PyTorch's default `collate_fn` correctly stacks tuples of tensors

### Code written
- `src/llm_from_scratch/data/dataset.py` — `LLMDataset` class
- `src/llm_from_scratch/data/loader.py` — `create_dataloader` function
- `examples/data/dataset.py` — Example demonstrating dataset and dataloader

### PLAN.md items completed
- [x] Build a `Dataset` and `DataLoader` for next-token prediction (sliding window)

### Open questions
- Next: Phase 2 — Attention Mechanism

## Session 5 — 2026-03-28 — Attention Mechanism (Part 1)

### What we covered
- Implemented simplified self-attention (dot product attention with no learnable parameters)
- Added trainable Q, K, V projections → single-head attention
- Implemented causal (masked) self-attention to prevent attending to future tokens
- Discussed why `sqrt(d_k)` scaling prevents softmax saturation
- Debugged mask implementation: `0 * -inf = nan`, fixed with `masked_fill`

### Key learnings
- Self-attention computes similarity via dot product, normalizes with softmax, then weighted-sums values
- Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`
- Causal mask: upper triangular with `-inf` above diagonal ensures autoregressive property
- `masked_fill` is differentiable and standard for applying masks — gradients don't need to flow through masked positions

### Code written
- `src/llm_from_scratch/attention/simplified.py` — `SimplifiedSelfAttention` class and `simplified_self_attention` function
- `src/llm_from_scratch/attention/qkv.py` — `self_attention`, `self_causal_attention` functions and `SingleHeadAttention`, `CausalSelfAttention` classes
- `examples/attention/simplified.py` — Example script for simplified attention

### PLAN.md items completed
- [x] Implement simplified self-attention (dot product attention)
- [x] Add trainable weights → single-head attention
- [x] Implement causal (masked) self-attention

---
  
## Session 6 — 2026-03-29 — Attention Mechanism (Part 2)

### What we covered
- Implemented multi-head attention with parameterizable attention function
- Discussed why one big projection matrix is more efficient than separate per-head matrices
- Added output projection `W_o` to combine information across heads
- Created `MultiHeadAttentionBase` with pluggable attention function, then subclassed `MultiHeadAttention` and `CausalMultiHeadAttention`
- Verified causal masking: each row only attends to current and past tokens

### Key learnings
- Multi-head attention splits `embed_dim` into `num_heads × head_dim`, processes in parallel, then concatenates
- `W_o` projection is essential for heads to "mix" — without it, heads don't interact
- Reshape pattern: `[batch, seq, embed] → view + transpose → [batch, heads, seq, head_dim]`
- Concat pattern: `[batch, heads, seq, head_dim] → transpose + view → [batch, seq, embed]`

### Code written
- `src/llm_from_scratch/attention/qkv.py` — Added `MultiHeadAttentionBase`, `MultiHeadAttention`, `CausalMultiHeadAttention`

### PLAN.md items completed
- [x] Extend to multi-head attention

### Open questions
- Next: Deep dives (attention visualization, computational complexity) or Phase 3 (GPT Architecture)

---

## Session 7 — 2026-03-29 — Attention Mechanism (Part 3: Deep Dives)

### What we covered
- Explored attention as a "soft dictionary lookup" — queries match all keys with learned weights
- Compared our attention with trained GPT-2 attention using visualizations
- Loaded pre-trained GloVe embeddings and visualized raw attention patterns
- Loaded GPT-2 via Hugging Face transformers and visualized learned attention weights
- Compared Layer 0 vs Layer 11 attention patterns: early layers more local, later layers more semantic
- Discussed computational complexity: O(n²) time and space, memory bottleneck for long sequences
- Reviewed modern solutions: sparse attention, Flash Attention, sliding window, linear attention, KV cache

### Key learnings
- Raw embeddings without learned Q/K projections don't produce meaningful attention patterns
- Learned attention shows what the model finds relevant through training
- Early layers capture local/syntactic patterns; later layers capture semantic/contextual relationships
- GPT-2 has 12 layers ×12 heads = 144 attention matrices to analyze
- O(n²) complexity makes long sequences expensive: 128K context → 64 GB per attention matrix
- Modern LLMs use various techniques to reduce complexity: Flash Attention, KV cache, GQA, sliding window

### Code written
- `examples/attention/attention_visualization.ipynb` — GloVe attention visualization
- `examples/attention/gpt2_attn_vis.ipynb` — GPT-2 attention visualization

### PLAN.md items completed
- [x] **Deep dive**: Attention as soft dictionary lookup, attention pattern visualization
- [x] **Deep dive**: Computational complexity of attention (O(n²) and why it matters)

### Open questions
- Next: Phase 3 — GPT Architecture
