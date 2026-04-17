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

---

## Session 8 — 2026-03-30 — GPT Architecture

### What we covered
- Implemented `GPTEmbedding` with token + positional embeddings
- Implemented `FeedForward` network with GELU activation and dropout
- Implemented `TransformerBlock` with pre-norm architecture (LayerNorm before sublayers)
- Implemented `GPT` class: embeddings → transformer blocks → final LayerNorm → output projection
- Discussed pre-norm vs post-norm: GPT-2 uses pre-norm for better gradient flow
- Implemented `generate` method with temperature, top-k sampling, and context window truncation

### Key learnings
- Pre-norm (GPT-2): `x = x + sublayer(layer_norm(x))` — LayerNorm before sublayer
- Post-norm (original Transformer): `x = layer_norm(x + sublayer(x))` — LayerNorm after sublayer
- Pre-norm helps with gradient flow and training stability in deep networks
- Output projection returns logits, not argmax — needed for cross-entropy loss during training
- Generation stops at EOS token in production, but we use `max_new_tokens` for simplicity

### Code written
- `src/llm_from_scratch/model/embeddings.py` — `GPTEmbedding` class
- `src/llm_from_scratch/model/transformer.py` — `FeedForward`, `TransformerBlock`, `GPT` classes

### PLAN.md items completed
- [x] Token embeddings + absolute positional embeddings
- [x] LayerNorm and residual connections
- [x] Feed-forward network (with GELU activation)
- [x] Assemble the Transformer block
- [x] Stack blocks into full GPT-2 architecture
- [x] Text generation (greedy, temperature, top-k, top-p sampling)

### Open questions
- Next: Deep dives (residual connections, pre-norm vs post-norm) or Phase 4 (Pretraining)

---

## Session 9 — 2026-03-30 — GPT Architecture Deep Dives

### What we covered
- Discussed why residual connections matter: gradient flow through deep networks
- Compared pre-norm vs post-norm architectures
- Created gradient flow demonstration comparing both approaches
- Observed that pre-norm gradients are 5x larger at embedding layers

### Key learnings
- Residual connections provide a "clean path" for gradients: `∂(x + sublayer(x))/∂x = 1 + ∂sublayer(x)/∂x`
- Pre-norm keeps LayerNorm out of the residual path, preserving gradients
- Post-norm gradients vanish at early layers — requires learning rate warmup
- Pre-norm (GPT-2, modern transformers) trains stably without warmup
- Demonstrated with 12-layer networks: pre-norm gradients ~5x larger at embeddings

### Code written
- `examples/model/gradient_flow_comparison.py` — Script comparing pre-norm vs post-norm gradient flow

### PLAN.md items completed
- [x] **Deep dive**: Why residual connections are critical (gradient flow)
- [x] **Deep dive**: Pre-norm vs post-norm architecture

### Open questions
- Next: Phase 4 — Pretraining

---

## Session 10 — 2026-04-06 — Pretraining (Part 1)

### What we covered
- Implemented `GPTTrainer` class with training loop structure
- Built cross-entropy loss computation for next-token prediction
- Added `GPT.test()` factory method for small test models
- Set up Tiny Shakespeare dataset download script
- Discussed hyperparameter tuning: learning rate (3e-4), weight decay (0.01)
- Training successfully runs with decreasing loss (~11 → ~5 in one epoch)

### Key learnings
- Cross-entropy loss expects logits `[batch*seq_len, vocab_size]` and targets `[batch*seq_len]`
- AdamW default lr (0.001) is too high for LLMs; typical range is 1e-4 to 3e-4
- Model must be moved to device via `model.to(device)` before training
- Stride controls overlap in sliding window: smaller stride = more data but more redundancy
- Batch size is constrained by GPU/MPS memory

### Code written
- `src/llm_from_scratch/training/trainer.py` — `GPTTrainer` class with `train_step`, `train_epoch`, `train` methods
- `src/llm_from_scratch/examples/training/trainer.py` — Training script for Tiny Shakespeare
- `src/llm_from_scratch/examples/data/dataset_tiny_shakespeare.py` — Dataset download and dataloader creation

### PLAN.md items completed
- [x] Cross-entropy loss for next-token prediction
- [x] Training loop with AdamW optimizer

### Open questions
- Next: Learning rate scheduling (warmup + cosine decay)

---

## Session 11 — 2026-04-06 — Pretraining (Part 2: LR Scheduling)

### What we covered
- Implemented learning rate warmup + cosine decay schedule
- Added GPT-2 style weight initialization (N(0, 0.02))
- Completed full training run on Tiny Shakespeare (10 epochs)
- Final loss: ~3.5-4.0 (perplexity ~33-55)

### Key learnings
- Warmup prevents early training instability: LR ramps from 0 to max_lr over first 10% of steps
- Cosine decay allows fine-tuning: LR decays smoothly from max_lr to min_lr (max_lr/10)
- GPT-2 initializes weights from N(0, 0.02) for embeddings and linear layers
- Weight decay (0.01) provides L2 regularization during AdamW optimization
- LR must be set before optimizer.step(), not after

### Code written
- `src/llm_from_scratch/training/trainer.py` — Added `get_lr()` method with warmup + cosine decay
- `src/llm_from_scratch/examples/training/trainer.py` — Added `init_weights()` function for GPT-2 style init

### PLAN.md items completed
- [x] Learning rate scheduling (warmup + cosine decay)

### Open questions
- Next: Evaluate with perplexity and sample generation

---

## Session 12 — 2026-04-06 — Sample Generation

### What we covered
- Added text generation to GPTTrainer — generates samples after each epoch
- Tested generation with "To be, or not" prompt
- Observed output quality is poor due to tiny model (2 layers, 64 dim)
- Discussed model size vs. output quality tradeoff

### Key learnings
- Tiny model (~50K params) learns basic patterns: character names, dialogue format
- Loss ~3.5 = perplexity ~33 → model still very uncertain
- Need larger model (more layers, wider embeddings) for coherent text generation
- `torch.no_grad()` during generation saves memory and speeds up inference

### Code written
- `src/llm_from_scratch/training/trainer.py` — Added `generate()` method with prompt, temperature, top_k

### PLAN.md items completed
- [x] Evaluate with sample generation

### Open questions
- Next: Train a larger model on cloud GPU

---

## Session 13 — 2026-04-07 — Cloud Training Setup

### What we covered
- Discussed GPT-2 model sizes (Small: 124M, Medium: 355M params)
- Added GPT.gpt2_small() and GPT.gpt2_medium() factory methods
- Created train_cloud.py for training on cloud GPUs
- Switched from Wikipedia to Wikitext-103 dataset (Wikipedia deprecated in datasets lib)
- Set up VS Code launch configuration for debugging

### Key learnings
- GPT-2 Small: 768 embed_dim, 12 layers, 12 heads, ~124M params
- GPT-2 Medium: 1024 embed_dim, 24 layers, 16 heads, ~355M params
- Wikitext-103 is a popular LM dataset (~500MB, cleaner than Wikipedia)
- A100 80GB can handle batch sizes of 64-128 for Small, 32-64 for Medium

### Code written
- `src/llm_from_scratch/model/transformer.py` — Added gpt2_small() and gpt2_medium() factory methods
- `src/llm_from_scratch/data/dataset.py` — Added from_tokens() class method for pre-tokenized data
- `src/llm_from_scratch/examples/training/train_cloud.py` — Cloud training script with checkpointing

### PLAN.md items completed
- None yet (cloud training script ready, needs to be run)

### Open questions
- Next: Run training on cloud GPU (RunPod/vast.ai)

---

## Session 14 — 2026-04-07 — Cloud Training (RTX 3090)

### What we covered
- Running train_cloud.py on RTX 3090 GPU
- Using Wikitext-103 dataset for training
- Training in progress

### Key learnings
- (To be filled after training completes)

### Code written
- (Training script already created in Session 13)

### PLAN.md items completed
- [x] Train a larger model on cloud GPU (in progress)

### Open questions
- Next: Evaluate with perplexity once training completes

---

## Session 15 — 2026-04-09 — Loading Pretrained GPT-2 Weights

### What we covered
- Analyzed HuggingFace GPT-2 architecture vs our implementation
- Mapped weight names between architectures (wte/wpe, c_attn/c_proj, etc.)
- Handled Conv1D transpose (HuggingFace stores weights as [in, out], we use [out, in])
- Implemented GPT.from_pretrained() to load weights from HuggingFace
- Verified identical outputs between our model and HuggingFace GPT-2

### Key learnings
- HuggingFace GPT-2 uses Conv1D which stores weights transposed vs nn.Linear
- GPT-2 combines Q, K, V into single c_attn tensor (shape [768, 2304]) vs our separate W_q, W_k, W_v
- Pre-norm architecture matches between our implementation and HuggingFace
- Weight tying: HuggingFace shares wte and lm_head weights
- Bias handling: GPT-2 has biases in attention layers (our original impl had bias=False)
- temperature=0 for greedy decoding produces deterministic, identical outputs

### Code written
- `src/llm_from_scratch/model/transformer.py` — Added from_pretrained() and _load_weights()
- `src/llm_from_scratch/model/transformer.py` — Added gpt2_large() factory method
- `src/llm_from_scratch/attention/attention.py` — Added bias to W_q, W_k, W_v, W_o
- `src/llm_from_scratch/model/embeddings.py` — Fixed attribute name (embedding -> token)
- `src/llm_from_scratch/examples/model/load_pretrianed_ours.py` — Test script for our model
- `src/llm_from_scratch/examples/model/load_pretrianed_hf.py` — Test script for HuggingFace model

### PLAN.md items completed
- [x] Understand GPT-2 weight layout from OpenAI/HuggingFace
- [x] Map weights to our architecture
- [x] Verify correctness via text generation
- [x] Compare our outputs to HuggingFace's GPT-2

### Open questions
- Next: Evaluate perplexity on held-out data

---

## Session 16 — 2026-04-09 — Perplexity Evaluation

### What we covered
- Implemented evaluate_perplexity() function for our model and HuggingFace model
- Investigated perplexity discrepancies and tokenizer mismatch issues
- Discovered context length affects perplexity significantly

### Key learnings
- Perplexity = exp(average cross-entropy loss), standard LM evaluation metric
- **Context length matters**: Longer sequences → lower perplexity (more context = better predictions)
  - 128 tokens: ~45-60 perplexity
  - 256 tokens: ~35-50 perplexity
  - 1024 tokens: ~22-30 perplexity (GPT-2's training context)
- **Tokenizer must match**: GPT-2 pretrained weights require GPT-2's tokenizer (vocab size 50257)
- **HuggingFace label handling**: GPT2LMHeadModel expects labels=input_ids (unshifted), shifts internally
- Our LLMDataset pre-shifts targets, which works for our model but needs adjustment for HF models

### Code written
- `src/llm_from_scratch/training/evaluate.py` — evaluate_perplexity() for our model
- `src/llm_from_scratch/training/evaluate_hf.py` — evaluate_perplexity() for HuggingFace models
- `src/llm_from_scratch/examples/pretrained/evaluate_gpt2.py` — Script to evaluate our model
- `src/llm_from_scratch/examples/pretrained/evaluate_gpt2_hf.py` — Script to evaluate HF model

### PLAN.md items completed
- [x] Evaluate perplexity on held-out data

### Open questions
- None — Phase 5 complete!

---

## Session 17 — 2026-04-12 — Classification Fine-tuning

### What we covered
- Refactored model architecture: GPT (base) → GPTForCausalLM / GPTForClassification
- Implemented GPTForClassification with classification head
- Implemented GPTForClassificationTrainer with evaluation metrics
- Created DatasetForClassification for HuggingFace datasets
- Set up IMDB sentiment classification training

### Key learnings
- Classification head uses last token's hidden state: `cls_head(out[:, -1, :])`
- Lower learning rate for fine-tuning: 5e-5 vs 3e-4 for pretraining
- Evaluation metrics: accuracy, precision, recall, F1
- Division by zero guards needed for precision/recall when TP=0
- `torch.no_grad()` essential for eval to avoid memory overhead

### Code written
- `src/llm_from_scratch/model/classification.py` — GPTForClassification class
- `src/llm_from_scratch/model/pretrained.py` — load_pretrained_cls() and load_pretrained_lm()
- `src/llm_from_scratch/training/classification.py` — GPTForClassificationTrainer with eval
- `src/llm_from_scratch/data/classification.py` — DatasetForClassification and create_dataloader
- `examples/training/classification.py` — IMDB sentiment classification training script

### PLAN.md items completed
- [x] Classification fine-tuning (e.g., spam detection or sentiment)

### Open questions
- Next: Instruction fine-tuning or LoRA deep dive

---

## Session 18 — 2026-04-13 — Instruction Fine-tuning

### What we covered
- Understood masked loss concept: only compute loss on response tokens, not prompt tokens
- Refactored `DatasetForInstructionFineTuning` to use `ignore_index=-100` instead of a separate mask tensor
- Simplified the pipeline: no new model class or trainer needed — `GPTForCausalLMTrainer` works with `CrossEntropyLoss(ignore_index=-100)`
- Added `eos_token_id` parameter to `GPTForCausalLM.generate()` for early stopping
- Updated `GPTForCausalLMTrainer.generate()` with instruction-style prompts
- Set up instruction fine-tuning training script using Alpaca dataset and GPT-2 Large
- Discussed training efficiency: gradient accumulation, mixed precision, length-grouped batching, gradient checkpointing

### Key learnings
- `ignore_index=-100` in `CrossEntropyLoss` is the standard way to mask loss — eliminates need for a separate mask tensor
- Padding `target_ids` with `-100` (not 0) ensures padding tokens don't contribute to loss
- Padding `input_ids` with 0 is not ideal (model attends to padding as real tokens) but works in practice; proper fix requires attention masks
- For pretraining, no padding is needed (all sequences are exactly `max_seq_len`)
- Instruction fine-tuning uses same causal LM training loop, just with masked targets
- GPT-2 Large (774M) on A100 80GB OOMs at batch size 10 due to activation memory; need gradient accumulation + mixed precision

### Code written
- `src/llm_from_scratch/data/instruction.py` — Refactored to use `-100` ignore_index instead of mask tensor
- `src/llm_from_scratch/model/causallm.py` — Added `eos_token_id` parameter to `generate()`
- `src/llm_from_scratch/training/causallm.py` — Updated `generate()` with instruction prompts and EOS stopping
- `examples/training/instruction.py` — Instruction fine-tuning training script

### PLAN.md items completed
- [x] Instruction fine-tuning (Alpaca-style)

### Open questions
- Next: Training efficiency deep dive (gradient accumulation, mixed precision, length-grouped batching, gradient checkpointing)

---

## Session 19 — 2026-04-17 — Bug Fixes and Attention Masks

### What we covered
- Reviewed and fixed 8 issues from IssuesToWorkOn.md
- Implemented weight tying between token embeddings and lm_head in GPTForCausalLM
- Implemented attention mask support throughout the entire model stack
- Added unit tests for attention mask behavior

### Key learnings
- Weight tying: `lm_head.weight = embedding.token.weight` shares parameters, saving ~38M params for GPT-2 Small and ensuring consistency when fine-tuning from pretrained weights
- `max_token_value + 1` for vocab_size is fragile — should use `n_vocab` instead
- Attention masks must be threaded through: data pipeline → model forward → transformer blocks → multi-head attention → scaled dot product attention
- `nn.Sequential` doesn't support extra kwargs — must use `nn.ModuleList` with manual iteration
- Mask shape contract: `scaled_dot_product_attention` expects mask with same batch dims as q/k/v; `MultiHeadAttention` reshapes `(batch, seq_len)` → `(batch, 1, seq_len)` to match its `(batch, num_heads, seq_len, head_dim)` input
- `GPTForCausalLM.generate()` needs to grow the attention mask alongside token_ids during autoregressive decoding

### Code written
- Fixed `GPTForClassification.forward()` ndim==2 case
- Fixed `TiktokenTokenizer.vocab_size` to use `n_vocab`
- Cached tokenized samples in `DatasetForClassification.__init__`
- Vectorized confusion matrix computation in `GPTForClassificationTrainer.eval()`
- Documented `dropout=0.0` in `GPT.tiny()` docstring
- Added shuffle support to `StreamingLLMDataset`
- Added weight tying in `GPTForCausalLM` (`lm_head.weight = embedding.token.weight`)
- Added `attn_mask` parameter: `scaled_dot_product_attention`, `SingleHeadAttention`, `MultiHeadAttention`, `TransformerBlock`, `GPT`, `GPTForCausalLM`, `GPTForClassification`
- Replaced `nn.Sequential` with `nn.ModuleList` for transformer blocks
- Added attention mask creation in classification and instruction data pipelines
- Updated trainers to thread attention masks
- Added `attn_mask` to `GPTForCausalLM.generate()` with mask growing logic
- Added 5 unit tests for attention mask behavior (blocking positions, batch masks, causal+mask, row sums, no-mask equivalence)

### PLAN.md items completed
- [x] Weight tying between token embeddings and lm_head (issue 3)
- [x] Attention mask support for padded tokens (issue 2)
