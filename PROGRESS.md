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

---

## Retroactive Entry — 2026-04-08 to 2026-04-15 — Cloud Infra, Refactoring & Training Efficiency

### What we covered
- Created `StreamingLLMDataset` (PyTorch `IterableDataset`) for lazy tokenization of HuggingFace streaming datasets, avoiding loading all tokens into memory
- Added periodic sample generation during training (configurable prompt, frequency, and token count)
- Refactored the model hierarchy: split `GPT` into a base class (returns hidden states, no LM head) and `GPTForCausalLM(GPT)` / `GPTForClassification(GPT)` subclasses
- Restructured the repo: moved `examples/` out of `src/`, created `scripts/` for training runs, added `argparse`-based CLI for all training scripts, added `get_device()` utility
- Extracted `GPTTrainer[M]` generic base class with template method pattern, eliminating duplicated training logic between `GPTForCausalLMTrainer` and `GPTForClassificationTrainer`
- Added gradient accumulation support (`grad_accml_steps`) to `GPTForCausalLMTrainer` — loss divided by accumulation steps, optimizer only steps after accumulating enough micro-batches
- Added mixed precision training (bfloat16 via `torch.autocast`) to `GPTForCausalLMTrainer`
- Added cloud training infrastructure: `train.sh` launcher, PyTorch CUDA 12.8 index in `pyproject.toml`, `setup_vast_pod.sh` and `sync.sh` for vast.ai

### Key learnings
- `IterableDataset` + `DataLoader(num_workers=0)` is the pattern for streaming datasets that can't fit in memory
- Gradient accumulation: divide loss by `grad_accml_steps` before `.backward()`, only call `optim.step()` + `zero_grad()` every N micro-batches — effectively simulates larger batch sizes
- Mixed precision with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` reduces memory usage and speeds up training on modern GPUs
- Template method pattern with `Generic`/`TypeVar` lets Python trainers share common logic while subclass hooks (`_on_log_step`, `_on_train_step_end`) customize behavior

### Code written
- `src/llm_from_scratch/data/dataset.py` — `StreamingLLMDataset` class with lazy tokenization and token buffer
- `src/llm_from_scratch/model/base.py` — `GPT` base class (refactored from `transformer.py`)
- `src/llm_from_scratch/model/causallm.py` — `GPTForCausalLM(GPT)` with `lm_head` and `generate()`
- `src/llm_from_scratch/model/classification.py` — `GPTForClassification(GPT)` with classification head
- `src/llm_from_scratch/training/base.py` — `GPTTrainer[M]` base class with LR schedule, gradient accumulation, mixed precision, checkpointing
- `src/llm_from_scratch/training/causallm.py` — `GPTForCausalLMTrainer` refactored to extend `GPTTrainer`
- `src/llm_from_scratch/training/classification.py` — `GPTForClassificationTrainer` refactored to extend `GPTTrainer`
- `src/llm_from_scratch/utils.py` — `get_device()` utility
- `scripts/` — Training scripts with `argparse` CLI (pretraining, finetuning, evaluation)
- `setup_vast_pod.sh`, `sync.sh`, `.syncignore` — vast.ai infrastructure
- `train.sh` — Cloud training launcher
- `pyproject.toml` — PyTorch CUDA 12.8 wheel index

### PLAN.md items completed
- [x] Gradient accumulation
- [x] Mixed precision training

### Open questions
- Gradient checkpointing still TODO
- Length-grouped batching still TODO

---

## Session 20 — 2026-04-19 — Length-Grouped Batching

### What we covered
- Implemented `MaxTokenCountBatchSampler` — sorts dataset by sequence length (with noise for randomization), groups similar-length items into batches, dynamically sizes batches to fit within a token budget
- Batch-level shuffling ensures the model doesn't see all short sequences before all long ones
- Per-batch shuffling randomizes item order within each batch
- Refactored `create_dataloader` to support either `batch_size` (fixed) or `max_tokens_per_batch` (dynamic)
- Added attention mask creation in collate function (`padded_input_ids != 0`)
- Wired attention masks through the entire pipeline: collate_fn → train_step → model → attention

### Key learnings
- Length-grouped batching reduces padding waste by grouping similar-length sequences in the same batch
- Without grouping, `[900, 50, 50, 50]` gets padded to 4×900 = 3600 tokens; with grouping, short and long sequences go into separate batches
- By itself, length-grouping doesn't fix OOM — it helps with speed and predictable memory usage
- Dynamic batch sizing (token budget) is what actually fixes OOM: fewer long sequences per batch, more short ones
- Adding random noise to lengths before sorting ensures same-length items aren't always in the same order
- Shuffling batch order prevents the model from seeing a length-biased data stream
- Attention masks tell the model to ignore padding tokens, preventing them from influencing hidden states

### Code written
- `src/llm_from_scratch/data/instruction.py` — `MaxTokenCountBatchSampler`, `_pad_sequences_fn` with attention masks, refactored `create_dataloader` with dispatch
- `src/llm_from_scratch/model/causallm.py` — `generate()` now accepts and grows `attn_mask`, zeros out tokens after EOS
- `src/llm_from_scratch/training/causallm.py` — `train_step` now accepts and passes `attention_mask`, batched generation in `_test_model`
- `scripts/finetuning/instruction.py` — Updated CLI with `--max_tokens_per_batch`

### PLAN.md items completed
- [x] Length-grouped batching with dynamic batch sizing

### Open questions
- Next: Gradient checkpointing

---

## Session 21 — 2026-04-19 — Gradient Checkpointing

### What we covered
- Implemented gradient checkpointing in `GPT.forward()` using `torch.utils.checkpoint.checkpoint()`
- Added `use_gradient_checkpointing` flag to `GPT.__init__()` that toggles checkpointing per transformer block
- Threaded `use_gradient_checkpointing` through `load_pretrained_lm` and `load_pretrained_cls`
- Measured memory reduction: 73GB → 33GB (~55% reduction) on GPT-2 Large

### Key learnings
- Gradient checkpointing stores activations only at checkpoint boundaries and recomputes them during backprop — trades ~30% more compute for massive memory savings
- `torch.utils.checkpoint.checkpoint(block, out, attn_mask, use_reentrant=False)` wraps each TransformerBlock; `use_reentrant=False` is the recommended modern mode
- Works on any device (CPU, MPS, CUDA) — no GPU-specific features required
- With 36 layers in GPT-2 Large, checkpointing every block gives maximum memory savings; grouping blocks (every 2-3) is possible but rarely worth the complexity

### Code written
- `src/llm_from_scratch/model/base.py` — Added `use_gradient_checkpointing` flag, `checkpoint()` call in forward loop
- `src/llm_from_scratch/model/pretrained.py` — Threaded `use_gradient_checkpointing` through `load_pretrained_lm` and `load_pretrained_cls`

### PLAN.md items completed
- [x] Gradient checkpointing

### Open questions
- Next: LoRA / QLoRA deep dive

---

## Session 22 — 2026-04-20 — LoRA

### What we covered
- Implemented `LoRALayer(nn.Module)` — low-rank adaptation layer that freezes original linear weights and adds trainable A/B matrices
- Implemented `lorafy()` methods on `GPT`, `TransformerBlock`, `FeedForward`, and `MultiHeadAttention` to apply LoRA to attention projections (W_q, W_k, W_v, W_o) and optionally FFN layers
- Verified that LoRA reduces trainable params from 779.9M to 5.8M on GPT-2 Large
- Measured ~25% speed improvement (from reduced optimizer overhead), with memory savings from eliminating gradient and optimizer state storage for frozen params

### Key learnings
- LoRA replaces full weight updates with low-rank decomposition: W' = W + BA, where B is (out_features, r) and A is (r, in_features)
- A is initialized with small Gaussian (σ=0.02), B is initialized with zeros — ensures BA=0 at start, so model begins identical to pretrained
- Scaling factor α/r controls the magnitude of the LoRA update, decoupling learning rate from rank
- With LoRA, everything is frozen except LoRA A/B matrices — including layers without LoRA adapters
- LoRA's main benefit is memory savings (~8.6GB for optimizer states on GPT-2 Large), not speed — forward/backward passes still go through all frozen weights
- Combined LoRA + gradient checkpointing gives maximum memory savings

### Code written
- `src/llm_from_scratch/model/lora.py` — `LoRALayer` class with `forward()`, `requires_grad_` override
- `src/llm_from_scratch/attention/scaled_dot_product.py` — Added `lorafy()` method and LoRA-wrapped projections in `MultiHeadAttention`
- `src/llm_from_scratch/model/base.py` — Added `lorafy()` to `FeedForward`, `TransformerBlock`, and `GPT`; freezes embeddings, LN, and FF layers

### PLAN.md items completed
- [x] **Deep dive: LoRA / QLoRA** — implement parameter-efficient fine-tuning

### Open questions
- Next: Evaluate fine-tuned models

---

## Session 23 — 2026-04-21 — RMSNorm

### What we covered
- Implemented `LayerNorm` and `RMSNorm` from scratch in `norm.py`, with float32 upcast for mixed precision stability
- Added `use_rms_norm` flag to `GPT`, `TransformerBlock`, `GPTForCausalLM`, `GPTForClassification`
- Swapped all `nn.LayerNorm` for `RMSNorm` when flag is enabled (ln1, ln2 in each block + final ln)
- Benchmarked RMSNorm vs LayerNorm during pretraining: ~1.5% speed difference (negligible, as expected — compute is dominated by matmuls)

### Key learnings
- RMSNorm removes mean subtraction and bias: `x / sqrt(mean(x²) + ε) * γ` — only re-scaling matters, not re-centering
- Mixed precision stability: must upcast to float32 before computing mean/variance/rms, then cast back to original dtype before the affine transform
- Conv1D transpose (HuggingFace stores weights as [in, out], we use [out, in])
- Speed difference is negligible because normalization layers are a tiny fraction of total compute in large models
- The real motivation for RMSNorm: simpler, fewer parameters, and used by all modern LLMs (LLaMA, Mistral, Gemma)

### Code written
- `src/llm_from_scratch/model/norm.py` — `LayerNorm` and `RMSNorm` implementations with float32 upcasting
- `src/llm_from_scratch/model/base.py` — Added `use_rms_norm` flag, conditional `RMSNorm` vs `nn.LayerNorm`
- `src/llm_from_scratch/model/causallm.py` — Threaded `use_rms_norm` through `GPTForCausalLM`
- `src/llm_from_scratch/model/classification.py` — Threaded `use_rms_norm` through `GPTForClassification`
- `scripts/pretraining/train.py` — Added `--use_rms_norm` CLI flag
- `examples/model/norm.py` — Example/test script for normalization layers

### PLAN.md items completed
- [x] **RMSNorm** — Replace LayerNorm with RMSNorm, understand why

### Open questions
- Next: SwiGLU

---

## Session 24 — 2026-04-22 — RoPE (Rotary Positional Embeddings)

### What we covered
- Implemented `RotaryEmbedding` module that precomputes cos/sin tables for all positions up to `max_seq_len`
- Implemented `apply_rotary_emb` function applying 2D rotation to pairs of dimensions in Q/K
- Wired RoPE into `MultiHeadAttention.forward()` — applied after reshaping Q/K into head format, before scaled_dot_product_attention
- Modified `GPTEmbeddings` to skip positional embeddings when `use_rope=True`
- Added `use_rope` flag through the entire model chain (GPT, TransformerBlock, MultiHeadAttention, factory methods)
- Fixed bug: `apply_rotary_emb` returns a new tensor, must assign `q = apply_rotary_emb(q, ...)` not just call it

### Key learnings
- RoPE encodes position by rotating pairs of dimensions: angle `m * θ_i` where `θ_i = 1 / (base^(2i/d))`
- The rotate-and-interleave trick: construct `(-x_2i+1, x_2i)` then `result = x * cos + rotated * sin` — avoids explicit pair handling
- RoPE is applied only to Q and K, never V — position information enters through attention scores
- With RoPE, `q_m · k_n` depends only on `(m - n)`, giving relative position encoding for free
- `nn.Buffer` (PyTorch 2.5+) is cleaner than `register_buffer` for non-parameter tensors that should move with `.to(device)`
- `repeat_interleave(repeats=2)` interleaves the half-dimension cos/sin to match the paired dimension layout

### Code written
- `src/llm_from_scratch/model/rope.py` — `RotaryEmbedding` class and `apply_rotary_emb` function
- `src/llm_from_scratch/attention/scaled_dot_product.py` — Added `use_rope` and `max_seq_len` params to `MultiHeadAttention`, RoPE application in forward
- `src/llm_from_scratch/model/embeddings.py` — Conditional positional embedding (skipped when `use_rope=True`)
- `src/llm_from_scratch/model/base.py` — Added `use_rope` flag to `GPT`, `TransformerBlock`, all factory methods
- `src/llm_from_scratch/model/causallm.py` — Threaded `use_rope` through `GPTForCausalLM`
- `src/llm_from_scratch/model/classification.py` — Threaded `use_rope` through `GPTForClassification`

### PLAN.md items completed
- [x] **RoPE** — Replace absolute positional embeddings with Rotary Positional Embeddings

### Open questions
- Next: GQA (Grouped Query Attention)

---

## Session 25 — 2026-04-23 — SwiGLU

### What we covered
- Implemented `FeedForwardSwiGLU` as a separate class alongside the existing `FeedForward`
- SwiGLU: `W_down(W_gate(x) ⊙ SiLU(W_up(x)))` — 3 linear layers instead of 2, with gating
- Added `use_swiglu` flag through the entire model chain
- LoRA support for all 3 SwiGLU projections (W_gate, W_up, W_down)

### Key learnings
- SwiGLU replaces GELU FFN with a gated architecture: gate controls which features pass through
- SiLU(x) = x * sigmoid(x), also called "swish" — hence Swi**GLU** (Swish-Gated Linear Unit)
- 3 linear layers instead of 2, but gating makes FFN more expressive; empirically matches GELU with fewer total parameters
- LLaMA uses `ffn_dim = int(2 * 4/3 * embed_dim)` to compensate for the extra gate projection; we keep `4 * embed_dim` for simplicity

### Code written
- `src/llm_from_scratch/model/base.py` — `FeedForwardSwiGLU` class with LoRA support, `use_swiglu` flag in `TransformerBlock`, `GPT`, all factory methods

### PLAN.md items completed
- [x] **SwiGLU** — Replace GELU FFN with SwiGLU activation

### Open questions
- Next: KV Cache

---

## Session 26 — 2026-04-23 — GQA (Grouped Query Attention)

### What we covered
- Implemented GQA in `MultiHeadAttention` with `num_kv_heads` parameter (defaults to `num_heads` for MHA backward compatibility)
- K/V projections reduced from `embed_dim → embed_dim` to `embed_dim → num_kv_heads * head_dim`
- KV heads expanded via `repeat_interleave(group_size, dim=1)` to match Q head count before attention computation
- Threaded `num_kv_threads` through `TransformerBlock`, `GPT`, `GPTForCausalLM`, `GPTForClassification`, all factory methods
- Fixed `GPTForCausalLM` and `GPTForClassification` missing `use_swiglu` and `num_kv_threads` params

### Key learnings
- GQA shares KV heads across groups of Q heads: `group_size = num_heads / num_kv_heads`
- Reduces KV cache size at inference time — dominant memory bottleneck in autoregressive generation
- `repeat_interleave` copies data; `.expand` creates a zero-copy view but requires `.contiguous()` before some ops — used `repeat_interleave` for clarity
- RoPE applied after KV expansion (standard approach, avoids subtle issues)
- Special cases: `num_kv_heads = num_heads` → MHA, `num_kv_heads = 1` → MQA (Multi-Query Attention)
- LLaMA-2 70B uses GQA (8 KV heads for 64 Q heads), LLaMA-1 uses MHA

### Code written
- `src/llm_from_scratch/attention/scaled_dot_product.py` — Added `num_kv_heads` param to `MultiHeadAttention`, smaller K/V projections, `repeat_interleave` expansion
- `src/llm_from_scratch/model/base.py` — Added `num_kv_threads` to `TransformerBlock`, `GPT`, factory methods
- `src/llm_from_scratch/model/causallm.py` — Threaded `use_swiglu` and `num_kv_threads`
- `src/llm_from_scratch/model/classification.py` — Threaded `use_swiglu` and `num_kv_threads`

### PLAN.md items completed
- [x] **Grouped Query Attention (GQA)** — Implement KV head sharing

### Open questions
- Next: KV Cache
