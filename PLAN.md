# LLM From Scratch — Project Plan

Building a large language model from scratch, following Sebastian Raschka's book as a
foundation while incorporating deep dives into modern techniques and architectures.

**Environment**: Python 3.12, uv, PyTorch. Development/testing on MacBook Pro (MPS),
heavy training offloaded to cloud GPUs.

**Total estimated effort**: ~30-45 sessions

---

## Phase 1: Tokenization

**Effort: 2-3 sessions**

- [x] Understand the tokenization spectrum: character → word → subword (BPE)
- [x] Build a simple tokenizer from scratch (vocabulary building, encode/decode)
- [x] **Deep dive**: Implement BPE merge algorithm by hand
- [x] Use `tiktoken` with GPT-2's tokenizer (`cl100k_base` / `gpt2` encoding)
- [x] Build a `Dataset` and `DataLoader` for next-token prediction (sliding window)

**Key concepts**: special tokens, vocabulary size tradeoffs, byte-level fallback

---

## Phase 2: Attention Mechanism

**Effort: 3-4 sessions**

- [x] Implement simplified self-attention (dot product attention)
- [x] Add trainable weights → single-head attention
- [x] Implement causal (masked) self-attention
- [x] Extend to multi-head attention
- [x] **Deep dive**: Attention as soft dictionary lookup, attention pattern visualization
- [x] **Deep dive**: Computational complexity of attention (O(n²) and why it matters)

---

## Phase 3: The GPT Architecture (GPT-2 Baseline)

**Effort: 3-4 sessions**

- [x] Token embeddings + absolute positional embeddings
- [x] LayerNorm and residual connections
- [x] Feed-forward network (with GELU activation)
- [x] Assemble the Transformer block
- [x] Stack blocks into full GPT-2 architecture
- [x] Text generation (greedy, temperature, top-k, top-p sampling)
- [x] **Deep dive**: Why residual connections are critical (gradient flow)
- [x] **Deep dive**: Pre-norm vs post-norm architecture

---

## Phase 4: Pretraining

**Effort: 3-5 sessions**

- [x] Cross-entropy loss for next-token prediction
- [x] Training loop with AdamW optimizer
- [x] Learning rate scheduling (warmup + cosine decay)
- [x] Training on a small corpus locally (MacBook MPS)
- [x] Evaluate with sample generation
- [x] Train a larger model on cloud GPU (in progress on RTX 3090)
- [x] **Deep dive**: Gradient accumulation for effective larger batch sizes
- [x] **Deep dive**: Mixed precision training

---

## Phase 5: Loading Pretrained Weights (GPT-2)

**Effort: 1-2 sessions**

- [x] Understand GPT-2 weight layout from OpenAI/HuggingFace
- [x] Map weights to our architecture
- [x] Verify correctness via text generation
- [x] Compare our outputs to HuggingFace's GPT-2
- [x] Evaluate perplexity on held-out data

---

## Phase 6: Fine-tuning

**Effort: 2-3 sessions**

- [x] Classification fine-tuning (e.g., spam detection or sentiment)
- [x] Instruction fine-tuning (Alpaca-style)
- [x] **Deep dive: Training efficiency** — gradient accumulation, mixed precision (bf16), length-grouped batching
- [x] Gradient checkpointing
- [x] **Deep dive: LoRA / QLoRA** — implement parameter-efficient fine-tuning
- [ ] Evaluate fine-tuned models

---

## Phase 7: Modern Architecture Deep Dives

**Effort: 5-8 sessions**

Evolving our GPT-2 toward a modern LLM (LLaMA-style):

- [x] **RMSNorm** — Replace LayerNorm with RMSNorm, understand why
- [x] **RoPE** — Replace absolute positional embeddings with Rotary Positional Embeddings
- [x] **SwiGLU** — Replace GELU FFN with SwiGLU activation
- [x] **Grouped Query Attention (GQA)** — Implement KV head sharing
- [x] **KV Cache** — Implement efficient autoregressive inference
- [x] **Flash Attention** — Understand the IO-aware algorithm (implement simplified version,
      use the real thing via PyTorch `scaled_dot_product_attention`)

---

## Phase 8: Alignment

**Effort: 3-4 sessions**

Aligning our model to follow instructions and preferences:

- [ ] Understand the alignment problem and the RLHF pipeline
- [ ] **Reward model training** — Train a model to score completions
- [ ] **DPO (Direct Preference Optimization)** — Simpler alternative to full RLHF; implement preference-based fine-tuning
- [ ] Implement alignment fine-tuning at small scale on our model

---

## Phase 9: Quantization

**Effort: 2-3 sessions**

Reducing model size and inference cost:

- [ ] **Post-training quantization (PTQ)** — int8 and int4 weight quantization
- [ ] **GPTQ / AWQ** — Weight-only quantization for efficient inference
- [ ] **Quantization-aware training (QAT)** — Train with quantization in the loop
- [ ] **KV cache quantization** — Compress cached K/V for longer contexts

---

## Phase 10: Practical Applications

**Effort: 3-4 sessions**

Fine-tuning production open-source models:

- [ ] **Qwen 3** — Understand architecture, load pretrained weights, fine-tune with LoRA/QLoRA
- [ ] **Gemini** — Understand architecture, fine-tune for specific tasks
- [ ] **Chat templates & instruction formatting** — Proper prompt formatting for chat models
- [ ] **Evaluation** — Benchmark fine-tuned models on downstream tasks

---

## Phase 11: Modern Innovations

**Effort: 3-5 sessions**

Modern architectural innovations beyond the GPT-2 baseline:

- [ ] **Mixture of Experts (MoE)** — Implement sparse MoE layer with top-k routing; understand load balancing and capacity factors
- [ ] **Multi-head Latent Attention (MLA)** — DeepSeek's compressed KV cache via low-rank projections
- [ ] **Mixture-of-Depths (MoD)** — Dynamically skip layers per token; not every token needs full computation
- [ ] **Differential Attention** — Dual softmax heads with subtractive gating to reduce attention noise

---

## Phase 12: Efficient Attention Kernels

**Effort: 4-5 sessions**

Low-level optimizations for attention compute and memory:

- [ ] **Sliding Window Attention** — Proper block-sparse implementation (not just masking); KV cache truncation to window_size
- [ ] **Linear Attention** — Kernel trick for O(n) complexity; Performer/Linear Transformer approach
- [ ] **Paged Attention** — vLLM-style KV cache memory management with non-contiguous blocks
- [ ] **Speculative Decoding** — Draft-then-verify with a small model; leverage KV cache for verification
- [ ] **Custom Triton kernel** — Write a minimal GPU attention kernel to understand kernel-level programming

---

## Phase 13: Multi-GPU Training & Inference

**Effort: 3-4 sessions**

Scaling training and inference across multiple GPUs:

- [ ] **Data Parallelism** — DDP, FSDP, and gradient accumulation
- [ ] **Tensor Parallelism** — Megatron-style layer splitting across GPUs
- [ ] **Pipeline Parallelism** — Split model stages across GPUs
- [ ] **Distributed training** — Run training on 2+ GPUs end-to-end

---

## Phase 14: Distributed & Long-Context Attention

**Effort: 2-3 sessions**

Scaling beyond single-GPU memory for long sequences:

- [ ] **Ring Attention** — Distribute sequences across GPUs with overlapping compute/communication
- [ ] **Sequence Parallelism** — Distribute sequence dimension across GPUs
- [ ] **Continuous batching** — Inference serving with dynamic batch management