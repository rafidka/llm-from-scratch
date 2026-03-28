# LLM From Scratch — Project Plan

Building a large language model from scratch, following Sebastian Raschka's book as a
foundation while incorporating deep dives into modern techniques and architectures.

**Environment**: Python 3.12, uv, PyTorch. Development/testing on MacBook Pro (MPS),
heavy training offloaded to cloud GPUs.

**Total estimated effort**: ~25-38 sessions

---

## Phase 1: Tokenization

**Effort: 2-3 sessions**

- [x] Understand the tokenization spectrum: character → word → subword (BPE)
- [x] Build a simple tokenizer from scratch (vocabulary building, encode/decode)
- [x] **Deep dive**: Implement BPE merge algorithm by hand
- [x] Use `tiktoken` with GPT-2's tokenizer (`cl100k_base` / `gpt2` encoding)
- [ ] Build a `Dataset` and `DataLoader` for next-token prediction (sliding window)

**Key concepts**: special tokens, vocabulary size tradeoffs, byte-level fallback

---

## Phase 2: Attention Mechanism

**Effort: 3-4 sessions**

- [ ] Implement simplified self-attention (dot product attention)
- [ ] Add trainable weights → single-head attention
- [ ] Implement causal (masked) self-attention
- [ ] Extend to multi-head attention
- [ ] **Deep dive**: Attention as soft dictionary lookup, attention pattern visualization
- [ ] **Deep dive**: Computational complexity of attention (O(n²) and why it matters)

---

## Phase 3: The GPT Architecture (GPT-2 Baseline)

**Effort: 3-4 sessions**

- [ ] Token embeddings + absolute positional embeddings
- [ ] LayerNorm and residual connections
- [ ] Feed-forward network (with GELU activation)
- [ ] Assemble the Transformer block
- [ ] Stack blocks into full GPT-2 architecture
- [ ] Text generation (greedy, temperature, top-k, top-p sampling)
- [ ] **Deep dive**: Why residual connections are critical (gradient flow)
- [ ] **Deep dive**: Pre-norm vs post-norm architecture

---

## Phase 4: Pretraining

**Effort: 3-5 sessions**

- [ ] Cross-entropy loss for next-token prediction
- [ ] Training loop with AdamW optimizer
- [ ] Learning rate scheduling (warmup + cosine decay)
- [ ] Training on a small corpus locally (MacBook MPS)
- [ ] Evaluate with perplexity and sample generation
- [ ] **Deep dive**: Gradient accumulation for effective larger batch sizes
- [ ] **Deep dive**: Mixed precision training

---

## Phase 5: Loading Pretrained Weights (GPT-2)

**Effort: 1-2 sessions**

- [ ] Understand GPT-2 weight layout from OpenAI/HuggingFace
- [ ] Map weights to our architecture
- [ ] Verify correctness via text generation
- [ ] Compare our outputs to HuggingFace's GPT-2

---

## Phase 6: Fine-tuning

**Effort: 2-3 sessions**

- [ ] Classification fine-tuning (e.g., spam detection or sentiment)
- [ ] Instruction fine-tuning (Alpaca-style)
- [ ] **Deep dive: LoRA / QLoRA** — implement parameter-efficient fine-tuning
- [ ] Evaluate fine-tuned models

---

## Phase 7: Modern Architecture Deep Dives

**Effort: 5-8 sessions**

Evolving our GPT-2 toward a modern LLM (LLaMA-style):

- [ ] **RMSNorm** — Replace LayerNorm with RMSNorm, understand why
- [ ] **RoPE** — Replace absolute positional embeddings with Rotary Positional Embeddings
- [ ] **SwiGLU** — Replace GELU FFN with SwiGLU activation
- [ ] **Grouped Query Attention (GQA)** — Implement KV head sharing
- [ ] **KV Cache** — Implement efficient autoregressive inference
- [ ] **Flash Attention** — Understand the IO-aware algorithm (implement simplified version,
      use the real thing via PyTorch `scaled_dot_product_attention`)

---

## Phase 8: DeepSeek Innovations

**Effort: 3-5 sessions**

- [ ] **Multi-head Latent Attention (MLA)** — DeepSeek's compressed KV cache approach
- [ ] **DeepSeekMoE** — Mixture of Experts with fine-grained expert segmentation
- [ ] **Auxiliary-loss-free load balancing** — How DeepSeek handles expert routing
- [ ] Study DeepSeek-V2/V3 architecture papers

---

## Phase 9: Alignment (RLHF / DPO)

**Effort: 3-4 sessions**

- [ ] Understand the alignment problem and RLHF pipeline
- [ ] Reward model training
- [ ] **Deep dive: DPO (Direct Preference Optimization)** — simpler alternative to full RLHF
- [ ] Implement preference-based fine-tuning on a small scale
