# Issues & Polish Items

## Critical Bugs

### 1. `GPTForClassification.forward()` ndim==2 case is wrong ✅ Fixed

```python
if out.ndim == 2:
    logits = self.cls_head(out[:, :])  # takes ALL tokens, not last
```

For a 2D input (no batch dim), this takes all tokens instead of the last token's representation. Should be `self.cls_head(out[-1, :])` to match the `ndim == 3` case which uses `out[:, -1, :]`.

**Fixed:** Changed to `self.cls_head(out[-1, :])` in `classification.py`.

### 2. No attention mask support — padded tokens corrupt attention

Neither the attention module nor the model supports attention masks (padding masks). This means:
- Classification and instruction fine-tuning with padded batches treats padding tokens as real tokens.
- `CrossEntropyLoss(ignore_index=-100)` masks the *loss* but not the *attention* — the model still attends to padding.
- This is a correctness issue for all fine-tuning paths with variable-length sequences.

## Architecture Issues

### 3. No weight tying between token embeddings and `lm_head`

GPT-2 ties `wte` and `lm_head` weights. Your `GPTForCausalLM` has separate parameters:

```python
self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
```

`load_weights()` copies `lm_head.weight` separately from `wte.weight`, so they'll diverge during fine-tuning. This wastes ~768 × 50257 ≈ 38M parameters and causes instability when fine-tuning from pretrained weights.

**Fix:** Add `self.lm_head.weight = self.embedding.token.weight` after `lm_head` creation, or use a flag to enable weight tying.

## Medium Issues

### 4. `TiktokenTokenizer.vocab_size` is fragile ✅ Fixed

```python
@property
def vocab_size(self):
    return self._encoding.max_token_value + 1
```

`max_token_value + 1` happens to give 50257 for `gpt2`, but only because GPT-2's tokens are contiguous. For other encodings with gaps in token IDs, this will be wrong.

**Fix:** Use `self._encoding.n_vocab` instead. **Done** — updated in `tiktoken_adapter.py`.

### 5. `DatasetForClassification` tokenizes twice ✅ Fixed

```python
# In __init__ (filtering):
self.samples = [sample for sample in hf_dataset if len(self.tokenizer.encode(...)) <= max_text_len]

# In __getitem__:
tokens = self.tokenizer.encode(text)  # tokenizes again
```

Each sample is tokenized once during filtering and again during access. For large datasets, this is wasteful.

**Fix:** Cache tokenized results during `__init__`, or at minimum cache during `__getitem__` with `functools.lru_cache` or similar. **Done** — now caches `(tokens, label)` tuples during `__init__`.

### 6. Classification evaluation computes metrics manually instead of using PyTorch/torchmetrics ✅ Fixed

`GPTForClassificationTrainer.eval()` manually computes TP, TN, FP, FN with a Python loop. This is slow and error-prone.

**Fix:** Use `torchmetrics` or at least vectorized tensor operations (`(preds == labels).sum()`, etc.). **Done** — replaced Python loop with vectorized tensor ops.

### 7. `GPT.tiny()` uses `dropout=0.0` but `GPT.small/medium/large()` use `dropout=0.1` ✅ Fixed

The `tiny()` factory disables dropout while others don't. This is fine for testing but should be documented. **Done** — added docstring noting dropout=0.0 for deterministic tests.

### 8. `StreamingLLMDataset` doesn't shuffle ✅ Fixed

The streaming dataset yields samples in order from the HuggingFace dataset. For training, this means the model sees data in a fixed order each epoch, which can hurt convergence.

**Fix:** Add optional shuffling via `hf_dataset.shuffle()` or buffer-based shuffling. **Done** — added `shuffle` and `shuffle_buffer_size` params.
