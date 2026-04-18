# Issues & Polish Items

## Critical Bugs

(DONE) ### 1. `train_cloud.py` imports from nonexistent module

`examples/training/train_cloud.py` imports `GPT` from `llm_from_scratch.model.transformer`, but that module doesn't exist. The `GPT` class lives in `llm_from_scratch.model.base`. This will crash at runtime.

**Fix:** Change import to `from llm_from_scratch.model.base import GPT` (or better, use `GPTForCausalLM`).

(INVALID) ### 2. `gradient_flow_comparison.py` has `loss.backback()` typo

Line ~108 in `examples/model/gradient_flow_comparison.py` has `loss.backback()` instead of `loss.backward()`. Will crash if run.

### 3. `GPTForClassification.forward()` ndim==2 case is wrong ✅ Fixed

```python
if out.ndim == 2:
    logits = self.cls_head(out[:, :])  # takes ALL tokens, not last
```

For a 2D input (no batch dim), this takes all tokens instead of the last token's representation. Should be `self.cls_head(out[-1, :])` to match the `ndim == 3` case which uses `out[:, -1, :]`.

### 4. `GPTForClassificationTrainer.get_lr()` returns 0 at step 0 ✅ Fixed

(Resolved by extracting GPTTrainer base class which uses 1-indexed steps.)

Unlike `GPTForCausalLMTrainer.get_lr()` which shifts to 1-indexed internally so step 0 gets a small but nonzero LR, the classification trainer's version computes:

```python
if step < self.warmup_steps:
    return self.max_lr * step / self.warmup_steps  # returns 0 when step=0
```

This means the first gradient update is effectively skipped.

**Fix:** Mirror the causal LM trainer's approach by converting to 1-indexed internally.

### 5. No attention mask support — padded tokens corrupt attention ✅ Fixed

Neither the attention module nor the model supports attention masks (padding masks). This means:
- Classification and instruction fine-tuning with padded batches treats padding tokens as real tokens.
- `CrossEntropyLoss(ignore_index=-100)` masks the *loss* but not the *attention* — the model still attends to padding.
- This is a correctness issue for all fine-tuning paths with variable-length sequences.

### 6. `train_cloud.py` uses base `GPT` but feeds hidden states to `CrossEntropyLoss` ✅ Fixed

(File renamed to train.py and now uses GPTForCausalLM.)

`GPT.forward()` returns hidden states (no `lm_head`), so `logits = model(input_ids)` in `train_cloud.py` returns `[batch, seq_len, embed_dim]` — not `[batch, seq_len, vocab_size]`. `CrossEntropyLoss` expects logits of shape `[*, vocab_size]`. This will crash or produce nonsense loss values.

**Fix:** Use `GPTForCausalLM` instead of `GPT`, which adds the `lm_head` projection.

---

## Architecture Issues

### 7. Duplicated training logic across 3+ trainers ✅ Fixed

(Resolved by extracting GPTTrainer base class.)

Three separate implementations of essentially the same training infrastructure:
- `GPTForCausalLMTrainer` (training/causallm.py)
- `GPTForClassificationTrainer` (training/classification.py)
- Standalone loop in `train_cloud.py`

All implement their own:
- `get_lr()` — three slightly different implementations (see issue #4)
- `init_weights()` — two separate copies
- Training loop boilerplate
- Checkpointing (only in `train_cloud.py`)

**Fix:** Extract a shared `LRScheduler`, `init_weights()`, and common trainer utilities. Consider a base `Trainer` class or at minimum shared tooling.

### 8. No weight tying between token embeddings and `lm_head` ✅ Fixed

GPT-2 ties `wte` and `lm_head` weights. Your `GPTForCausalLM` has separate parameters:

```python
self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
```

`load_weights()` copies `lm_head.weight` separately from `wte.weight`, so they'll diverge during fine-tuning. This wastes ~768 × 50257 ≈ 38M parameters and causes instability when fine-tuning from pretrained weights.

**Fix:** Add `self.lm_head.weight = self.embedding.token.weight` after `lm_head` creation, or use a flag to enable weight tying.

### 9. Missing module: `model/transformer.py` referenced by examples ✅ Fixed

(References updated to use model.base and model.causallm.)

Multiple examples and `examples/README.md` reference `src/llm_from_scratch/model/transformer.py`, but this file doesn't exist. The `GPT` class is in `model/base.py`, `GPTForCausalLM` is in `model/causallm.py`.

**Fix:** Either create a deprecated re-export shim at `model/transformer.py`, or update all references.

### 10. Reduced test coverage — only attention is tested

Only 11 tests exist, all for `attention/`. Zero tests for:
- `data/` (dataset, dataloader, instruction dataset, classification dataset)
- `model/` (GPT, embeddings, classification, causal LM, pretrained loading)
- `tokenizers/` (BPE, simple, tiktoken)
- `training/` (trainers, evaluation, LR scheduling)

For a polished repo, each module should have at least basic smoke tests verifying shapes, forward passes, encode/decode roundtrips, and LR schedule correctness.

---

## Medium Issues

### 11. `TiktokenTokenizer.vocab_size` is fragile

```python
@property
def vocab_size(self):
    return self._encoding.max_token_value + 1
```

`max_token_value + 1` happens to give 50257 for `gpt2`, but only because GPT-2's tokens are contiguous. For other encodings with gaps in token IDs, this will be wrong.

**Fix:** Use `self._encoding.n_vocab` instead.

### 12. `DatasetForClassification` tokenizes twice

```python
# In __init__ (filtering):
self.samples = [sample for sample in hf_dataset if len(self.tokenizer.encode(...)) <= max_text_len]

# In __getitem__:
tokens = self.tokenizer.encode(text)  # tokenizes again
```

Each sample is tokenized once during filtering and again during access. For large datasets, this is wasteful.

**Fix:** Cache tokenized results during `__init__`, or at minimum cache during `__getitem__` with `functools.lru_cache` or similar.

### 13. Classification evaluation computes metrics manually instead of using PyTorch/torchmetrics

`GPTForClassificationTrainer.eval()` manually computes TP, TN, FP, FN with a Python loop. This is slow and error-prone.

**Fix:** Use `torchmetrics` or at least vectorized tensor operations (`(preds == labels).sum()`, etc.).

### 14. `GPT.tiny()` uses `dropout=0.0` but `GPT.small/medium/large()` use `dropout=0.1`

The `tiny()` factory disables dropout while others don't. This is fine for testing but should be documented.

### 15. Inconsistent `device` handling in trainers ✅ Fixed

(Resolved by extracting GPTTrainer base class with consistent device handling.)

- `GPTForCausalLMTrainer` moves data to device inside `train_step()` — good
- `GPTForClassificationTrainer` moves data to device inside `train_epoch()` — also fine
- `train_cloud.py` moves data to device inside the training loop — inconsistent pattern
- `GPTForClassificationTrainer.eval()` doesn't move `input_ids`/`true_labels` to device before inference... wait, it does. But `all_true_labels` and `all_pred_labels` are concatenated on device, which is fine.
- `evaluate_perplexity()` calls `model.to(device)` which is redundant if model is already on device, and doesn't put it back on the original device after.

### 16. `StreamingLLMDataset` doesn't shuffle

The streaming dataset yields samples in order from the HuggingFace dataset. For training, this means the model sees data in a fixed order each epoch, which can hurt convergence.

**Fix:** Add optional shuffling via `hf_dataset.shuffle()` or buffer-based shuffling.

### 17. `GPTForCausalLM.generate()` doesn't use `torch.no_grad()` ✅ Fixed

(Added `@torch.no_grad()` decorator and return type annotation.)

The `generate()` method runs full forward passes with gradient tracking enabled. The trainer's `_test_model()` calls it inside `@torch.no_grad()`, but direct calls to `generate()` will build up computation graphs and OOM on long generation.

**Fix:** Wrap `generate()` internals in `torch.no_grad()`, or at minimum document that callers should use it.

### 18. `examples/training/causallm.py` and `evaluation.py` use relative imports

```python
from ..data.dataset_tiny_shakespeare import get_dataloader
```

This breaks when running `python -m examples.training.causallm`. Either use absolute imports or restructure examples as a proper package.

---

## Nitpicks & Polish

### 19. Empty `README.md`

The README is empty — for a polished repo, this needs at least a project description, setup instructions, and usage examples.

### 20. Stale/incorrect references in `examples/README.md`

References `src/llm_from_scratch/training/trainer.py` (doesn't exist — it's now `training/causallm.py`), `src/llm_from_scratch/model/transformer.py` (doesn't exist — now split into `base.py`, `causallm.py`, etc.), and `src/llm_from_scratch/attention/qkv.py` (doesn't exist — now `scaled_dot_product.py`).

### 21. Dead TODO in `data/loader.py`

```python
# TODO Get rid of this later.
def create_dataloader(dataset: LLMDataset, batch_size: int, shuffle: bool):
```

Either delete it or resolve the TODO. It's still referenced by `examples/data/dataset.py`.

### 22. `GPTEmbeddings` naming inconsistency ✅ Fixed

(Renamed `self.embedding` to `self.embeddings` to match class name.)

The class is `GPTEmbeddings` but the instance variable in `GPT` is `self.embedding` (singular). Pick one convention — either `self.embeddings` (matching class) or rename class to `GPTEmbedding` (matching instance).

### 23. Inconsistent comments in `GPT.base.forward()`

```python
return out  # todo what is the shape?
```

This TODO should be resolved. The shape is clearly `[batch, seq_len, embed_dim]`.

### 24. No `__all__` exports in any module

For API clarity, define `__all__` in each module to make the public interface explicit.

### 25. Example scripts mix with test-like files

- `examples/tokenizers/test_bpe_corpus.py` — a test script in `examples/`
- `examples/model/transformer_with_topk.py` and `transformer_without_topk.py` — superseded by `GPTForCausalLM.generate()`

Consider cleaning up or marking these as legacy.

### 26. `BPETokenizer` special token IDs are hardcoded

```python
UNK = 0
EOS = 1
# ...
token_id = 100  # Token 0 is UNK; 1 is EOS; 2 to 99 are left for custom tokens.
```

Token IDs 2–99 are reserved but never used. This is fine for a toy tokenizer but could be confusing.

### 27. `setup_vast_pod.sh` lists packages one per line without line continuation

```bash
sudo apt install -y
    bat \
    neovim \
```

This will fail because `apt install -y` will execute before the packages. Needs `\` after `apt install -y`.

### 28. `GPTForClassificationTrainer.sample_classification()` has hardcoded demo texts

```python
texts = [
    "This movie was absolutely fantastic!...",
    ...
]
```

These are only relevant for IMDB/sentiment. Should be configurable or in the example script, not in the library.

### 29. `GPTForCausalLMTrainer._test_model()` hardcodes generation params

```python
output_ids = self.model.generate(
    input_ids, max_new_tokens=256, temperature=0.2, top_k=40, ...
)
```

These should be configurable (e.g., `test_max_new_tokens`, `test_temperature`).

### 30. No checkpoint/save/resume support in the library trainers

Only `train_cloud.py` has checkpointing. `GPTForCausalLMTrainer` and `GPTForClassificationTrainer` don't save any checkpoints or support resuming training.

### 31. `.gitignore` excludes `.vscode/` but it's checked in

`.gitignore` has `.vscode/` but the `.vscode/` directory with `launch.json` and `settings.json` exists in the repo. This means `.vscode/` was committed before the gitignore rule was added, and git is still tracking it.

**Fix:** `git rm -r --cached .vscode/` to untrack it.

### 32. Factory methods don't match GPT-2 specs exactly

- `GPT.small()` — correct (768/12/12)
- `GPT.medium()` — correct (1024/16/24)
- `GPT.large()` — correct (1280/20/36)
- No `GPT.xl()` — GPT-2 XL (1600/25/48, 1.5B params) is missing
- The docstrings say "124M parameters", "355M parameters", "774M parameters" but these counts depend on vocab_size which is a parameter. The actual parameter count will differ.

### 33. `GPTForCausalLM.generate()` type annotation returns untyped `Tensor` ✅ Fixed

(Added `-> Tensor` return type annotation.)

The return type is not annotated. Should be `-> Tensor`.

### 34. `evaluate_perplexity()` prints progress inside the function

Logging with `print()` is fine for scripts but not for a library. Consider accepting a logger or returning results without side effects.