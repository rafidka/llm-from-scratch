# Scripts

Training, evaluation, and finetuning scripts. All scripts accept CLI arguments — use `--help` to see available options.

---

## 1. Pretraining

### `pretraining/train.py`

Trains GPT-2 on Wikitext-103 (streaming). Supports `small` (124M) and `medium` (355M) model sizes with cosine LR schedule, checkpointing, and periodic sample generation. Designed for cloud GPUs.

```bash
uv run python scripts/pretraining/train.py
uv run python scripts/pretraining/train.py --model_size medium --epochs 2 --lr 5e-4
uv run python scripts/pretraining/train.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--model_size` | `small` | Model size (`small`, `medium`) |
| `--epochs` | `1` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-4` | Peak learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_seq_len` | `1024` | Maximum sequence length |
| `--stride` | `512` | Sliding window stride |
| `--warmup_ratio` | `0.1` | Fraction of steps for LR warmup |
| `--checkpoint_dir` | `checkpoints` | Directory to save checkpoints |
| `--checkpoint_every` | `1000` | Save checkpoint every N steps |
| `--log_every` | `100` | Log loss every N steps |
| `--generate_every` | `100` | Generate sample text every N steps |
| `--generation_prompt` | `I am going to the bank to` | Prompt for sample generation |
| `--generation_tokens` | `100` | Tokens to generate in samples |

### `pretraining/train_tiny.py`

Trains a tiny GPT model (~50K params) on Tiny Shakespeare. Downloads the dataset automatically. Intended for local testing and quick iteration.

```bash
uv run python scripts/pretraining/train_tiny.py
uv run python scripts/pretraining/train_tiny.py --epochs 5 --lr 1e-3
uv run python scripts/pretraining/train_tiny.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `3e-4` | Peak learning rate |
| `--max_seq_len` | `256` | Maximum sequence length |
| `--stride` | `128` | Sliding window stride |
| `--batch_size` | `4` | Batch size |
| `--weight_decay` | `0.01` | AdamW weight decay |

### `pretraining/evaluation.py`

Evaluates perplexity of a pretrained GPT-2 model on Wikitext-103 validation data. Perplexity = exp(cross-entropy loss), lower is better. Supports all GPT-2 sizes and can limit evaluation steps.

```bash
uv run python scripts/pretraining/evaluation.py
uv run python scripts/pretraining/evaluation.py --model_size gpt2-medium --max_steps 500
uv run python scripts/pretraining/evaluation.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--model_size` | `gpt2` | Pretrained model (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) |
| `--max_seq_len` | `1024` | Maximum sequence length |
| `--stride` | `64` | Sliding window stride |
| `--batch_size` | `4` | Batch size |
| `--max_steps` | `None` | Limit evaluation steps (default: full dataset) |

---

## 2. Finetuning

### `finetuning/classification.py`

Finetunes a pretrained GPT-2 model for text classification on the IMDB dataset (binary sentiment). Freezes the transformer backbone and trains a classification head. Periodically prints sample predictions and evaluation metrics (accuracy, precision, recall, F1).

```bash
uv run python scripts/finetuning/classification.py
uv run python scripts/finetuning/classification.py --base_model gpt2-medium --epochs 5 --lr 3e-5
uv run python scripts/finetuning/classification.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--base_model` | `gpt2` | Base model (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) |
| `--epochs` | `3` | Number of training epochs |
| `--lr` | `5e-5` | Peak learning rate |
| `--batch_size` | `4` | Batch size |
| `--max_seq_len` | `1024` | Maximum sequence length |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--num_classes` | `2` | Number of output classes |

### `finetuning/instruction.py`

Finetunes a pretrained GPT-2 model for instruction following on the Alpaca dataset. Uses gradient accumulation and optional mixed precision (CUDA only). Periodically generates sample responses to evaluation prompts.

```bash
uv run python scripts/finetuning/instruction.py
uv run python scripts/finetuning/instruction.py --base_model gpt2-medium --epochs 3 --lr 3e-5
uv run python scripts/finetuning/instruction.py --use_mixed_precision
uv run python scripts/finetuning/instruction.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--base_model` | `gpt2-large` | Base model (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `5e-5` | Peak learning rate |
| `--batch_size` | `8` | Batch size |
| `--max_seq_len` | `1024` | Maximum sequence length |
| `--stride` | `512` | Sliding window stride |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--grad_accml_steps` | `5` | Gradient accumulation steps |
| `--use_mixed_precision` | `False` | Enable mixed precision (requires CUDA) |