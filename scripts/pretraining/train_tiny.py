import argparse
import os

import requests
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import LLMDataset
from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.base import Tokenizer
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.causallm import GPTForCausalLMTrainer
from llm_from_scratch.utils import get_device

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT-tiny on Tiny Shakespeare (for local testing)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


def download_tiny_shakespeare(save_path: str) -> str:
    response = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return save_path


def get_dataloader(
    tokenizer: Tokenizer,
    max_seq_len: int = 256,
    stride: int = 128,
    batch_size: int = 4,
):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")
    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare...")
        download_tiny_shakespeare(filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"Found existing file at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset size: {len(text):,} characters")

    dataset = LLMDataset(tokenizer, text, max_seq_len, stride)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:  # type: ignore
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = TiktokenTokenizer()
    dl = get_dataloader(tokenizer, args.max_seq_len, args.stride, args.batch_size)

    model = GPTForCausalLM.tiny(tokenizer.vocab_size, args.max_seq_len)
    model.apply(init_weights)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: tiny, Parameters: {num_params:,}")

    optim = AdamW(model.parameters(), weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()

    trainer = GPTForCausalLMTrainer(
        model, tokenizer, optim, loss_fn, args.epochs, args.lr, dl, device
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train(args)
