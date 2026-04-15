import argparse
from pathlib import Path

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import StreamingLLMDataset
from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.causallm import GPTForCausalLMTrainer
from llm_from_scratch.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 on cloud GPU")
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of passes through the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--generation_prompt",
        type=str,
        default="I am going to the bank to",
        help="Prompt for sample generation",
    )
    parser.add_argument(
        "--generation_tokens",
        type=int,
        default=100,
        help="Number of tokens to generate in samples",
    )
    return parser.parse_args()


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:  # type: ignore
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_model(model_size: str, vocab_size: int, max_seq_len: int) -> GPTForCausalLM:
    if model_size == "tiny":
        return GPTForCausalLM.tiny(vocab_size, max_seq_len)
    elif model_size == "small":
        return GPTForCausalLM.small(vocab_size, max_seq_len)
    elif model_size == "medium":
        return GPTForCausalLM.medium(vocab_size, max_seq_len)
    elif model_size == "large":
        return GPTForCausalLM.large(vocab_size, max_seq_len)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


WIKITEXT_103_TOKENS_ESTIMATE = 100_000_000


def load_wikipedia_data(
    tokenizer: TiktokenTokenizer,
    max_seq_len: int,
    stride: int,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    print("Loading Wikitext-103 dataset (streaming)...")
    dataset = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )

    streaming_dataset = StreamingLLMDataset(dataset, tokenizer, max_seq_len, stride)

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = TiktokenTokenizer()
    model = create_model(args.model_size, tokenizer.vocab_size, args.max_seq_len)
    model.apply(init_weights)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size}, Parameters: {num_params:,}")

    dataloader = load_wikipedia_data(
        tokenizer, args.max_seq_len, args.stride, args.batch_size
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()

    total_steps = args.epochs * (
        WIKITEXT_103_TOKENS_ESTIMATE // args.stride // args.batch_size
    )

    trainer = GPTForCausalLMTrainer(
        model=model,
        tokenizer=tokenizer,
        optim=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        max_lr=args.lr,
        data_loader=dataloader,
        device=device,
        warmup_ratio=args.warmup_ratio,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        test_prompts=[args.generation_prompt],
        generation_tokens=args.generation_tokens,
        total_steps=total_steps,
    )

    print(f"Training for {args.epochs} epochs (~{trainer.total_steps} steps)")
    print(f"Warmup steps: {trainer.warmup_steps}, Batch size: {args.batch_size}")

    trainer.train()

    print("Training complete!")
    final_path = Path(args.checkpoint_dir) / "final_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
