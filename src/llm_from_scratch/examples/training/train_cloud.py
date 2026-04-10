import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import StreamingLLMDataset
from llm_from_scratch.model.transformer import GPT
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 on cloud GPU")
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "medium"]
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes through the dataset")
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
        "--subset_size", type=int, default=None, help="Use subset of data for debugging"
    )
    parser.add_argument(
        "--generate_every",
        type=int,
        default=100,
        help="Generate sample text every N steps",
    )
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


def create_model(model_size: str, vocab_size: int, max_seq_len: int) -> GPT:
    if model_size == "small":
        return GPT.gpt2_small(vocab_size, max_seq_len)
    elif model_size == "medium":
        return GPT.gpt2_medium(vocab_size, max_seq_len)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_wikipedia_data(
    tokenizer: TiktokenTokenizer,
    max_seq_len: int,
    stride: int,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    print("Loading Wikitext-103 dataset (streaming)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    streaming_dataset = StreamingLLMDataset(
        dataset, tokenizer, max_seq_len, stride
    )

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


def save_checkpoint(
    model: GPT,
    optimizer: AdamW,
    epoch: int,
    training_step: int,
    loss: float,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{training_step}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": training_step,
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def generate_sample(
    model: GPT,
    tokenizer: TiktokenTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> None:
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=1.0)
        generated_tokens = output_ids[0].tolist()  # type: ignore[assignment]
        generated_text = tokenizer.decode(generated_tokens)  # type: ignore[arg-type]
        print("\n--- Sample Generation ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-------------------------\n")
    model.train()


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
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

    WIKITEXT_103_TOKENS_ESTIMATE = 100_000_000
    samples_per_epoch = WIKITEXT_103_TOKENS_ESTIMATE // args.stride
    steps_per_epoch = samples_per_epoch // args.batch_size
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = int(max_steps * args.warmup_ratio)
    min_lr = args.lr / 10
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"Training for {args.epochs} epochs (~{max_steps} steps)")
    print(f"Warmup steps: {warmup_steps}, Batch size: {args.batch_size}")

    global_step = 0
    losses: list[float] = []
    last_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        for _, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            lr = get_lr(global_step, warmup_steps, max_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))

            loss.backward()
            optimizer.step()  # type: ignore[union-attr]

            last_loss = loss.item()
            losses.append(last_loss)

            if global_step % args.log_every == 0:
                avg_loss = sum(losses[-args.log_every :]) / len(
                    losses[-args.log_every :]
                )
                print(
                    f"Epoch {epoch} | Step {global_step} | LR {lr:.2e} | Loss {last_loss:.4f} | Avg Loss {avg_loss:.4f}"
                )

            if args.generate_every > 0 and global_step % args.generate_every == 0 and global_step > 0:
                generate_sample(
                    model, tokenizer, device, args.generation_prompt, args.generation_tokens
                )

            if global_step % args.checkpoint_every == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step, last_loss, checkpoint_dir
                )

            global_step += 1

        save_checkpoint(model, optimizer, epoch, global_step, last_loss, checkpoint_dir)

    print("Training complete!")
    final_checkpoint = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Saved final model to {final_checkpoint}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
