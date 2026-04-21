import argparse

import dotenv
from datasets import load_dataset  # type: ignore[import-untyped]
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.data.instruction import _IGNORE_INDEX, create_dataloader
from llm_from_scratch.model.pretrained import load_pretrained_lm
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.causallm import GPTForCausalLMTrainer
from llm_from_scratch.utils import get_device

dotenv.load_dotenv()

EVAL_PROMPTS = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nExplain why the sky is blue in simple terms.\n\n### Response:\n",
    "### Instruction:\nList three benefits of exercise.\n\n### Response:\n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune GPT-2 for instruction following on Alpaca dataset"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2-large",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accml_steps", type=int, default=5)
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        default=False,
        help="Enable mixed precision training (requires CUDA)",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable mixed precision training (requires CUDA)",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="If enabled, train LoRA adapters.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = TiktokenTokenizer()
    ds = load_dataset("tatsu-lab/alpaca")
    if args.batch_size and args.max_tokens_per_batch:
        raise RuntimeError("Cannot specify both batch_size and max_tokens_per_batch")
    elif args.max_tokens_per_batch:
        dl = create_dataloader(
            ds["train"],
            tokenizer,
            args.max_seq_len,
            max_tokens_per_batch=args.max_tokens_per_batch,
        )
    elif args.batch_size:
        dl = create_dataloader(
            ds["train"],
            tokenizer,
            args.max_seq_len,
            batch_size=args.batch_size or 8,
        )
    else:
        raise RuntimeError("Need to specify either batch_size or max_tokens_per_batch")

    model = load_pretrained_lm(
        args.base_model,
        max_seq_len=args.max_seq_len,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    if args.lora:
        model.lorafy(16, alpha=32, sigma=0.02)
    model.to(device)

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {args.base_model}, "
        f"Total Parameters: {params_total:,}, "
        f"Trainable Parameters: {params_trainable:,}"
    )

    optim = AdamW(model.parameters(), weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss(ignore_index=_IGNORE_INDEX)

    trainer = GPTForCausalLMTrainer(
        model=model,
        tokenizer=tokenizer,
        optim=optim,
        loss_fn=loss_fn,
        epochs=args.epochs,
        max_lr=args.lr,
        data_loader=dl,
        device=device,
        test_prompts=EVAL_PROMPTS,
        grad_accml_steps=args.grad_accml_steps,
        use_mixed_precision=args.use_mixed_precision,
        checkpoint_dir="./checkpoints",
        checkpoint_every=1000,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train(args)
