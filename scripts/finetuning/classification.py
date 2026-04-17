import argparse

from datasets import load_dataset  # type: ignore[import-untyped]
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.data.classification import create_dataloader
from llm_from_scratch.model.pretrained import load_pretrained_cls
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.classification import GPTForClassificationTrainer
from llm_from_scratch.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune GPT-2 for text classification on IMDB"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--eval_every_step", type=int, default=500)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = TiktokenTokenizer()
    imdb = load_dataset("stanfordnlp/imdb")
    train_dl = create_dataloader(
        imdb["train"], tokenizer, args.batch_size, args.max_seq_len
    )
    eval_dl = create_dataloader(
        imdb["test"], tokenizer, args.batch_size, args.max_seq_len
    )

    model = load_pretrained_cls(
        args.base_model,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.base_model}, Parameters: {num_params:,}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()

    trainer = GPTForClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        optim=optim,
        loss_fn=loss_fn,
        epochs=args.epochs,
        max_lr=args.lr,
        data_loader=train_dl,
        device=device,
        eval_loader=eval_dl,
        eval_every_step=args.eval_every_step,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train(args)
