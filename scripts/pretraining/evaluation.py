import argparse

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import StreamingLLMDataset
from llm_from_scratch.model.pretrained import load_pretrained_lm
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.evaluation import evaluate_perplexity
from llm_from_scratch.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained GPT-2 perplexity on Wikitext-103"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Limit evaluation steps (default: full dataset)",
    )
    return parser.parse_args()


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


def evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = TiktokenTokenizer()
    model = load_pretrained_lm(args.model_size, max_seq_len=args.max_seq_len)

    dataloader = load_wikipedia_data(
        tokenizer, args.max_seq_len, args.stride, args.batch_size
    )
    perplexity = evaluate_perplexity(
        model, dataloader, device=device, max_steps=args.max_steps
    )

    print(f"Perplexity: {perplexity:.2f}")

    # You can use the code below to run evaluation on HuggingFace's model for
    # comparison.
    # from llm_from_scratch.training.evaluation import evaluate_perplexity_hf
    # from transformers import GPT2LMHeadModel, GPT2Tokenizer
    #
    # hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    #
    # dataloader = load_wikipedia_data(hf_tokenizer, 1024, 64, 4)
    # perplexity = evaluate_perplexity_hf(hf_model, dataloader, device=device)
    #
    # print(f"HuggingFace perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
