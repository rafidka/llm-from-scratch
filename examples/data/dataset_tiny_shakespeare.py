import os
import requests

from llm_from_scratch.data.dataset import LLMDataset
from llm_from_scratch.data.loader import create_dataloader
from llm_from_scratch.tokenizers.base import Tokenizer
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare(save_path: str = "tiny_shakespeare.txt") -> str:
    response = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return save_path


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


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

    text = load_text(filepath)
    print(f"Dataset size: {len(text):,} characters")

    dataset = LLMDataset(tokenizer, text, max_seq_len, stride)

    return create_dataloader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    tokenizer = TiktokenTokenizer()

    dataloader = get_dataloader(tokenizer)
    print(f"Number of batches: {len(dataloader)}")

    inputs, targets = next(iter(dataloader))
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
