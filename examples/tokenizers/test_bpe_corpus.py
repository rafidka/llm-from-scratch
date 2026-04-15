import os

import requests

from llm_from_scratch.tokenizers.bpe import BPETokenizer

THE_VERDICT_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"


def download_the_verdict(save_path: str) -> str:
    response = requests.get(THE_VERDICT_URL, timeout=30)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return save_path


data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)
filepath = os.path.join(data_dir, "the_verdict.txt")

if not os.path.exists(filepath):
    print("Downloading The Verdict text...")
    download_the_verdict(filepath)
    print(f"Saved to {filepath}")
else:
    print(f"Found existing file at {filepath}")

with open(filepath, "r", encoding="utf-8") as f:
    corpus = f.read()

# Use a subset for faster testing
corpus = corpus[:10000]

print(f"Corpus length: {len(corpus)} characters")
print()

# Train with a reasonable number of merges
num_merges = 100
print(f"Training BPE with {num_merges} merges...")
tokenizer = BPETokenizer(corpus, num_merges)
print()

# Check what merges were learned
print(f"Vocab size: {len(tokenizer._token_to_id)}")  # type: ignore
print()
print("First 50 merges:")
for i, merge in enumerate(tokenizer.merges[:50]):
    print(f"  {i + 1}. {merge}")
print()
print("Last 50 merges:")
for i, merge in enumerate(tokenizer.merges[-50:]):
    print(f"  {len(tokenizer.merges) - 50 + i + 1}. {merge}")
print()

# Test encode/decode
test_texts = [
    "The artist was a young man.",
    "Dorian Gray looked at the picture.",
    "Hello world!",
]

for test_text in test_texts:
    encoded = tokenizer.encode(test_text)
    decoded = "".join(tokenizer.decode(encoded))
    print(f"Original: {test_text}")
    print(f"Encoded:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"Decoded:  {decoded}")
    print()
