import os

import requests

from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
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
    corpus = f.read()[:10000]

gpt2 = TiktokenTokenizer("gpt2")

our_bpe = BPETokenizer(corpus, 100)

# Compare tokenization
test_texts = [
    "The artist was a young man.",
    "Hello world!",
    "Machine learning is fascinating.",
]

print("GPT-2 (50,257 vocab) vs Our BPE (174 vocab)")
print("=" * 50)

for text in test_texts:
    gpt2_enc = gpt2.encode(text)
    our_enc = our_bpe.encode(text)

    print(f"Text: {text}")
    print(f"  GPT-2: {len(gpt2_enc)} tokens")
    print(f"         {gpt2_enc}")
    print(f"  Ours:  {len(our_enc)} tokens")
    if len(our_enc) > 15:
        print(f"         {our_enc[:15]}...")
    else:
        print(f"         {our_enc}")
    print()
