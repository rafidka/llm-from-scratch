from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.tokenizers.bpe import BPETokenizer

# Load GPT-2 tokenizer
gpt2 = TiktokenTokenizer("gpt2")

# Our BPE tokenizer (trained on small corpus)
with open("data/the_verdict.txt", "r") as f:
    corpus = f.read()[:10000]
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
