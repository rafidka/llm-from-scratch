from llm_from_scratch.tokenizers.bpe import BPETokenizer

# Load the text
with open("data/the_verdict.txt", "r") as f:
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
