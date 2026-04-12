import torch
from llm_from_scratch.model.causallm import GPTForCausalLM

# Small model for testing
model = GPTForCausalLM(
    vocab_size=1000,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    max_seq_len=128,
    dropout=0.0,
)
model.eval()
# Start with a token
token_ids = torch.randint(0, 1000, (1, 5))  # [1, 5]
# Generate
with torch.no_grad():
    output = model.generate(token_ids, max_new_tokens=10, temperature=1.0, top_k=50)
print(f"Input shape: {token_ids.shape}")
print(f"Output shape: {output.shape}")  # Should be [1, 15]

print(f"Input: {token_ids}")
print(f"Output: {output}")
