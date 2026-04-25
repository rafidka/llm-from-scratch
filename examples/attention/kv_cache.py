import torch
from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer

tokenizer = TiktokenTokenizer()
model = GPTForCausalLM.tiny(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=128,
    use_rope=True,
    use_rms_norm=True,
    use_swiglu=True,
    num_kv_threads=2,
)

prompt = "Hello world"
token_ids = torch.tensor([tokenizer.encode(prompt)]).view(1, -1)

# Generate with KV cache
output = model.generate(token_ids, max_new_tokens=20, temperature=0)
print("Generated:", tokenizer.decode(output[0].tolist()))

# Verify: generate without cache (old-style, full forward each step)
# We can't easily compare since generate() now always uses cache,
# but we can verify the model forward works with and without cache.

# Forward without cache
ret1 = model.forward(token_ids)
print("Forward (no cache) output shape:", ret1.output.shape)
print("KV caches:", len(ret1.kv_caches), "layers, cache shape:", ret1.kv_caches[0][0].shape)

# Forward with cache (add one new token)
new_token = torch.tensor([[tokenizer.encode("x")[0]]])
ret2 = model.forward(new_token, kv_caches=ret1.kv_caches)
print("Forward (with cache) output shape:", ret2.output.shape)
print("KV caches:", len(ret2.kv_caches), "layers, cache shape:", ret2.kv_caches[0][0].shape)
