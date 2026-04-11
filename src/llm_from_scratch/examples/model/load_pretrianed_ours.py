import torch

from llm_from_scratch.model.pretrained import load_pretrained
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer

tokenizer = TiktokenTokenizer()

model = load_pretrained("gpt2", max_seq_len=1024)
model.eval()
prompt = "The quick brown fox"
input_ids = torch.tensor([tokenizer.encode(prompt)])
output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.0, top_k=40)
print(tokenizer.decode(output_ids[0].tolist()))
