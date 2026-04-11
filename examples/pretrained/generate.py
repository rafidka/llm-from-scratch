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

# You can use the code below to generate with HuggingFace model.
#
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
#
# hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
# hf_model.eval()
# inputs = hf_tokenizer("The quick brown fox", return_tensors="pt")
# outputs = hf_model.generate(
#     **inputs, max_new_tokens=20, temperature=0.0, top_k=40, do_sample=False
# )
# print(hf_tokenizer.decode(outputs[0]))
#
