from transformers import GPT2LMHeadModel, GPT2Tokenizer

hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_model.eval()
inputs = hf_tokenizer("The quick brown fox", return_tensors="pt")
outputs = hf_model.generate(
    **inputs, max_new_tokens=20, temperature=0.0, top_k=40, do_sample=False
)
print(hf_tokenizer.decode(outputs[0]))
