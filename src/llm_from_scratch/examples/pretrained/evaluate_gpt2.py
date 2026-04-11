import torch

from llm_from_scratch.examples.training.train_cloud import load_wikipedia_data
from llm_from_scratch.model.transformer import GPT
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.evaluate import evaluate_perplexity


tokenizer = TiktokenTokenizer()
model = GPT.from_pretrained("gpt2", max_seq_len=1024)

dataloader = load_wikipedia_data(tokenizer, 1024, 64, 4)
perplexity = evaluate_perplexity(model, dataloader, device=torch.device("mps"))

print(perplexity)
