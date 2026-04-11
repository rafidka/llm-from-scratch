import torch

from llm_from_scratch.examples.training.train_cloud import load_wikipedia_data
from llm_from_scratch.training.evaluate_hf import evaluate_perplexity

from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataloader = load_wikipedia_data(tokenizer, 1024, 64, 4)
perplexity = evaluate_perplexity(model, dataloader, device=torch.device("mps"))

print(perplexity)
