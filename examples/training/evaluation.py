import torch

from llm_from_scratch.model.pretrained import load_pretrained
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.evaluation import evaluate_perplexity

from .train_cloud import load_wikipedia_data

tokenizer = TiktokenTokenizer()
model = load_pretrained("gpt2", max_seq_len=1024)

dataloader = load_wikipedia_data(tokenizer, 1024, 64, 4)
perplexity = evaluate_perplexity(model, dataloader, device=torch.device("mps"))

print(perplexity)


# You can use the code to below to run evaluation on the HuggingFace's model for
# comparison.
# from llm_from_scratch.training.evaluation import evaluate_perplexity_hf
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
#
# dataloader = load_wikipedia_data(tokenizer, 1024, 64, 4)
# perplexity = evaluate_perplexity_hf(model, dataloader, device=torch.device("mps"))
#
# print(perplexity)
#
