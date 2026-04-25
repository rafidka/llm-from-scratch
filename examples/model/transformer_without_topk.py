import torch

from llm_from_scratch.model.causallm import GPTForCausalLM

input_ids = torch.randint(0, 10, (3, 15))
print("input_ids", input_ids.shape)

temperature = 1.0

gpt = GPTForCausalLM(1000, 64, 8, 8, 32, 0.1)
logits = gpt.forward(input_ids).output
print("logits", logits.shape)
last_logit = logits[:, -1, :]
print("last_logits", last_logit.shape)
last_logit = torch.softmax(last_logit / temperature, dim=-1)

pick = torch.multinomial(last_logit, num_samples=1)
print(pick.shape)
new_input_ids = torch.cat((input_ids, pick), dim=-1)
print("new_input_ids", new_input_ids.shape)
