import torch

from llm_from_scratch.model.causallm import GPTForCausalLM

input_ids = torch.randint(0, 10, (3, 15))
print("input_ids", input_ids.shape)

temperature = 1.0
top_k = 11

gpt = GPTForCausalLM(1000, 64, 8, 8, 32, 0.1)
logits = gpt.forward(input_ids)
print("logits", logits.shape)
last_logit = logits[:, -1, :]
print("last_logits", last_logit.shape)
last_logit = torch.softmax(last_logit / temperature, dim=-1)
top_k_probs, top_k_indices = torch.topk(last_logit, k=top_k, dim=-1)
print("top_k_probs", top_k_probs)
print("top_k_indices", top_k_indices)

next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
print("next_token_idx", next_token_idx.shape)
# pick_idx = distr.indices[-1][pick]
# print("pick_idx", pick_idx.shape)
# print("pick_idx value", pick_idx)
#
# new_input_ids = torch.cat((input_ids, pick_idx), dim=-1)
# print("new_input_ids", new_input_ids.shape)
#
