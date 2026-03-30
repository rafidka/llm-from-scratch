import torch

weight_count = 10
weights = list(range(weight_count))

torch.multinomial(torch.tensor(weights, dtype=torch.float), num_samples=100

