import torch
from llm_from_scratch.model.classification import GPTForClassification

batch = 4
seq_len = 25
num_classes = 12

# Small model for testing
model = GPTForClassification(
    vocab_size=1000,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    max_seq_len=128,
    dropout=0.0,
    num_classes=num_classes,
)
input_ids = torch.randint(0, 1000, (batch, seq_len))

output = model(input_ids).output
print(output.shape)
