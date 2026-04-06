import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.examples.data.dataset_tiny_shakespeare import get_dataloader
from llm_from_scratch.model.transformer import GPT
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.trainer import GPTTrainer

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)


def create_trainer(
    vocab_size: int, max_seq_len: int, epochs: int, lr: float, weight_decay: float
):
    model = GPT.test(vocab_size, max_seq_len)
    model.to(device)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()

    return GPTTrainer(model, optim, loss_fn, epochs, device)


max_seq_len = 256
epochs = 3
lr = 3e-4
weight_decay = 0.01
tokenizer = TiktokenTokenizer()
dl = get_dataloader(tokenizer, max_seq_len)
trainer = create_trainer(
    tokenizer.vocab_size,
    max_seq_len,
    epochs,
    lr,
    weight_decay,
)
trainer.train(dl)
