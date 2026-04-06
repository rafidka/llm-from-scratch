import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.data.loader import GPTDataLoader
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


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:  # type: ignore
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # LayerNorm initializes itself (weight=1, bias=0 by default)


def create_trainer(
    vocab_size: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dl: GPTDataLoader,
):
    model = GPT.test(vocab_size, max_seq_len)
    model.apply(init_weights)
    model.to(device)
    optim = AdamW(model.parameters(), weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()

    return GPTTrainer(model, optim, loss_fn, epochs, lr, dl, device)


max_seq_len = 256
epochs = 10
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
    dl,
)
trainer.train()
