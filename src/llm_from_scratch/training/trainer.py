import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from llm_from_scratch.data.loader import GPTDataLoader
from llm_from_scratch.model.transformer import GPT
from tqdm import tqdm


class GPTTrainer:
    def __init__(
        self,
        model: GPT,
        optim: Optimizer,
        loss_fn: CrossEntropyLoss,
        epochs: int,
        device: torch.device,
    ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

    def train_step(self, input_ids: Tensor, target_ids: Tensor):
        self.optim.zero_grad()

        logits = self.model(input_ids)  # shape [batch, seq_len, vocab_size]
        loss = self.loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))

        loss.backward()

        self.optim.step()

        return loss.item()

    def train_epoch(self, loader: GPTDataLoader):
        for idx, (input_ids, target_ids) in tqdm(enumerate(loader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            loss = self.train_step(input_ids, target_ids)
            if idx % 50 == 0:
                print(f"Loss: {loss}")

    def train(self, loader: GPTDataLoader):
        self.model.train()  # set the model to training mode
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            self.train_epoch(loader)
