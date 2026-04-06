import math

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from tqdm import tqdm

from llm_from_scratch.data.loader import GPTDataLoader
from llm_from_scratch.model.transformer import GPT
from llm_from_scratch.tokenizers.base import Tokenizer


class GPTTrainer:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        optim: Optimizer,
        loss_fn: CrossEntropyLoss,
        epochs: int,
        max_lr: float,
        loader: GPTDataLoader,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

        self.dataloader = loader
        self.max_lr = max_lr
        self.min_lr = max_lr / 10
        self.max_steps = epochs * len(loader)
        self.warmup_steps = self.max_steps / 10

    def get_lr(
        self,
        step: int,
    ) -> float:
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        # cosine decay phase
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def train_step(self, epoch: int, step: int, input_ids: Tensor, target_ids: Tensor):
        # Set the learning rate
        abs_step = epoch * len(self.dataloader) + step
        lr = self.get_lr(abs_step)
        for pg in self.optim.param_groups:
            pg["lr"] = lr

        self.optim.zero_grad()

        logits = self.model(input_ids)  # shape [batch, seq_len, vocab_size]
        loss = self.loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))

        loss.backward()

        self.optim.step()

        if abs_step % 50 == 0:
            print(f"Step {abs_step}: lr={lr:.2e}, Loss: {loss:.4f}")

        return loss.item()

    def train_epoch(self, epoch: int):
        for step, (input_ids, target_ids) in tqdm(enumerate(self.dataloader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            self.train_step(epoch, step, input_ids, target_ids)

    @torch.no_grad()
    def generate(self):
        prompt = "To be, or not "
        input_ids = (
            torch.tensor(self.tokenizer.encode(prompt)).view(1, -1).to(self.device)
        )
        output_ids = self.model.generate(
            input_ids, max_new_tokens=50, temperature=0.8, top_k=40
        )
        print(self.tokenizer.decode(output_ids[0].tolist()))

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # set the model to training mode
            print(f"Epoch {epoch}")
            self.train_epoch(epoch)

            self.model.eval()
            self.generate()
