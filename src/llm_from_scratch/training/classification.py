import math

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from tqdm import tqdm

from llm_from_scratch.data.classification import DataLoader
from llm_from_scratch.model.classification import GPTForClassification
from llm_from_scratch.tokenizers.base import Tokenizer


class GPTForClassificationTrainer:
    def __init__(
        self,
        model: GPTForClassification,
        tokenizer: Tokenizer,
        optim: Optimizer,
        loss_fn: CrossEntropyLoss,
        epochs: int,
        max_lr: float,
        loader: DataLoader,
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

    def train_step(self, epoch: int, step: int, input_ids: Tensor, target_cls: Tensor):
        # Set the learning rate
        abs_step = epoch * len(self.dataloader) + step
        lr = self.get_lr(abs_step)
        for pg in self.optim.param_groups:
            pg["lr"] = lr

        self.optim.zero_grad()

        logits = self.model(input_ids)  # shape [batch, num_classes]
        loss = self.loss_fn(logits, target_cls)

        loss.backward()

        self.optim.step()

        if abs_step % 10 == 0:
            print(f"Step {abs_step}: lr={lr:.2e}, Loss: {loss:.4f}")
            self.sample_classification()

        return loss.item()

    def sample_classification(self):
        texts = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "One of the best films I've seen this year. Highly recommend.",
            "Brilliant storytelling and amazing cinematography. A masterpiece.",
            "The cast delivered outstanding performances. I was moved to tears.",
            "Terrible movie. Complete waste of time. The plot made no sense.",
            "I fell asleep halfway through. Boring and predictable.",
            "The worst film I've ever watched. Avoid at all costs.",
            "Poor acting, weak script, and terrible direction. Very disappointing.",
        ]

        self.model.eval()
        with torch.no_grad():
            input_ids = pad_sequence(
                [torch.tensor(self.tokenizer.encode(t)) for t in texts],
                batch_first=True,
                padding_value=0,
            )
            input_ids = input_ids.to(self.device)
            logits = self.model(input_ids)
            preds = logits.argmax(dim=-1)
            for text, pred in zip(texts, preds):
                label = "Positive" if pred.item() == 1 else "Negative"
                print(f"{label} - {text[:50]}...")
        self.model.train()

    def train_epoch(self, epoch: int):
        for step, (input_ids, target_cls) in tqdm(enumerate(self.dataloader)):
            input_ids = input_ids.to(self.device)
            target_cls = target_cls.to(self.device)
            self.train_step(epoch, step, input_ids, target_cls)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # set the model to training mode
            print(f"Epoch {epoch}")
            self.train_epoch(epoch)
