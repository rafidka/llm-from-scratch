import math
from itertools import islice

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
        train_loader: DataLoader,
        device: torch.device,
        eval_loader: DataLoader | None = None,  # TODO move up after train_loader
        eval_every_step: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eval_every_step = eval_every_step
        self.max_lr = max_lr
        self.min_lr = max_lr / 10
        self.max_steps = epochs * len(train_loader)
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
        abs_step = epoch * len(self.train_loader) + step
        lr = self.get_lr(abs_step)
        for pg in self.optim.param_groups:
            pg["lr"] = lr

        self.optim.zero_grad()

        logits = self.model(input_ids)  # shape [batch, num_classes]
        loss = self.loss_fn(logits, target_cls)

        loss.backward()

        self.optim.step()

        # if abs_step % 100 == 0:
        if abs_step > 0 and abs_step % self.eval_every_step == 0:
            print(f"Step {abs_step}: lr={lr:.2e}, Loss: {loss:.4f}")
            self.sample_classification()

            if self.eval_loader:
                self.eval()

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
        for step, (input_ids, target_cls) in tqdm(enumerate(self.train_loader)):
            input_ids = input_ids.to(self.device)
            target_cls = target_cls.to(self.device)
            self.train_step(epoch, step, input_ids, target_cls)

    def eval(self):
        if not self.eval_loader:
            raise RuntimeError("eval_loader is not set.")
        self.model.eval()
        all_true_labels = torch.empty((0), device=self.device)
        all_pred_labels = torch.empty((0), device=self.device)
        for input_ids, true_labels in tqdm(
            islice(self.eval_loader, 100)
        ):  # we don't need to use a lot of samples.
            input_ids = input_ids.to(self.device)
            true_labels = true_labels.to(self.device)
            pred_labels = self.model(input_ids).argmax(dim=-1)

            all_true_labels = torch.cat((all_true_labels, true_labels))
            all_pred_labels = torch.cat((all_pred_labels, pred_labels))

        tp, tn, fp, fn = 0, 0, 0, 0
        for true_label, pred_label in zip(all_true_labels, all_pred_labels):
            t = int(true_label.item())
            p = int(pred_label.item())

            if t == p:
                if p == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if p == 1:
                    fp += 1
                else:
                    fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print("accuracy", accuracy)
        print("precision", precision)
        print("recall", recall)
        print("f1", f1)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # set the model to training mode
            print(f"Epoch {epoch}")
            self.train_epoch(epoch)
