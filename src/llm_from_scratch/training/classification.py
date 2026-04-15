from itertools import islice

import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from llm_from_scratch.model.classification import GPTForClassification
from llm_from_scratch.tokenizers.base import Tokenizer
from llm_from_scratch.training.base import GPTTrainer


class GPTForClassificationTrainer(GPTTrainer[GPTForClassification]):
    def __init__(
        self,
        model: GPTForClassification,
        tokenizer: Tokenizer,
        optim: Optimizer,
        loss_fn: CrossEntropyLoss,
        epochs: int,
        max_lr: float,
        data_loader: DataLoader,
        device: torch.device,
        eval_loader: DataLoader | None = None,
        eval_every_step: int = 500,
        grad_accml_steps: int = 1,
        use_mixed_precision: bool = False,
        warmup_ratio: float = 0.1,
        log_every: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 1000,
        total_steps: int | None = None,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optim=optim,
            loss_fn=loss_fn,
            epochs=epochs,
            max_lr=max_lr,
            data_loader=data_loader,
            device=device,
            grad_accml_steps=grad_accml_steps,
            use_mixed_precision=use_mixed_precision,
            warmup_ratio=warmup_ratio,
            log_every=log_every,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            total_steps=total_steps,
        )
        self.eval_loader = eval_loader
        self.eval_every_step = eval_every_step

    def train_step(self, epoch: int, step: int, input_ids: Tensor, target_cls: Tensor):
        with self.mp_context:
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_cls)
        (loss / self.grad_accml_steps).backward()

        self._optim_step()

        self.last_loss = loss.item()
        self._on_train_step_end(epoch, step)
        return self.last_loss

    def _on_log_step(self, step_abs: int) -> None:
        print(f"Step {step_abs}: lr={self.lr:.2e}, Loss: {self.last_loss:.4f}")
        self.sample_classification()
        if self.eval_loader:
            self.eval()

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

    def eval(self):
        if not self.eval_loader:
            raise RuntimeError("eval_loader is not set.")
        with torch.no_grad():
            self.model.eval()
            all_true_labels = torch.empty((0), device=self.device)
            all_pred_labels = torch.empty((0), device=self.device)
            for input_ids, true_labels in tqdm(islice(self.eval_loader, 100)):
                input_ids = input_ids.to(self.device)
                true_labels = true_labels.to(self.device)
                pred_labels = self.model(input_ids).argmax(dim=-1)

                all_true_labels = torch.cat((all_true_labels, true_labels))
                all_pred_labels = torch.cat((all_pred_labels, pred_labels))

        tp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
        tn = ((pred_labels == 0) & (true_labels == 0)).sum().item()
        fp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
        fn = ((pred_labels == 0) & (true_labels == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        print("accuracy", accuracy)
        print("precision", precision)
        print("recall", recall)
        print("f1", f1)
