import contextlib
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.base import Tokenizer


class GPTForCausalLMTrainer:
    def __init__(
        self,
        model: GPTForCausalLM,
        tokenizer: Tokenizer,
        optim: Optimizer,
        loss_fn: CrossEntropyLoss,
        epochs: int,
        max_lr: float,
        data_loader: DataLoader,
        device: torch.device,
        grad_accml_steps: int = 1,
        test_prompts: list[str] | None = None,
        use_mixed_precision: bool = False,
        warmup_ratio: float = 0.1,
        log_every: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 1000,
        generation_tokens: int = 256,
        total_steps: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.max_lr = max_lr
        self.min_lr = max_lr / 10
        self.lr = 0.0
        self.data_loader = data_loader
        self.device = device
        self.grad_accml_steps = grad_accml_steps
        self.test_prompts = test_prompts
        self.use_mixed_precision = use_mixed_precision
        self.mp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_mixed_precision
            else contextlib.nullcontext()
        )
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.generation_tokens = generation_tokens
        self.last_loss: float = 0.0

        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = epochs * len(data_loader)
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self._global_step = 0

    def get_lr(self, step_abs: int) -> float:
        """Returns the learning rate for a given absolute training step.

        The externally provided `step_abs` is expected to be 0-indexed across the
        full training run. Internally, it is converted to 1-indexed so that:

        - steps 1 through `warmup_steps` linearly warm up from a small positive
        learning rate to `max_lr`
        - subsequent steps follow a cosine decay from `max_lr` down toward
        `min_lr`

        This means the very first optimizer update uses a non-zero learning rate
        (min_lr), the warmup phase ends exactly at `max_lr`, and the last training
        step uses min_lr.

        Args:
            step_abs: Absolute 0-indexed training step over the full training run.

        Returns:
            The learning rate for this step.

        Raises:
            ValueError: If `step_abs` is out of range.
        """
        if not (0 <= step_abs < self.total_steps):
            raise ValueError("step_abs out of range")

        step_abs = step_abs + 1

        if step_abs <= self.warmup_steps:
            return self.max_lr * step_abs / self.warmup_steps

        progress = (step_abs - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def _optim_step(self):
        self.lr = self.get_lr(self._global_step)

        if (self._global_step + 1) % self.grad_accml_steps != 0 and (
            self._global_step + 1
        ) < self.total_steps:
            return

        for pg in self.optim.param_groups:
            pg["lr"] = self.lr

        self.optim.step()
        self.optim.zero_grad()

    def _save_checkpoint(self, epoch: int, step_abs: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step_abs}.pt"
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "epoch": epoch,
                "step": step_abs,
                "loss": self.last_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    def train_step(self, epoch: int, step: int, input_ids: Tensor, target_ids: Tensor):
        with self.mp_context:
            logits = self.model(input_ids)
            loss = self.loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))
        (loss / self.grad_accml_steps).backward()

        self._optim_step()

        self.last_loss = loss.item()
        step_abs = self._global_step

        if step_abs % self.log_every == 0 and step_abs > 0:
            print(f"Step {step_abs}: lr={self.lr:.2e}, Loss: {self.last_loss:.4f}")
            self._test_model()

        if (
            self.checkpoint_dir is not None
            and step_abs % self.checkpoint_every == 0
            and step_abs > 0
        ):
            self._save_checkpoint(epoch, step_abs)

        self._global_step += 1
        return self.last_loss

    def train_epoch(self, epoch: int):
        for step, (input_ids, target_ids) in tqdm(enumerate(self.data_loader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            self.train_step(epoch, step, input_ids, target_ids)

    @torch.no_grad()
    def _test_model(self):
        if not self.test_prompts:
            return
        self.model.eval()
        for prompt in self.test_prompts:
            input_ids = (
                torch.tensor(self.tokenizer.encode(prompt)).view(1, -1).to(self.device)
            )
            with self.mp_context:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.generation_tokens,
                    temperature=0.2,
                    top_k=10,
                    eos_token_id=self.tokenizer.encode("\n")[0],
                )
            print("----------")
            print()
            print(self.tokenizer.decode(output_ids[0].tolist()))
        print()
        print("----------")
        self.model.train()

    def train(self):
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch}")
            self.model.train()
            self.train_epoch(epoch)
