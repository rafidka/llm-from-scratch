import math

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
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

        if abs_step % 100 == 0 and abs_step > 0:
            print(f"Step {abs_step}: lr={lr:.2e}, Loss: {loss:.4f}")
            self.generate()

        return loss.item()

    def train_epoch(self, epoch: int):
        for step, (input_ids, target_ids) in tqdm(enumerate(self.dataloader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            self.train_step(epoch, step, input_ids, target_ids)

    @torch.no_grad()
    def generate(self):
        # TODO Make the prompt a parameter so we don't have to manually change the
        # prompt for different training types.
        # prompt = "The quick brown fox"

        prompts = [
            "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
            "### Instruction:\nExplain why the sky is blue in simple terms.\n\n### Response:\n",
            "### Instruction:\nList three benefits of exercise.\n\n### Response:\n",
        ]
        for prompt in prompts:
            input_ids = (
                torch.tensor(self.tokenizer.encode(prompt)).view(1, -1).to(self.device)
            )
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.8,
                top_k=40,
                eos_token_id=self.tokenizer.encode("<|endoftext|>")[0],
            )
            print("----------")
            print()
            print(self.tokenizer.decode(output_ids[0].tolist()))
        print()
        print("----------")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # set the model to training mode
            print(f"Epoch {epoch}")
            self.train_epoch(epoch)

            self.model.eval()
            self.generate()
