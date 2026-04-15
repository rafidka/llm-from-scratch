import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.base import Tokenizer
from llm_from_scratch.training.base import GPTTrainer


class GPTForCausalLMTrainer(GPTTrainer[GPTForCausalLM]):
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
        self.test_prompts = test_prompts
        self.generation_tokens = generation_tokens

    def train_step(self, epoch: int, step: int, input_ids: Tensor, target_ids: Tensor):
        with self.mp_context:
            logits = self.model(input_ids)
            loss = self.loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))
        (loss / self.grad_accml_steps).backward()

        self._optim_step()

        self.last_loss = loss.item()
        self._on_train_step_end(epoch, step)
        return self.last_loss

    def _on_log_step(self, step_abs: int) -> None:
        print(f"Step {step_abs}: lr={self.lr:.2e}, Loss: {self.last_loss:.4f}")
        self._test_model()

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
                    top_k=40,
                    eos_token_id=self.tokenizer.encode("\n")[0],
                )
            print("----------")
            print()
            print(self.tokenizer.decode(output_ids[0].tolist()))
        print()
        print("----------")
        self.model.train()
