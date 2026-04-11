from itertools import islice

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from torch.utils.data import DataLoader


def evaluate_perplexity(
    model: GPT2LMHeadModel,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_steps: int | None = None,
) -> float:
    loss_fn = CrossEntropyLoss()

    losses = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for idx, (input_ids, target_ids) in enumerate(
            tqdm(islice(dataloader, max_steps))
        ):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            output = model(input_ids)
            logits = output.logits
            loss = loss_fn(logits.flatten(0, 1), target_ids.flatten(0, 1))
            losses.append(loss.item())

            if idx % 50 == 0:
                perp = torch.tensor(losses).mean().exp().item()
                print(f"Perplexity {perp}")

        return torch.tensor(losses).mean().exp().item()
