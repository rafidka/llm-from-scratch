import random
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from llm_from_scratch.tokenizers.base import Tokenizer


def _format_prompt(instruction: str, input: str) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input:
        prompt += f"### Input:\n{input}\n\n"
    prompt += "### Response:\n"
    return prompt


_IGNORE_INDEX = -100


class DatasetForInstructionFineTuning(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Tokenizer,
        max_seq_len: int,
        instruction_field_name: str = "instruction",
        input_field_name: str = "input",
        output_field_name: str = "output",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.eos = tokenizer.encode("<|edoftext|>")[0]
        self.instruction_field_name = instruction_field_name
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

        self.all_input_ids = []
        self.prompt_lens = []
        for row in self.hf_dataset:
            inst = row["instruction"]
            inp = row["input"]
            answer = row["output"]
            prompt = _format_prompt(inst, inp)
            prompt_ids = self.tokenizer.encode(prompt)
            answer_ids = self.tokenizer.encode(answer)
            if len(prompt_ids) + len(answer_ids) > max_seq_len:
                continue
            input_ids = prompt_ids + answer_ids

            self.all_input_ids.append(input_ids)
            self.prompt_lens.append(len(prompt_ids))

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < 0 or idx >= len(self.all_input_ids):
            raise IndexError()
        input_ids = self.all_input_ids[idx]
        prompt_len = self.prompt_lens[idx]
        mask = [_IGNORE_INDEX] * (prompt_len - 1)
        target_ids = mask + input_ids[prompt_len:] + [self.eos]

        return torch.tensor(input_ids), torch.tensor(target_ids)


class MaxTokenCountBatchSampler(Sampler):
    def __init__(
        self, dataset: DatasetForInstructionFineTuning, max_tokens_per_batch: int
    ):
        self.dataset = dataset
        self.batches = []
        batch = []

        noise_factor = 5
        idx_map = sorted(
            [(idx, len(ids)) for idx, ids in enumerate(self.dataset.all_input_ids)],
            key=lambda item: item[1] + random.uniform(-noise_factor, noise_factor),
        )
        for idx, seq_len in idx_map:
            if sum(s for _, s in batch) + seq_len > max_tokens_per_batch:
                # Shuffle each batch
                random.shuffle(batch)
                self.batches.append(batch)
                batch = []
            batch.append((idx, seq_len))
        if batch:
            # Shuffle each batch
            random.shuffle(batch)
            self.batches.append(batch)

        # Shuffle the list of batches.
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield [idx for idx, _ in batch]


def _pad_sequences_fn(batch):
    # Separate sequences and labels
    # Assumes dataset __getitem__ returns (tensor, label)
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # Pad sequences to match the longest one in this batch
    # batch_first=True makes shape (Batch, Seq_Len, Features)
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_target_ids = pad_sequence(
        target_ids, batch_first=True, padding_value=_IGNORE_INDEX
    )
    attention_mask = (padded_input_ids != 0).long()
    return padded_input_ids, padded_target_ids, attention_mask


def create_dataloader(
    hf_dataset: Any,
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int | None = None,
    max_tokens_per_batch: int | None = None,
):
    if batch_size and max_tokens_per_batch:
        raise RuntimeError("Cannot specify both batch_size and max_tokens_per_batch")
    elif batch_size is not None:
        return _create_dataloader_with_batch_size(
            hf_dataset, tokenizer, max_seq_len, batch_size
        )
    elif max_tokens_per_batch is not None:
        return _create_dataloader_with_max_tokens_per_batch(
            hf_dataset, tokenizer, max_seq_len, max_tokens_per_batch
        )
    else:
        raise RuntimeError(
            "You should specify either batch_size or max_tokens_per_batch"
        )


def _create_dataloader_with_batch_size(
    hf_dataset: Any,
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
):
    dataset = DatasetForInstructionFineTuning(hf_dataset, tokenizer, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_pad_sequences_fn,
    )


def _create_dataloader_with_max_tokens_per_batch(
    hf_dataset: Any,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_tokens_per_batch: int,
):

    dataset = DatasetForInstructionFineTuning(hf_dataset, tokenizer, max_seq_len)
    return DataLoader(
        dataset,
        collate_fn=_pad_sequences_fn,
        batch_sampler=MaxTokenCountBatchSampler(dataset, max_tokens_per_batch),
    )
