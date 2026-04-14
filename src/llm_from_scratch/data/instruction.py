from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

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
        self.eos = tokenizer.encode("<|endoftext|>")[0]
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


def create_dataloader(hf_dataset, tokenizer, batch_size, max_seq_len):
    def collate_fn(batch):
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

        return padded_input_ids, padded_target_ids

    dataset = DatasetForInstructionFineTuning(hf_dataset, tokenizer, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
