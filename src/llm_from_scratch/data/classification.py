import torch
from typing import Any, Iterator

from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from llm_from_scratch.tokenizers.base import Tokenizer


class DatasetForClassification(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Tokenizer,
        max_text_len: int,
        text_field_name: str = "text",
        label_field_name: str = "label",
    ):
        self.tokenizer = tokenizer
        self.text_field_name = text_field_name
        self.label_field_name = label_field_name
        self.samples: list[tuple[list[int], int]] = []
        for sample in hf_dataset:
            text: str = sample.get(text_field_name, "")
            label = sample.get(label_field_name, None)
            if not text or not text.strip() or label is None:
                continue
            if not isinstance(text, str) or not isinstance(label, int):
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_text_len:
                self.samples.append((tokens, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError()

        tokens, label = self.samples[idx]
        return tensor(tokens), tensor(label)


class StreamingDatasetForClassification(IterableDataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Tokenizer,
        max_text_len: int,
        text_field_name: str = "text",
        label_field_name: str = "label",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.text_field_name = text_field_name
        self.label_field_name = label_field_name

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for sample in self.hf_dataset:
            text: str = sample.get(self.text_field_name, "")
            label: int = sample.get(self.label_field_name, None)
            if not text or not text.strip() or label is None:
                continue
            if not isinstance(text, str):
                # TODO Generate a warning or raise an exception.
                continue
            if not isinstance(label, int):
                # TODO Maybe allow float as well?
                # TODO Generate a warning or raise an exception.
                continue

            if len(text) > self.max_text_len:
                # TODO Should we generate a message saying we ignored this field?
                continue

            tokens = self.tokenizer.encode(text)

            yield tensor(tokens), tensor(label)


def create_dataloader(hf_dataset, tokenizer, batch_size, max_text_len):
    def collate_fn(batch):
        # Separate sequences and labels
        # Assumes dataset __getitem__ returns (tensor, label)
        sequences = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])

        # Pad sequences to match the longest one in this batch
        # batch_first=True makes shape (Batch, Seq_Len, Features)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

        return padded_sequences, labels

    dataset = DatasetForClassification(hf_dataset, tokenizer, max_text_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
