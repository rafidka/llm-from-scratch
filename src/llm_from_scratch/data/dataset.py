from typing import Any, Iterator

from math import floor
from random import shuffle as random_shuffle

from torch.utils.data import Dataset, IterableDataset
from torch import Tensor, tensor

from llm_from_scratch.tokenizers.base import Tokenizer


class LLMDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        tokenizer: Tokenizer,
        text: str,
        max_length: int,
        stride: int,
    ):
        self.text = text
        self.tokens = tokenizer.encode(text)
        self.max_length = max_length
        self.stride = stride

        # tokens: 12
        # max_length: 8
        # stride: 3
        # 0-7 -> 1-8    - need 9 elements for a length of 1
        # 3-10 -> 4-11  - need 12 elements for a length of 2
        # 6-13 -> 7-14  - need 15 elements for a length of 3
        self._len = max(
            floor((len(self.tokens) - self.max_length - 1) / self.stride) + 1,
            0,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < 0 or idx >= self._len:
            raise IndexError()

        start = idx * self.stride
        end = start + self.max_length
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]
        return tensor(input_ids), tensor(target_ids)


class StreamingLLMDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Dataset that streams text from a HuggingFace dataset and yields samples on-the-fly.

    This class implements an iterable dataset that processes text data from a HuggingFace
    dataset in a streaming fashion. It tokenizes the text, maintains a buffer of tokens,
    and yields input-target pairs for language model training with a specified stride.

    Attributes:
        hf_dataset: The HuggingFace dataset object to stream text from.
        tokenizer: The tokenizer instance used to encode text into tokens.
        max_length: The maximum length of each input sequence.
        stride: The stride for sliding window sampling over the token buffer.
        buffer_limit: The maximum size of the token buffer before trimming (default: 50000).
        shuffle: Whether to shuffle the HuggingFace dataset before iterating (default: False).
        shuffle_buffer_size: Buffer size for shuffling when shuffle=True (default: 10_000).

    Yields:
        tuple[Tensor, Tensor]: A tuple containing the input tensor and the target tensor
        for each sample, where the target is the shifted version of the input.
    """

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
        buffer_limit: int = 50000,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10_000,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.buffer_limit = buffer_limit
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        token_buffer: list[int] = []
        pos = 0

        dataset = self.hf_dataset
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=None)

        for example in dataset:
            text: str = example.get("text", "")
            if not text or not text.strip():
                continue

            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)

            while pos + self.max_length + 1 <= len(token_buffer):
                input_ids = token_buffer[pos : pos + self.max_length]
                target_ids = token_buffer[pos + 1 : pos + self.max_length + 1]
                yield tensor(input_ids), tensor(target_ids)
                pos += self.stride

            if pos > self.buffer_limit:
                token_buffer = token_buffer[pos:]
                pos = 0
