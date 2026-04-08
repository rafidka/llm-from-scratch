from math import floor

from torch.utils.data import Dataset
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

    @classmethod
    def from_tokens(cls, tokens: list[int], max_length: int, stride: int):
        """Create dataset from pre-tokenized tokens."""
        dataset = cls.__new__(cls)
        dataset.text = ""
        dataset.tokens = tokens
        dataset.max_length = max_length
        dataset.stride = stride
        dataset._len = max(
            floor((len(tokens) - max_length - 1) / stride) + 1,
            0,
        )
        return dataset

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
