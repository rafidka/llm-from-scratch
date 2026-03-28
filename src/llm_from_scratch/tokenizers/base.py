from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        pass
