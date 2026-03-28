import tiktoken


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self._encoding = tiktoken.get_encoding(encoding_name=encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)
