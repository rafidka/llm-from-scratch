"""Tokenizer implementations for text encoding and decoding."""

import re

UNK_TOKEN = "<|unk|>"  # noqa: S105
ENDOFTEXT_TOKEN = "<|endoftext|>"  # noqa: S105


class SimpleTokenizer:
    """A simple tokenizer that splits text into tokens."""

    def __init__(self, text: str) -> None:
        """Initialize the tokenizer with a training text.

        Args:
            text: The text corpus used to build the tokenizer vocabulary.
        """
        tokens = sorted(set(self._tokenize(text)))
        tokens.append(UNK_TOKEN)
        tokens.append(ENDOFTEXT_TOKEN)
        self._tokens_to_ids = {token: idx for idx, token in enumerate(tokens)}
        self._ids_to_tokens = dict(enumerate(tokens))

    def _tokenize(self, text: str) -> list[str]:
        split_by = r'([,.:;?_!"()\']|--|\s)'

        return [token.strip() for token in re.split(split_by, text) if token.strip()]

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: The text to encode.

        Returns:
            A list of integer token IDs.
        """
        tokens = self._tokenize(text)
        ids = [
            self._tokens_to_ids[token]
            if token in self._tokens_to_ids
            else self._tokens_to_ids[UNK_TOKEN]
            for token in tokens
        ]
        ids.append(self._tokens_to_ids[ENDOFTEXT_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded text string.
        """
        return " ".join(self._ids_to_tokens[id_] for id_ in ids)
