import re
from collections import Counter
from dataclasses import dataclass

UNK = 0
UNK_STR = "<|unk|>"
EOS = 1
EOS_STR = "<|eos|>"


class BPETokenizer:
    def __init__(self, corpus: str, num_merges: int):
        self._train(corpus, num_merges)

    def _train(self, corpus: str, num_merges: int):
        words = self._tokenize(corpus)
        freqs = Counter(words)
        self.mappings = {tuple(word): count for word, count in freqs.items()}

        self.merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            self._merge_loop()

        self.tokens = sorted(
            list(
                set(corpus)
                | set(token for tokens in self.mappings.keys() for token in tokens)
            )
        )
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        token_id = 100  # Token 0 is UNK; 1 is EOS; 2 to 99 are left for custom tokens.
        for token in self.tokens:
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            token_id += 1

        # Add UNK and EOS
        self._token_to_id[UNK_STR] = UNK
        self._token_to_id[EOS_STR] = EOS
        self._id_to_token[UNK] = UNK_STR
        self._id_to_token[EOS] = EOS_STR

    def _tokenize(self, text: str) -> tuple[str, ...]:
        # split_by = r'([,.:;?_!"()\']|--|\s)'
        split_by = r"( ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+)"

        return tuple(token for token in re.split(split_by, text) if token)

    def _merge_loop(self):
        @dataclass
        class _PairInfo:
            count: int
            found_in: set[tuple[str, ...]]

        # Find the most frequent pair.
        pairs_info: dict[tuple[str, str], _PairInfo] = {}
        for tokens, count in self.mappings.items():
            for t1, t2 in zip(tokens, tokens[1:]):
                pair = (t1, t2)
                if pair in pairs_info:
                    pair_info = pairs_info[pair]
                    pair_info.count += count
                    pair_info.found_in.add(tokens)
                else:
                    pairs_info[pair] = _PairInfo(count, {tokens})

        if not pairs_info:
            return  # No more pairs to merge

        most_freq_pair = max(pairs_info.items(), key=lambda p: p[1].count)

        pair, pair_info = most_freq_pair
        self.merges.append(pair)
        for tokens in pair_info.found_in:
            new_tokens = self._apply_merge(tokens, pair)
            self.mappings[new_tokens] = self.mappings[tokens]
            del self.mappings[tokens]

    def _apply_merge(
        self,
        tokens: tuple[str, ...],
        pair_to_merge: tuple[str, str],
    ) -> tuple[str, ...]:
        merged_str = pair_to_merge[0] + pair_to_merge[1]

        new_tokens: list[str] = []
        idx = 0
        while idx < len(tokens):
            if (
                idx < len(tokens) - 1
                and (tokens[idx], tokens[idx + 1]) == pair_to_merge
            ):
                new_tokens.append(merged_str)
                idx += 2
            else:
                new_tokens.append(tokens[idx])
                idx += 1

        return tuple(new_tokens)

    def encode(self, text: str) -> tuple[int, ...]:
        tokens: list[str] = []
        for token in self._tokenize(text):
            sub_tokens = tuple(token)

            for merge in self.merges:
                sub_tokens = self._apply_merge(sub_tokens, merge)

            tokens.extend(sub_tokens)

        # Convert to IDs
        return tuple(self._token_to_id.get(token, UNK) for token in tokens)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        def id_to_token(token_id: int) -> str:
            if token_id not in self._id_to_token:
                raise RuntimeError(f"Invalid token id: {token_id}")
            return self._id_to_token[token_id]

        # Convert to IDs
        return "".join(id_to_token(token_id) for token_id in token_ids)
