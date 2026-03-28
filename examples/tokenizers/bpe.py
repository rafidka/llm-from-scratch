from llm_from_scratch.tokenizers.bpe import BPETokenizer

corpus = "the cat sat on the mat. the cat ate the rat. the rat sat on the mat."
tokenizer = BPETokenizer(corpus, 3)

token_ids = tokenizer.encode(corpus)
print(token_ids)

tokens = tokenizer.decode(token_ids)
print("".join(tokens))
