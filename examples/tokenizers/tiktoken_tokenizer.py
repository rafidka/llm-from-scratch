from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer

tokenizer = TiktokenTokenizer()


text = "the cat sat on the mat. the cat ate the rat. the rat sat on the mat."
token_ids = tokenizer.encode(text)
print(token_ids)

tokens = tokenizer.decode(token_ids)
print("".join(tokens))
