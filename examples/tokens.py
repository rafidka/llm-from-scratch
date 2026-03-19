from llm_from_scratch.tokens import SimpleTokenizer

corpus = "Hello, world. Is this-- is this working?"
tokenizer = SimpleTokenizer(corpus)

original_text = "Hello, my name is Rafid"
tokens = tokenizer.encode(original_text)
print(tokens)

reconstructed_text = tokenizer.decode(tokens)
print("Original text: ", original_text)
print("Reconstructed text: ", reconstructed_text)
