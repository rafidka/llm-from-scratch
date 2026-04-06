from llm_from_scratch.data.dataset import LLMDataset


from llm_from_scratch.data.loader import create_dataloader
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer


tokenizer = TiktokenTokenizer()
text = "the cat sat on the mat. the cat ate the rat. the rat sat on the mat."
ds = LLMDataset(tokenizer, text, 8, 3)

dl = create_dataloader(ds, batch_size=2, shuffle=False)

for inputs, targets in dl:
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)
    print()
