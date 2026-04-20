import torch

from llm_from_scratch.model.causallm import GPTForCausalLM
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.utils import get_device

device = get_device()
model_data = torch.load(
    "./checkpoints/checkpoint_epoch8_step4500.pt",
    map_location=device,
)
model_state_dict = model_data["model_state_dict"]
tokenizer = TiktokenTokenizer()
model = GPTForCausalLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=1280,
    num_heads=20,
    num_layers=36,
    max_seq_len=1024,
    dropout=0.0,
)
model.load_state_dict(model_state_dict)
model.to(device)

with torch.no_grad():
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower().strip() == "exit":
            print("Bye!")
            break
        prompt = f"### Instruction:\n{user_query}\n\n### Response:\n"
        input_ids = torch.tensor(tokenizer.encode(prompt), device=device).view(1, -1)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.1,
            top_k=10,
            eos_token_id=tokenizer.encode("<|endoftext|>")[0],
        )
        print(tokenizer.decode(output_ids[0].tolist()))
