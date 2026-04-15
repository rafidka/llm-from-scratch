import torch
import dotenv
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.data.instruction import _IGNORE_INDEX, create_dataloader
from torch.utils.data import DataLoader
from llm_from_scratch.model.pretrained import load_pretrained_lm
from llm_from_scratch.tokenizers.base import Tokenizer
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.causallm import GPTForCausalLMTrainer

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)

BATCH_SIZE = 8
EPOCHS = 5
EVAL_PROMPTS = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nExplain why the sky is blue in simple terms.\n\n### Response:\n",
    "### Instruction:\nList three benefits of exercise.\n\n### Response:\n",
]
GRAD_ACCML_STEPS = 5
LR = 5e-5
MAX_SEQ_LEN = 1024
WEIGHT_DECAY = 0.01

dotenv.load_dotenv()


def create_trainer(
    tokenizer: Tokenizer,
    max_seq_len: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dl: DataLoader,
):
    model = load_pretrained_lm("gpt2-large", max_seq_len)
    model.to(DEVICE)
    optim = AdamW(model.parameters(), weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(ignore_index=_IGNORE_INDEX)

    return GPTForCausalLMTrainer(
        model,
        tokenizer,
        optim,
        loss_fn,
        epochs,
        lr,
        dl,
        DEVICE,
        test_prompts=EVAL_PROMPTS,
        grad_accml_steps=GRAD_ACCML_STEPS,
        use_mixed_precision=True,
    )


tokenizer = TiktokenTokenizer()
ds = load_dataset("tatsu-lab/alpaca")
dl = create_dataloader(ds["train"], tokenizer, BATCH_SIZE, MAX_SEQ_LEN)
trainer = create_trainer(
    tokenizer,
    MAX_SEQ_LEN,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    dl,
)
trainer.train()
