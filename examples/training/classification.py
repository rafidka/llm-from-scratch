import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from llm_from_scratch.data.classification import create_dataloader
from llm_from_scratch.model.pretrained import load_pretrained_cls
from llm_from_scratch.tokenizers.tiktoken_adapter import TiktokenTokenizer
from llm_from_scratch.training.classification import GPTForClassificationTrainer

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)


def create_trainer(
    base_model_name: str,
    num_classes: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
):
    tokenizer = TiktokenTokenizer()  # Uses GPT-2 encoding by default
    imdb = load_dataset("stanfordnlp/imdb")
    train_dl = create_dataloader(imdb["train"], tokenizer, batch_size, max_seq_len)
    eval_dl = create_dataloader(imdb["test"], tokenizer, batch_size, max_seq_len)

    model = load_pretrained_cls(
        base_model_name,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
    )
    model.to(device)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()

    return GPTForClassificationTrainer(
        model,
        tokenizer,
        optim,
        loss_fn,
        epochs,
        lr,
        device,
        train_dl,
        eval_dl,
    )


if __name__ == "__main__":
    batch_size = 4
    epochs = 3
    lr = 5e-5
    num_classes = 2
    max_seq_len = 1024
    weight_decay = 0.01

    trainer = create_trainer(
        base_model_name="gpt2",
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
    )
    trainer.train()
