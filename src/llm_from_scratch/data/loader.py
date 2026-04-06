from torch import Tensor
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import LLMDataset

type GPTDataLoader = DataLoader[tuple[Tensor, Tensor]]


def create_dataloader(
    dataset: LLMDataset, batch_size: int, shuffle: bool
) -> GPTDataLoader:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl
