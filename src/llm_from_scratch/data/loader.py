from torch import Tensor
from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import LLMDataset

type LLMDataLoader = DataLoader[tuple[Tensor, Tensor]]


def create_dataloader(
    dataset: LLMDataset, batch_size: int, shuffle: bool
) -> LLMDataLoader:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl
