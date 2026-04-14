from torch.utils.data import DataLoader

from llm_from_scratch.data.dataset import LLMDataset


# TODO Get rid of this later.
def create_dataloader(dataset: LLMDataset, batch_size: int, shuffle: bool):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl
