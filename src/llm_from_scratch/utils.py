import torch


def get_device() -> torch.device:
    """Returns the best available torch device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The device to be used for tensor operations,
            prioritizing CUDA if available, then MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
