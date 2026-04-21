import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int], eps: float = 1e-05):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: Tensor):
        if x.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            exp_shape = f"(*, {', '.join(str(_) for _ in self.normalized_shape)})"
            raise RuntimeError(f"Expected input shape {exp_shape}, but got {x.shape}.")

        norm_dims = tuple(range(-len(self.normalized_shape), 0))

        # Upcast to float32 (in case the input is fp16/bf16) during the calculation of
        # STD, as fp16/bf16 can overflow or lose precision.
        x_float = x.float()
        mean = x_float.mean(dim=norm_dims, keepdim=True)
        var = (x_float - mean).square().mean(dim=norm_dims, keepdim=True)
        std = torch.sqrt(var + self.eps)
        norm = ((x_float - mean) / std).to(x.dtype)

        return norm * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int], eps: float = 1e-05):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: Tensor):
        if x.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            exp_shape = f"(*, {', '.join(str(_) for _ in self.normalized_shape)})"
            raise RuntimeError(f"Expected input shape {exp_shape}, but got {x.shape}.")

        norm_dims = tuple(range(-len(self.normalized_shape), 0))

        # Upcast to float32 (in case the input is fp16/bf16) during the calculation of
        # RMS, as fp16/bf16 can overflow or lose precision.
        ms = x.float().square().mean(dim=norm_dims, keepdim=True)
        rms = torch.sqrt(ms + self.eps).to(x.dtype)

        norm = x / rms

        return norm * self.weight
