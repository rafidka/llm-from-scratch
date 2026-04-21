from torch import Tensor, nn
import torch


class LoRALayer(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, sigma: float = 0.02):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.sigma = sigma
        self.a = nn.Parameter(torch.normal(0.0, sigma, (rank, linear.in_features)))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank))

        self.linear.requires_grad_(False)  # Freeze the linear layer.

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.linear.requires_grad_(False)  # keep base layer frozen no matter what
        return self

    def forward(self, x: Tensor):
        # x: [<batches>, in_features]
        # weight: [out_features, in_features]
        linear_out = self.linear(x)
        lora_out = x @ self.a.t() @ self.b.t()
        return linear_out + lora_out * self.alpha / self.rank
