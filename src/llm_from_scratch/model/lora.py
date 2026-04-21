from torch import Tensor, nn


class LoRALayer(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, sigma: float = 0.02):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.a = nn.Parameter(
            linear.weight.new_empty(rank, linear.in_features).normal_(0.0, sigma)
        )
        self.b = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank))

        self.linear.requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.linear.requires_grad_(False)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.linear.requires_grad_(False)
        return self

    def forward(self, x: Tensor) -> Tensor:
        # x: [<batches>, in_features]
        # weight: [out_features, in_features]
        linear_out = self.linear(x)
        lora_out = (x @ self.a.t()) @ self.b.t()
        return linear_out + lora_out * self.scaling
