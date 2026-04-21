import torch
from torch.nn import functional as F

from llm_from_scratch.model.norm import LayerNorm, RMSNorm

torch.manual_seed(42)


def layer_norm():
    print("LayerNorm")
    x = torch.randn(5, 2, 3)
    ln = LayerNorm(3)
    my_norm = ln(x)
    print(my_norm)
    torch_norm = F.layer_norm(x, (3,), ln.weight, ln.bias, ln.eps)
    print(torch_norm)
    print(torch.allclose(my_norm, torch_norm))


def rms_norm():
    print("RMSNorm")
    x = torch.randn(5, 2, 3)
    ln = RMSNorm(3)
    my_norm = ln(x)
    print(my_norm)
    torch_norm = F.rms_norm(x, (3,), ln.weight, ln.eps)
    print(torch_norm)
    print(torch.allclose(my_norm, torch_norm))


layer_norm()
rms_norm()
