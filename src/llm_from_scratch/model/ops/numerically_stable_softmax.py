import torch
from torch import Tensor

def softmax(x: Tensor, dim: int) -> Tensor:
    # 1) 支持负 dim
    if dim < 0:
        dim = x.dim() + dim

    # 2) 范围检查（可选，但很有用）
    if not (0 <= dim < x.dim()):
        raise IndexError(f"dim out of range: dim={dim}, x.dim()={x.dim()}")

    # 3) 数值稳定 softmax
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)