from __future__ import annotations

from typing import Iterable
import torch
from torch import nn, Tensor


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Gradient clipping using global L2 norm.

    This modifies gradients IN PLACE.

    Algorithm:
        1. Compute global L2 norm over all parameter gradients
        2. If norm > max_l2_norm:
               scale all gradients by
               max_l2_norm / (norm + eps)

    Args:
        parameters:
            Iterable of model parameters (nn.Parameter)

        max_l2_norm:
            Maximum allowed L2 norm

        eps:
            Numerical stability constant (default = 1e-6)
    """
    # -------------------------
    # Step 1: find device
    # -------------------------

    device = None

    for param in parameters:
        if param.grad is not None:
            device = param.grad.device
            break

    # no gradients → nothing to do
    if device is None:
        return
    

    # -------------------------
    # Step 2: compute global L2 norm squared
    # -------------------------

    total_norm_sq = 0.0

    total_norm_sq = torch.zeros((), device=device)

    for param in parameters:
        if param.grad is not None:
            total_norm_sq += torch.sum(param.grad * param.grad)

    total_norm = torch.sqrt(total_norm_sq)

    # global L2 norm
    total_norm = total_norm_sq ** 0.5

    # -------------------------
    # Step 3: check if clipping needed
    # -------------------------

    if total_norm <= max_l2_norm:
        return

    # scaling factor
    scale = max_l2_norm / (total_norm + eps)

    # -------------------------
    # Step 4: scale gradients IN PLACE
    # -------------------------

    for param in parameters:

        if param.grad is None:
            continue

        param.grad.mul_(scale)
