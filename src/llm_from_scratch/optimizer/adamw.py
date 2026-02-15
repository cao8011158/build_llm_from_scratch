# src/llm_from_scratch/optim/adamw.py
from __future__ import annotations

from typing import Iterable, Tuple, Optional, Dict, Any

import math
import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer (decoupled weight decay)

    For each parameter θ, keeps state:
      - m: first moment estimate (same shape as θ)
      - v: second moment estimate (same shape as θ)
      - step: integer timestep t (starts at 0; we increment before using)

    Update (t starts at 1):
      m = β1 m + (1-β1) g
      v = β2 v + (1-β2) g^2
      α_t = α * sqrt(1-β2^t) / (1-β1^t)
      θ = θ - α_t * m / (sqrt(v) + ε)
      θ = θ - α * λ * θ          (decoupled weight decay using base lr α)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,                                # α
        betas: Tuple[float, float] = (0.9, 0.999),       # (β1, β2)
        eps: float = 1e-8,                               # ϵ
        weight_decay: float = 0.0,                       # λ
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {beta2}")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
              # Skip parameters that did not receive gradients in this backward pass
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("This AdamW implementation does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    # Initialize m, v, step
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m: Tensor = state["m"]
                v: Tensor = state["v"]

                # t starts at 1 in the algorithm: increment before using
                state["step"] += 1
                t: int = state["step"]

                # m ← β1 m + (1-β1) g
                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # v ← β2 v + (1-β2) g^2
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # α_t ← α * sqrt(1-β2^t) / (1-β1^t)
                bias_correction1 = 1.0 - (beta1 ** t)
                bias_correction2 = 1.0 - (beta2 ** t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # θ ← θ − α_t * m / (sqrt(v) + ε)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

                # θ ← θ − α * λ * θ   (decoupled weight decay, using base lr α)
                if weight_decay != 0.0:
                    p.add_(p, alpha=-(lr * weight_decay))

        return loss
