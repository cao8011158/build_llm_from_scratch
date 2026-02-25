from __future__ import annotations

import math
import torch
from torch import Tensor
from torch import nn

import einx  


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class PositionWiseFeedForward(nn.Module):
    """
    SwiGLU position-wise FFN (no bias), implemented with einx:

        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    Shapes:
        x:  (..., d_model)
        W1: (d_ff, d_model)
        W3: (d_ff, d_model)
        W2: (d_model, d_ff)

    d_ff ≈ (8/3) * d_model, rounded up to a multiple of 64.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)

        if d_ff is None:
            d_ff = int((8.0 / 3.0) * self.d_model)
            d_ff = _round_up_to_multiple(d_ff, 64)
        self.d_ff = int(d_ff)

        # No bias terms (modern LLM convention)
        self.W1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        def init_(w: Tensor) -> None:
            fan_in = w.shape[-1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)

        init_(self.W1)
        init_(self.W3)
        init_(self.W2)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)

        # a,b: (..., d_ff)
        # "... d, h d -> ... h" means: contract over d_model=d, produce hidden=d_ff=h
        a = einx.dot("... d, h d -> ... h", x, self.W1)
        b = einx.dot("... d, h d -> ... h", x, self.W3)

        # SiLU(a) = a * sigmoid(a), then GLU-style gating with b
        y = (a * torch.sigmoid(a)) * b  # (..., d_ff)

        # out: (..., d_model)
        # "... h, d h -> ... d" means: contract over hidden=h, produce model=d
        out = einx.dot("... h, d h -> ... d", y, self.W2)
        return out
