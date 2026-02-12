import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    A bias-free linear layer:
        y = x @ W^T

    Parameters
    ----------
    in_features : int
    out_features : int
    device : torch.device | None
    dtype : torch.dtype | None
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        # Store W (NOT transposed)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using truncated normal.
        Modern transformer default:
            std = 1 / sqrt(in_features)
        """
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.trunc_normal_(self.W, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            (..., in_features)

        output:
            (..., out_features)
        """
        return x @ self.W.T
