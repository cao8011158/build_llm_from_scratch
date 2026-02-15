# src/llm_from_scratch/data/get_batch.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[Tensor, Tensor]:
    """
    Sample a batch of sequences and corresponding next-token targets.

    Args:
        x : np.ndarray
            1D numpy array of token IDs, shape (N,)

        batch_size : int
            Number of sequences per batch (B)

        context_length : int
            Length of each sequence (T)

        device : str
            Device string, e.g.:
                'cpu'
                'cuda'
                'cuda:0'
                'mps'

    Returns:
        inputs  : Tensor, shape (B, T)
        targets : Tensor, shape (B, T)

        Both are torch.long and on the specified device.

    targets are inputs shifted by one token.
    """

    # --------------------------------------------------
    # Total number of tokens
    # --------------------------------------------------
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array of token IDs, got shape={x.shape}")

    N = int(x.shape[0])

    # --------------------------------------------------
    # Need at least (context_length + 1) tokens to form
    # (inputs length T) and (targets length T) with shift by 1
    # --------------------------------------------------
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    min_required = context_length + 1
    if N < min_required:
        raise ValueError(
            f"x is too short to sample a batch: N={N}, context_length={context_length} "
            f"(need at least {min_required} tokens)"
        )

    # --------------------------------------------------
    # Randomly sample starting indices
    #
    # valid range:
    #
    # start + context_length + 1 <= N
    #
    # so:
    #
    # start <= N - context_length - 1
    # --------------------------------------------------
    max_start = N - context_length - 1

    starts = np.random.randint(
        low=0,
        high=max_start + 1,
        size=(batch_size,),
    )

    # --------------------------------------------------
    # Vectorized indexing (faster than Python loops):
    #
    # idx[i, j] = starts[i] + j
    # inputs  = x[idx]
    # targets = x[idx + 1]
    # --------------------------------------------------
    offsets = np.arange(context_length, dtype=starts.dtype)          # (T,)
    idx = starts[:, None] + offsets[None, :]                         # (B, T)

    inputs_np = x[idx]
    targets_np = x[idx + 1]

    # --------------------------------------------------
    # Convert to torch tensor and move to device
    # --------------------------------------------------
    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)

    return inputs, targets
