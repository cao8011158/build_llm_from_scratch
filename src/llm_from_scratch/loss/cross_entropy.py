# src/llm_from_scratch/loss/cross_entropy.py

from __future__ import annotations

import torch
from torch import Tensor


def cross_entropy_loss(
    logits: Tensor,      # (..., vocab_size)
    targets: Tensor,     # (...)
) -> Tensor:
    """
    Numerically-stable cross entropy loss.

    Computes:
        ℓ = -log softmax(logits)[targets]

    Assumptions:
        - Batch-like dimensions come first.
        - Vocabulary dimension is the last dimension.
        - targets.shape == logits.shape[:-1]

    Returns:
        Scalar tensor (mean loss over all batch elements).
    """

    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dimension, got {logits.ndim}")

    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape must match logits.shape[:-1], "
            f"got targets={targets.shape}, logits={logits.shape}"
        )

    if targets.dtype not in (torch.int32, torch.int64):
        targets = targets.long()

    # =========================================================
    # 1️⃣ Subtract max for numerical stability
    # =========================================================
    # logsumexp trick: subtract max before exp
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    # =========================================================
    # 2️⃣ Compute logsumexp in stable form
    # =========================================================
    exp_shifted = torch.exp(shifted_logits)
    sum_exp = exp_shifted.sum(dim=-1)
    logsumexp = torch.log(sum_exp)

    # =========================================================
    # 3️⃣ Extract logits at target positions
    # =========================================================
    target_indices = targets.unsqueeze(-1)
    target_logits = shifted_logits.gather(dim=-1, index=target_indices)
    target_logits = target_logits.squeeze(-1)

    # =========================================================
    # 4️⃣ Cross entropy
    # ℓ = - target_logit + logsumexp
    # =========================================================
    loss = -target_logits + logsumexp

    # =========================================================
    # 5️⃣ Return mean over batch-like dimensions
    # =========================================================
    return loss.mean()
