# llm_from_scratch/model/embedding.py
from __future__ import annotations

import torch
from torch import nn


class Embedding(nn.Module):
    """
    A minimal embedding layer that mimics torch.nn.Embedding (without using it).

    Args:
        num_embeddings: vocabulary size (V)
        embedding_dim: embedding dimension (D = d_model)
        device: optional torch.device
        dtype: optional torch.dtype (usually float32/bfloat16/float16)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        # Weight matrix shape: (V, D)
        weight = torch.empty(
            (self.num_embeddings, self.embedding_dim),
            device=device,
            dtype=dtype,
        )
        self.W = nn.Parameter(weight)

        # Init rule from spec: Embedding ~ N(0, 1) truncated to [-3, 3]
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: Long tensor with shape (...), typically (B, T)
        returns: embedding vectors with shape (..., D), typically (B, T, D)
        """
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()

        # Use advanced indexing into (V, D) -> (..., D)
        # This is allowed (requirement is: don't use nn.Embedding / F.embedding)
        return self.W[token_ids]
