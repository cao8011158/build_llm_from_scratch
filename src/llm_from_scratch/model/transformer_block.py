# src/llm_from_scratch/model/transformer_block.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from llm_from_scratch.model.RMSNorm import RMSNorm
from llm_from_scratch.model.positionwise_feedforward import PositionWiseFeedForward
from llm_from_scratch.model.gqa_self_attention import GroupedQuerySelfAttention


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block:

        y = x + Attn(RMSNorm(x))
        z = y + FFN(RMSNorm(y))

    Parameters (required by assignment):
      d_model: int
      num_heads: int
      d_ff: int

    We implement "MultiHeadSelfAttention" using  GroupedQuerySelfAttention by setting:
      num_q_heads = num_heads
      num_kv_heads = num_heads
    which degenerates GQA -> standard MHA behavior.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        rope_theta: float | None = None,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        # --- Pre-norm layers ---
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # --- Attention (use your GQA module as MHA) ---
        self.attn = GroupedQuerySelfAttention(
            d_model=d_model,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            rope_theta=rope_theta,
            use_rope= use_rope,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        # --- FFN (SwiGLU FFN you already implemented) ---
        self.ffn = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        x: (B, T, d_model)
        token_positions: (B, T) or (T,) or None (depending on your attention impl)

        returns: (B, T, d_model)
        """
        # 1) Attention sublayer (pre-norm)
        h = self.norm1(x)
        x = x + self.attn(h, token_positions=token_positions)

        # 2) FFN sublayer (pre-norm)
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x
