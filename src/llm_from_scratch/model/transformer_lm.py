# src/llm_from_scratch/model/transformer_lm.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from llm_from_scratch.model.embedding import Embedding
from llm_from_scratch.model.RMSNorm import RMSNorm
from llm_from_scratch.model.linear import Linear
from llm_from_scratch.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    Transformer Language Model (Figure 1 style):

        token_ids -> token_embedding
                +  position_embedding
                -> stack of num_layers TransformerBlock
                -> final RMSNorm
                -> vocab projection (Linear) -> logits over vocab

    Notes:
      - Returns logits (no softmax). Softmax/loss is applied outside.
      - RoPE is enabled by default (use_rope=True) and token_positions will be passed
        into each TransformerBlock (and thus into your GQA attention).
    """

    def __init__(
        self,
        *,
        vocab_size: int,            # number of tokens in the vocabulary
        d_model: int,               # hidden size of the Transformer
        num_heads: int,             # number of attention heads
        d_ff: int,                  # feed-forward network hidden dimension
        num_layers: int,            # number of stacked Transformer blocks

        use_rope: bool = False,               # whether to enable RoPE in attention
        rope_theta: float | None = None,      # base frequency for RoPE
        max_seq_len: int = 2048,              # maximum sequence length supported by RoPE/embedding
        device=None,                          # device for model parameters
        dtype=None,                           # data type of model parameters
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.use_rope = bool(use_rope)

        # ---------------------------------------------------------
        # maximum sequence length supported by RoPE/embedding
        # ---------------------------------------------------------
        self.max_seq_len = int(max_seq_len)

        # Token embedding (learned)
        self.tok_embed = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            device=device,
            dtype=dtype,
        )

        # Position embedding (learned)
        # ---------------------------------------------------------
        # IMPORTANT:
        #   - use_rope == True  -> ONLY use RoPE, DO NOT add learned PE
        #   - use_rope == False -> ONLY use learned PE, DO NOT pass token_positions
        # ---------------------------------------------------------
        if not self.use_rope:
            self.pos_embed = Embedding(
                num_embeddings=self.max_seq_len,
                embedding_dim=self.d_model,
                device=device,
                dtype=dtype,
            )
        else:
            self.pos_embed = None

        # Stack Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    max_seq_len=self.max_seq_len,
                    device=device,
                    dtype=dtype,
                    use_rope=use_rope,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output layers
        self.norm_final = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_in=self.d_model,
            d_out=self.vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: (B, T) long
        returns: logits (B, T, vocab_size)
        """
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()

        if token_ids.dim() != 2:
            raise ValueError(f"Expected token_ids shape (B, T), got {tuple(token_ids.shape)}")

        B, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")

        device = token_ids.device

        # positions
        pos_t = torch.arange(T, device=device, dtype=torch.long)   # (T,)
        pos_bt = pos_t.unsqueeze(0).expand(B, T)                   # (B, T)

        # embeddings: (B,T,D) + (T,D) -> broadcast to (B,T,D)
        x = self.tok_embed(token_ids)          # (B, T, D)

        # ---------------------------------------------------------
        # IMPORTANT:
        #   - use_rope == True  -> ONLY use RoPE, DO NOT add learned PE
        #   - use_rope == False -> ONLY use learned PE, DO NOT pass token_positions
        # ---------------------------------------------------------
        if self.use_rope:
            # ✅ pass token_positions for RoPE
            token_positions = pos_bt
        else:
            # ✅ use learned PE, do not use RoPE
            x = x + self.pos_embed(pos_t)      # (B, T, D)
            token_positions = None

        # pass token_positions into each TransformerBlock
        # TransformerBlock forward signature:
        #   def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor
        # This is correct: token_positions is optional and is only used when RoPE is enabled.
        for blk in self.blocks:
            x = blk(x, token_positions=token_positions)

        x = self.norm_final(x)                # (B, T, D)
        logits = self.lm_head(x)              # (B, T, vocab_size)
        return logits
