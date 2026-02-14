from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import einx

from llm_from_scratch.model.ops.scaled_dot_product_attention import scaled_dot_product_attention
from llm_from_scratch.model.ops.RoPE import RotaryPositionalEmbedding
from llm_from_scratch.model.linear import Linear   


class GroupedQuerySelfAttention(nn.Module):
    """
------------------------------------------------
    Grouped Query Attention (GQA) self-attention with causal masking + RoPE.
------------------------------------------------

        num_q_heads : H                     Number of Query heads

        num_kv_heads : Hkv                  Number of Key/Value heads

        Constraint: Hq % Hkv == 0

        group_size = Hq // Hkv                   Number of Query heads sharing one KV head

        head_dim  =  d_model // Hq           Dimension per head

        Shapes:
        x: (B, T, d_model)
        token_positions: (B, T) or (T,) (broadcastable)

        Returns:
        out: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # basic structure
        # ---------------------------------------------------------
        if d_model % num_q_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_q_heads, got d_model={d_model}, num_q_heads={num_q_heads}"
            )

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads must be divisible by num_kv_heads, got {num_q_heads} and {num_kv_heads}"
            )

        self.d_model = int(d_model)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)

        self.head_dim = self.d_model // self.num_q_heads
        self.group_size = self.num_q_heads // self.num_kv_heads

        # ---------------------------------------------------------
        #  RoPE requires an even dimensionality
        # ---------------------------------------------------------
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got head_dim={self.head_dim}"
            )

        # ---------------------------------------------------------
        # Projection matrices
        # Use the customed Linear implementation   y = x @ W^T
        # Q uses Hq heads
        # K/V use Hkv heads
        # ---------------------------------------------------------


        self.WQ = Linear(
            self.d_model,
            self.num_q_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.WK = Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.WV = Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------
        # Output projection (merge heads back to d_model)  (B,T,H_q ​× d_h​​)→(B,T,d_model​)
        # ---------------------------------------------------------

        self.WO = Linear(
            self.num_q_heads * self.head_dim,
            self.d_model,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------
        # Rotary Positional Embedding
        # ---------------------------------------------------------

        self.rope = RotaryPositionalEmbedding(
            theta=float(rope_theta),
            d_k=self.head_dim,
            max_seq_len=int(max_seq_len),
            device=device,
        )

    # =============================================================
    # causal mask
    # =============================================================


    @staticmethod
    def _make_causal_mask(T: int, device: torch.device) -> Tensor:
        """
        Returns a (T, T) boolean mask where True means "allowed to attend".
        """
        return torch.ones((T, T), device=device, dtype=torch.bool).tril()

    # =============================================================
    # forward
    # =============================================================
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        x: (B, T, d_model)

        token_positions: (B, T) or (T,) (broadcastable to (B, T))
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected x to be 3D (B,T,d_model), got {tuple(x.shape)}"
            )

        B, T, _ = x.shape
        device = x.device

        Hq = self.num_q_heads
        Hkv = self.num_kv_heads
        D = self.head_dim
        G = self.group_size

        # =========================================================
        # 1️⃣ Linear projections
        # =========================================================
   
        Q = self.WQ(x)                # Q: (B,T,Hq*D)
        K = self.WK(x)                # K: (B,T,Hkv*D)
        V = self.WV(x)                # V: (B,T,Hkv*D)


        # =========================================================
        # 2️⃣ Split heads
        # =========================================================
        # (B, T, H*D) -> (B, T, H, D)

        Q = einx.rearrange("b t (h d) -> b t h d", Q, h=Hq)
        K = einx.rearrange("b t (h d) -> b t h d", K, h=Hkv)
        V = einx.rearrange("b t (h d) -> b t h d", V, h=Hkv)

        # =========================================================
        # 3️⃣ token_positions broadcast
        # =========================================================

        if token_positions.dim() == 1:
              # (T,) -> (B, T)
            token_positions_bt = token_positions.unsqueeze(0).expand(B, T)
        else:
            token_positions_bt = token_positions

        token_positions_bt = token_positions_bt.to(device=device)


        # =========================================================
        # 4️⃣ Apply RoPE
        # =========================================================
        # treat head as batch


        # ---- Q ----
        Q_bh = einx.rearrange("b t h d -> (b h) t d", Q)        # (B*Hq, T, D)
        pos_q = token_positions_bt.unsqueeze(1).expand(B, Hq, T)        # (B, Hq, T)
        pos_q = einx.rearrange("b h t -> (b h) t", pos_q)               # (B*Hq, T)
        Q_bh = self.rope(Q_bh, pos_q)
        Q = einx.rearrange("(b h) t d -> b t h d", Q_bh, b=B, h=Hq)


        # ---- K ----
        K_bh = einx.rearrange("b t h d -> (b h) t d", K)             # (B*Hkv, T, D)
        pos_k = token_positions_bt.unsqueeze(1).expand(B, Hkv, T)           # (B, Hkv, T)
        pos_k = einx.rearrange("b h t -> (b h) t", pos_k)           # (B*Hkv, T)
        K_bh = self.rope(K_bh, pos_k)   
        K = einx.rearrange("(b h) t d -> b t h d", K_bh, b=B, h=Hkv)


        # =========================================================
        # 5️⃣ Expand KV → match Q heads 
        # =========================================================
        # (B,T,Hkv,D) -> (B,T,Hkv,G,D) -> (B,T,Hkv*G,D) = (B,T,Hq,D)

        K_exp = K.unsqueeze(3).expand(B, T, Hkv, G, D)
        V_exp = V.unsqueeze(3).expand(B, T, Hkv, G, D)

        K_exp = einx.rearrange("b t hk g d -> b t (hk g) d", K_exp)     # (B, T, Hq, D)
        V_exp = einx.rearrange("b t hk g d -> b t (hk g) d", V_exp)     # (B, T, Hq, D)


        # =========================================================
        # 6️⃣ causal mask
        # =========================================================

        causal_mask = self._make_causal_mask(T, device=device)          # (T, T) bool


        # =========================================================
        # 7️⃣ attention per head
        # =========================================================

        Qh = einx.rearrange("b t h d -> (b h) t d", Q)          # (B*Hq, T, D)
        Kh = einx.rearrange("b t h d -> (b h) t d", K_exp)      # (B*Hq, T, D)
        Vh = einx.rearrange("b t h d -> (b h) t d", V_exp)      # (B*Hq, T, D)

        Oh = scaled_dot_product_attention(                  # (B*Hq, T, D)
            Qh,
            Kh,
            Vh,
            mask=causal_mask,
        )           

        # =========================================================
        # 8️⃣ merge heads + output projection
        # =========================================================
        
         # (B*Hq, T, D) -> (B, T, Hq*D) -> (B, T, d_model)
        O = einx.rearrange("(b h) t d -> b t (h d)", Oh, b=B, h=Hq)

        return self.WO(O)
