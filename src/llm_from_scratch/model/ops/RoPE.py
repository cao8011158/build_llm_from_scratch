import torch
import torch.nn as nn
import einx


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for last-dim d_k (must be even).
    Applies position-dependent 2D rotations to pairs (0,1), (2,3), ...
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        half = d_k // 2
        k = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-2.0 * k / d_k)  # (half,)

        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (L,)
        angles = pos[:, None] * inv_freq[None, :]  # (L, half)

        cos = torch.cos(angles)  # (L, half)
        sin = torch.sin(angles)  # (L, half)

        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)  (broadcast-compatible with x's leading dims)
        returns: same shape as x
        """
        assert x.size(-1) == self.d_k, f"Expected last dim {self.d_k}, got {x.size(-1)}"
        assert x.size(-2) == token_positions.size(-1), (
            f"seq_len mismatch: x has {x.size(-2)} but token_positions has {token_positions.size(-1)}"
        )

        pos = token_positions.long()
        # 注意：pos 需要在 [0, max_seq_len-1]，否则会越界
        cos = self.cos_cache[pos]  # (..., seq_len, half)
        sin = self.sin_cache[pos]  # (..., seq_len, half)

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        # 用 einx 把最后一维 d_k 拆成 (half, 2)，2 对应 (even, odd)
        # x2: (..., seq_len, half, 2)
        x2 = einx.rearrange("... s (h two) -> ... s h two", x, two=2)

        x_even = x2[..., 0]  # (..., seq_len, half)
        x_odd  = x2[..., 1]  # (..., seq_len, half)

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # 再 stack 回 (..., seq_len, half, 2)，最后用 einx 拼回 (..., seq_len, d_k)
        out2 = torch.stack((out_even, out_odd), dim=-1)  # (..., seq_len, half, 2)
        out = einx.rearrange("... s h two -> ... s (h two)", out2)

        return out
