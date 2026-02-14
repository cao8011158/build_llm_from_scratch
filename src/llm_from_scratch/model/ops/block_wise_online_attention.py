import math
import torch
from torch import Tensor


def _online_update(m: Tensor, l: Tensor, acc: Tensor, scores: Tensor, v: Tensor):
    """
    Online softmax update for a tile.

    m:   (..., q, 1) running row-wise max
    l:   (..., q, 1) running row-wise sum exp(scores - m)
    acc: (..., q, dv) running numerator sum exp(scores - m) @ v
    scores: (..., q, k) this tile's scores (already scaled + masked with -inf)
    v:      (..., k, dv) this tile's V

    Returns updated (m, l, acc).
    """
    # tile max per query row
    tile_max = scores.max(dim=-1, keepdim=True).values          # (..., q, 1)
    m_new = torch.maximum(m, tile_max)                          # (..., q, 1)

    # rescale old accumulators to new max reference
    exp_old = torch.exp(m - m_new)                              # (..., q, 1)

    # exp of new tile under new max
    p = torch.exp(scores - m_new)                               # (..., q, k)

    l_new = exp_old * l + p.sum(dim=-1, keepdim=True)           # (..., q, 1)
    acc_new = exp_old * acc + p @ v                             # (..., q, dv)

    return m_new, l_new, acc_new


def blockwise_online_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    causal: bool = True,
    q_block: int = 64,
    k_block: int = 128,
    mask: Tensor | None = None,
    upcast_accumulators: bool = True,
) -> Tensor:
    """
    Blockwise attention with online softmax (FlashAttention math), pure PyTorch.

    Shapes:
      Q: (..., Tq, d)
      K: (..., Tk, d)
      V: (..., Tk, dv)
      mask (optional): bool, broadcastable to (..., Tq, Tk). True=allow, False=block.

    Returns:
      O: (..., Tq, dv)

    Notes:
      - Does NOT materialize full (..., Tq, Tk) scores.
      - For numerical stability with fp16/bf16 inputs, set upcast_accumulators=True
        (accumulators in float32).
    """
    if Q.size(-1) != K.size(-1):
        raise ValueError(f"d mismatch: Q.d={Q.size(-1)} vs K.d={K.size(-1)}")
    if K.size(-2) != V.size(-2):
        raise ValueError(f"Tk mismatch: K.T={K.size(-2)} vs V.T={V.size(-2)}")

    d = Q.size(-1)
    Tq = Q.size(-2)
    Tk = K.size(-2)
    dv = V.size(-1)

    # choose accumulator dtype (industry standard: fp32 accum)
    acc_dtype = torch.float32 if (upcast_accumulators and Q.dtype in (torch.float16, torch.bfloat16)) else Q.dtype

    O = torch.empty(*Q.shape[:-1], dv, device=Q.device, dtype=Q.dtype)

    scale = 1.0 / math.sqrt(d)

    # iterate over query blocks
    for qs in range(0, Tq, q_block):
        qe = min(qs + q_block, Tq)
        q = Q[..., qs:qe, :]                                      # (..., q, d)
        q_len = qe - qs

        # online accumulators per query row
        m = torch.full((*q.shape[:-1], 1), float("-inf"), device=Q.device, dtype=acc_dtype)  # (..., q, 1)
        l = torch.zeros((*q.shape[:-1], 1), device=Q.device, dtype=acc_dtype)                # (..., q, 1)
        acc = torch.zeros((*q.shape[:-1], dv), device=Q.device, dtype=acc_dtype)             # (..., q, dv)

        # iterate over key blocks
        for ks in range(0, Tk, k_block):
            ke = min(ks + k_block, Tk)
            k = K[..., ks:ke, :]                                      # (..., k, d)
            v = V[..., ks:ke, :]                                      # (..., k, dv)

            # compute tile scores in acc_dtype for stability
            scores = (q.to(acc_dtype) @ k.to(acc_dtype).transpose(-1, -2)) * scale  # (..., q, k)

            # causal mask for this tile (self-attn case)
            if causal:
                qi = torch.arange(qs, qe, device=Q.device).view(-1, 1)  # (q,1)
                kj = torch.arange(ks, ke, device=Q.device).view(1, -1)  # (1,k)
                allowed = (kj <= qi)                                    # (q,k)
                scores = scores.masked_fill(~allowed, float("-inf"))

            # optional external mask (broadcastable)
            if mask is not None:
                tile_mask = mask[..., qs:qe, ks:ke]
                if tile_mask.dtype != torch.bool:
                    raise TypeError(f"mask must be bool, got {tile_mask.dtype}")
                scores = scores.masked_fill(~tile_mask, float("-inf"))

            # online update
            m, l, acc = _online_update(m, l, acc, scores, v.to(acc_dtype))

        # finalize block output
        out_block = acc / l                                          # (..., q, dv)
        O[..., qs:qe, :] = out_block.to(Q.dtype)

    return O
