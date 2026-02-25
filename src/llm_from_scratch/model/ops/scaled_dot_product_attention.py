import math
import torch
from torch import Tensor
import einx

from llm_from_scratch.model.ops.numerically_stable_softmax import softmax


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Shapes:
      Q: (..., n, d_k)
      K: (..., m, d_k)
      V: (..., m, d_v)
      mask (optional):
        - (n, m) boolean
        - (..., n, m) boolean, broadcastable to scores (..., n, m)
        where True = allow, False = block
    Returns:
      (..., n, d_v)
    """
    if Q.size(-1) != K.size(-1):
        raise ValueError(f"Q and K must have same d_k, got {Q.size(-1)} vs {K.size(-1)}")
    if K.size(-2) != V.size(-2):
        raise ValueError(f"K and V must have same seq_len (m), got {K.size(-2)} vs {V.size(-2)}")

    d_k = Q.size(-1)

    # scores: (..., n, d_k) ⋅ (..., m, d_k) over d_k -> (..., n, m)
    scores = einx.dot("... n d, ... m d -> ... n m", Q, K) / math.sqrt(d_k)

    if mask is not None:
        # dtype check / normalize raise Error if mask is not torch.bool
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be boolean, got {mask.dtype}")

        # device align (masked_fill requires same device)
        if mask.device != scores.device:
            mask = mask.to(device=scores.device)

        n, m = scores.size(-2), scores.size(-1)

        # shape check: last two dims must be (n, m)
        if mask.dim() < 2:
            raise ValueError(f"mask must have at least 2 dims, got dim={mask.dim()} shape={tuple(mask.shape)}")
        if tuple(mask.shape[-2:]) != (n, m):
            raise ValueError(f"mask last two dims must be (n, m)=({n},{m}), got {tuple(mask.shape)}")

        # broadcast-compatibility check for leading dims
        score_lead = scores.shape[:-2]
        mask_lead = mask.shape[:-2]
        try:
            torch.broadcast_shapes(score_lead, mask_lead)
        except RuntimeError as e:
            raise ValueError(
                f"mask leading dims {mask_lead} not broadcastable to scores leading dims {score_lead}"
            ) from e

        neg_inf = scores.new_full((), float("-inf"))
        scores = scores.masked_fill(~mask, neg_inf)

    attn = softmax(scores, dim=-1)

    # out: (..., n, m) ⋅ (..., m, d_v) over m -> (..., n, d_v)
    out = einx.dot("... n m, ... m dv -> ... n dv", attn, V)
    return out
