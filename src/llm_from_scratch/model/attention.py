import math
import torch
from torch import Tensor

from llm_from_scratch.model.ops.numerically_stable_softmax import softmax



def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    Shapes:
      Q: (..., n, d_k)
      K: (..., m, d_k)
      V: (..., m, d_v)
      mask (optional): (n, m) boolean where True = allow attention, False = block

    Returns:
      (..., n, d_v)
    """
    # --- basic shape checks (lightweight) ---
    if Q.size(-1) != K.size(-1):
        raise ValueError(f"Q and K must have same d_k, got {Q.size(-1)} vs {K.size(-1)}")
    if K.size(-2) != V.size(-2):
        raise ValueError(f"K and V must have same seq_len (m), got {K.size(-2)} vs {V.size(-2)}")

    d_k = Q.size(-1)

    # 1) scores = Q K^T / sqrt(d_k)
    #    (..., n, d_k) @ (..., d_k, m) -> (..., n, m)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2) apply mask by adding -inf where mask is False
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be boolean, got {mask.dtype}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (n, m), got dim={mask.dim()} shape={tuple(mask.shape)}")
        n, m = scores.size(-2), scores.size(-1)
        if mask.shape != (n, m):
            raise ValueError(f"mask shape must be (n, m)=({n},{m}), got {tuple(mask.shape)}")

        # broadcast mask to scores: (..., n, m)
        # where mask == False -> set score to -inf so softmax gives 0 prob
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3) attention probabilities along keys dimension (last dim = m)
    attn = softmax(scores, dim=-1)

    # 4) weighted sum: (..., n, m) @ (..., m, d_v) -> (..., n, d_v)
    out = torch.matmul(attn, V)
    return out
