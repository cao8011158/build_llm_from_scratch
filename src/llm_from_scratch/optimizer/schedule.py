# src/llm_from_scratch/optim/schedule.py
from __future__ import annotations

import math


def lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Cosine annealing learning-rate schedule with warmup (LLaMA style).

    Piecewise definition :

    Warm-up:
        if t < T_w:
            alpha_t = (t / T_w) * alpha_max

    Cosine annealing:
        if T_w <= t <= T_c:
            alpha_t = alpha_min + 0.5 * (1 + cos(((t - T_w) / (T_c - T_w)) * pi)) * (alpha_max - alpha_min)

    Post-annealing:
        if t > T_c:
            alpha_t = alpha_min

    Notes / edge cases:
      - If T_w == 0, warmup is skipped and we start cosine from t=0.
      - Requires T_c >= T_w (otherwise schedule is ill-defined).
      - t is assumed to be a non-negative integer step index.
    """
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if T_w < 0 or T_c < 0:
        raise ValueError(f"T_w and T_c must be non-negative, got T_w={T_w}, T_c={T_c}")
    if T_c < T_w:
        raise ValueError(f"Require T_c >= T_w, got T_w={T_w}, T_c={T_c}")

    # Post-annealing
    if t > T_c:
        return float(alpha_min)

    # Warm-up (skip if T_w == 0)
    if T_w > 0 and t < T_w:
        return float((t / T_w) * alpha_max)

    # Cosine annealing (covers T_w <= t <= T_c, including T_w==0)
    if T_c == T_w:
        # Degenerate: cosine interval length is 0 → immediately at alpha_min for t==T_c==T_w
        return float(alpha_min)

    progress = (t - T_w) / (T_c - T_w)  # in [0, 1]
    cosine = math.cos(progress * math.pi)
    alpha_t = alpha_min + 0.5 * (1.0 + cosine) * (alpha_max - alpha_min)
    return float(alpha_t)
