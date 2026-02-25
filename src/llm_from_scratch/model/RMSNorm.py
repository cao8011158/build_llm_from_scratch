import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # 可学习 gain 参数 g_i
        # shape: (d_model,)
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始 dtype（FP16/BF16 训练非常重要）
        in_dtype = x.dtype

        # 👉 upcast 防止平方溢出
        x = x.to(torch.float32)

        # ---- RMS 计算 ----
        # mean over hidden dim
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )

        # ---- 归一化 + gain ----
        y = x / rms
        y = y * self.weight

        # cast 回原 dtype
        return y.to(in_dtype)
