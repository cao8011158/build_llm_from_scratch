from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 读取与该参数相关的状态
                t = state.get("t", 0)  # 获取当前迭代次数

                grad = p.grad.data  # 梯度

                # 参数更新
                p.data -= lr / math.sqrt(t + 1) * grad

                # 更新迭代计数
                state["t"] = t + 1

        return loss
