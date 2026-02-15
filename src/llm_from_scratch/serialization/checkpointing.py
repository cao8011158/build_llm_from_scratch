from __future__ import annotations

import os
from typing import BinaryIO, IO, Union

import torch
from torch import nn
from torch.optim import Optimizer


PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: PathOrFile,
) -> None:
    """
    Dump model/optimizer/iteration into `out` (path or file-like object).
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(ckpt, out)


def load_checkpoint(
    src: PathOrFile,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load checkpoint from `src` and restore model/optimizer. Return saved iteration.
    """
    ckpt = torch.load(src, map_location="cpu")

    # restore
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["iteration"])
