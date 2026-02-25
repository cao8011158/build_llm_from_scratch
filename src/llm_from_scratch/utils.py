# src/llm_from_scratch/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict, Tuple
import numpy as np
import random
import torch
import torch.nn as nn
from typing import Optional


SettingsDict = dict[str, Any]  # 如果你已有更严格类型就用你的


def resolve_dataset_files(settings: SettingsDict, dataset: str) -> tuple[Path, Path]:
    """
    Resolve train/valid file paths from *new* config format only:

    data:
      root: ...
      datasets:
        tinystories:
          train_file: ...
          valid_file: ...
        owt:
          train_file: ...
          valid_file: ...
    """
    data = settings["data"]
    root = Path(data["root"]).expanduser()
    ds = data["datasets"][dataset]
    train_path = root / ds["train_file"]
    valid_path = root / ds["valid_file"]
    return train_path, valid_path

def _np_dtype(dtype_str: str) -> np.dtype:
    s = dtype_str.lower()
    if s == "uint16":
        return np.uint16
    if s == "int32":
        return np.int32
    if s == "int64":
        return np.int64
    raise ValueError(f"Unknown memmap dtype: {dtype_str}")


def load_memmap(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    d = cfg["data"]
    mm = d.get("memmap", {}) or {}
    if not bool(mm.get("enabled", False)):
        raise ValueError("data.memmap.enabled must be true to use memmap training.")

    train_bin = Path(str(mm["train_bin"]))
    valid_bin = Path(str(mm["valid_bin"]))
    dtype = _np_dtype(str(mm.get("dtype", "uint16")))

    if not train_bin.exists():
        raise FileNotFoundError(f"train_bin not found: {train_bin}")
    if not valid_bin.exists():
        raise FileNotFoundError(f"valid_bin not found: {valid_bin}")

    train_data = np.memmap(train_bin, dtype=dtype, mode="r")
    val_data = np.memmap(valid_bin, dtype=dtype, mode="r")

    if train_data.ndim != 1 or val_data.ndim != 1:
        raise ValueError("memmap arrays must be 1D token-id sequences.")

    T = int(cfg["model"]["context_length"])
    if train_data.shape[0] < T + 1:
        raise ValueError(f"train.bin too small: N={train_data.shape[0]} < T+1={T+1}")
    if val_data.shape[0] < T + 1:
        raise ValueError(f"val.bin too small: N={val_data.shape[0]} < T+1={T+1}")

    return train_data, val_data

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("fp32", "float32", "f32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "f16"):
        return torch.float16
    raise ValueError(f"Unknown dtype: {dtype_str}")



def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def checkpoint_paths(cfg: Dict[str, Any], it: int) -> Tuple[Path, Optional[Path]]:
    c = cfg["checkpoint"]
    save_dir = Path(str(c["save_dir"]))
    ensure_dir(save_dir)

    prefix = str(c.get("filename_prefix", "ckpt_step"))
    filename = f"{prefix}{it:08d}.pt"
    ckpt_path = save_dir / filename

    latest_path = None
    if bool(c.get("write_latest", False)):
        latest_path = save_dir / "latest.pt"
    return ckpt_path, latest_path