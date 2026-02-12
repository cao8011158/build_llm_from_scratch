# src/llm_from_scratch/utils.py
from __future__ import annotations
from pathlib import Path
from typing import TypedDict, NotRequired, Any


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
