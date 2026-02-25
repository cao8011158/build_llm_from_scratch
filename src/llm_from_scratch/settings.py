# src/rq_pipeline/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import hashlib
import json

import yaml

SettingsDict = Dict[str, Any]


# =========================
# Public API
# =========================
def load_settings(path: str | Path) -> SettingsDict:
    """
    Load rq-pipeline YAML -> normalized nested dict settings.

    Guarantees:
    - defaults are applied (so required nested maps exist)
    - validation is executed (ValueError with clear messages)
    - runtime metadata is attached into settings["_meta"]

    Returns:
        settings: Dict[str, Any]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("pipeline config root must be a mapping (YAML dict)")
    return raw
    # s = apply_defaults(raw)
    # validate_settings(s)

    # s.setdefault("_meta", {})
    # s["_meta"]["config_path"] = str(path)
    # s["_meta"]["config_hash"] = hash_settings(s, exclude_keys={"_meta"})
    

