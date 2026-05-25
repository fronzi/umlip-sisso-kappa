"""I/O and checkpoint utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------- logging
def get_logger(name: str = "umlip_kappa", level: int = logging.INFO) -> logging.Logger:
    """Singleton-ish logger with a sane format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        h.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


# ----------------------------------------------------------------- config I/O
def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------------------------------------------- checkpointing
def material_dir(output_dir: str | os.PathLike, mp_id: str) -> Path:
    return ensure_dir(Path(output_dir) / mp_id)


def checkpoint_path(output_dir: str | os.PathLike, mp_id: str, tag: str) -> Path:
    return material_dir(output_dir, mp_id) / f"{tag}.json"


def has_checkpoint(output_dir: str | os.PathLike, mp_id: str, tag: str) -> bool:
    return checkpoint_path(output_dir, mp_id, tag).exists()


def save_checkpoint(output_dir: str | os.PathLike, mp_id: str, tag: str, payload: Any) -> None:
    """JSON dump that handles numpy arrays and dataclasses."""

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Unhandled type: {type(obj)!r}")

    path = checkpoint_path(output_dir, mp_id, tag)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_convert)


def load_checkpoint(output_dir: str | os.PathLike, mp_id: str, tag: str) -> Any:
    with open(checkpoint_path(output_dir, mp_id, tag), "r") as f:
        return json.load(f)


# ------------------------------------------------------------------ hashing
def structure_hash(structure) -> str:
    """Deterministic short hash of a pymatgen structure for caching."""
    s = structure.as_dict() if hasattr(structure, "as_dict") else str(structure)
    return hashlib.sha1(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()[:12]
