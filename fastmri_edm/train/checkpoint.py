"""Atomic checkpoints: full training states and EMA snapshots."""
import os
from pathlib import Path

import torch


def save(obj: dict, path) -> None:
    """Write a checkpoint atomically."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def state_path(run_dir, kimg: int) -> Path:
    """Path of the full training state at a given kimg."""
    return Path(run_dir) / f"state-{kimg:08d}.pt"


def ema_path(run_dir, kimg: int) -> Path:
    """Path of the EMA snapshot at a given kimg."""
    return Path(run_dir) / f"ema-{kimg:08d}.pt"


def latest(run_dir, prefix: str = "state") -> Path | None:
    """Newest checkpoint with the given prefix."""
    paths = sorted(Path(run_dir).glob(f"{prefix}-*.pt"))
    return paths[-1] if paths else None


def prune(run_dir, keep: int = 2) -> None:
    """Delete all but the newest full training states."""
    for path in sorted(Path(run_dir).glob("state-*.pt"))[:-keep]:
        path.unlink()


def load(path, device="cpu") -> dict:
    """Load a checkpoint onto a device."""
    return torch.load(path, map_location=device, weights_only=False)
