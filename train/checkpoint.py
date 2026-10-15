"""Atomic, resumable training checkpoints (docs/10_TRAINING.md section 10; prereg/14 gate 3, resume check).

A checkpoint holds the parameters, the optimiser state, the torch / NumPy / Python RNG states, the step counter and a
caller-defined ``extra`` dict (data-order position, DAgger round, hashes). It is written to a temporary file and renamed,
so a crash never leaves a half-written checkpoint under the final name. The last ``KEEP_LAST`` step checkpoints are
kept, plus ``best.pt``.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch

KEEP_LAST = 3
_STEP_RE = re.compile(r"^step_(\d{8})\.pt$")


def rng_state() -> dict:
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(run_dir, step: int, model: torch.nn.Module, optimiser: torch.optim.Optimizer, extra: dict | None = None,
         best: bool = False) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    obj = dict(step=int(step), model=model.state_dict(), optimiser=optimiser.state_dict(), rng=rng_state(),
               extra=dict(extra or {}))
    path = run_dir / f"step_{step:08d}.pt"
    _atomic_save(obj, path)
    if best:
        _atomic_save(obj, run_dir / "best.pt")
    for old in step_checkpoints(run_dir)[:-KEEP_LAST]:
        old.unlink()
    return path


def step_checkpoints(run_dir) -> list[Path]:
    """Step checkpoints in step order."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    return sorted(p for p in run_dir.iterdir() if _STEP_RE.match(p.name))


def latest(run_dir) -> Path | None:
    c = step_checkpoints(run_dir)
    return c[-1] if c else None


def load(path, model: torch.nn.Module, optimiser: torch.optim.Optimizer) -> dict:
    """Restore model, optimiser and RNG states; returns the checkpoint (``step``, ``extra``)."""
    obj = torch.load(path, weights_only=False)
    model.load_state_dict(obj["model"])
    optimiser.load_state_dict(obj["optimiser"])
    set_rng_state(obj["rng"])
    return obj
