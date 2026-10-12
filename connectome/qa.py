"""Quality gates of the connectome data layer (docs/05_ARCHITECTURE_PLAN.md 3.0, CLAUDE.md rule 5).

The symmetry gate generalises scripts/analysis/dn_lr_asymmetry.py: a left / right readout or path may only be
used in an experiment when the input-synapse ratio of its two sides satisfies |log2(R/L)| < threshold, for the
total and for the excitatory input (review H2: v1 chose DNp15, R/L = 1.66x, without checking). The result is
always reported, pass or fail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from connectome.schema import Connectome

SYMMETRY_THRESHOLD = 0.5      # |log2(R/L)|; configs/assumptions.json: connectome.symmetry_threshold


def input_synapses(cx: Connectome) -> tuple[np.ndarray, np.ndarray]:
    """(total, excitatory) incoming synapses per neuron (engine order). Excitatory needs a signed graph and
    counts edges with sign > 0; an unsigned graph returns NaN for it."""
    _, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.float64)
    total = np.bincount(post, weights=w, minlength=cx.n_neurons)
    if not cx.signed:
        return total, np.full(cx.n_neurons, np.nan)
    exc = np.bincount(post, weights=np.where(cx.edges["sign"].to_numpy() > 0, w, 0.0), minlength=cx.n_neurons)
    return total, exc


@dataclass(frozen=True)
class GateResult:
    name: str
    n_left: int
    n_right: int
    in_left: float
    in_right: float
    exc_left: float
    exc_right: float
    log2_total: float
    log2_exc: float
    threshold: float
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def symmetry_gate(cx: Connectome, left, right, name: str = "", threshold: float = SYMMETRY_THRESHOLD,
                  _cache: tuple | None = None) -> GateResult:
    """Gate of one left / right pair of neuron sets (engine indices). A side without neurons, or without any
    input, fails (the ratio is undefined)."""
    total, exc = _cache if _cache is not None else input_synapses(cx)
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    tl, tr = float(total[left].sum()), float(total[right].sum())
    el, er = float(np.nansum(exc[left])), float(np.nansum(exc[right]))

    def lg(r, l):
        return float(np.log2(r / l)) if r > 0 and l > 0 else float("nan")

    lt = lg(tr, tl)
    le = lg(er, el) if cx.signed else float("nan")
    ok = len(left) > 0 and len(right) > 0 and np.isfinite(lt) and abs(lt) < threshold
    if cx.signed:
        ok = ok and np.isfinite(le) and abs(le) < threshold
    return GateResult(name, len(left), len(right), tl, tr, el, er, lt, le, threshold, bool(ok))
