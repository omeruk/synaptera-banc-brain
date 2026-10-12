"""Pre-registered brain readouts, resolved through the crosswalk and gated by symmetry (review H2).

Ported from synaptera@c94796d:flight/readouts.py. Types come from configs/readouts.yaml and are located by
cell-type name in the neuron table of the dataset (MN9 by its dataset alias, connectome.crosswalk). Changes:
  * no hard-wired file names or FlyWire row indices: sets are engine indices of a Connectome;
  * every lateral readout passes ``connectome.qa.symmetry_gate`` before it can drive a command; a failing
    type is *recorded and rejected* (``GateFailure`` on use) unless the caller names it in ``waive``, which is
    then written to the run record. The v1 post-hoc choice of DNp15 as the sole yaw drive is not inherited;
  * rates are read from engine groups (``NeuralEngine.rates``), no per-step spike-count vector is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from configs import load
from connectome import crosswalk as cwk
from connectome import qa
from connectome.schema import Connectome

CFG = load("readouts")
READOUT_TYPES = tuple(CFG["readout_types"])
STEER_TYPES = tuple(CFG["steer_types"])
SIDES = ("L", "R")
_ALIAS = {"MN9": {"flywire783": r"^CB0701$", "banc": r"^MN9$"}}


class GateFailure(RuntimeError):
    """A readout that failed the symmetry gate was used without a waiver."""


def _query(t: str) -> cwk.TypeQuery:
    return cwk.TypeQuery(t, _ALIAS.get(t, {"*": rf"^{re.escape(t)}$"}))


def locate(cx: Connectome, dataset: str, types=READOUT_TYPES) -> dict:
    """{type: {"L": engine indices, "R": engine indices}} by crosswalk name match on primary cell types.
    A type without neurons on one side raises (v1 behaviour)."""
    cw = cwk.crosswalk_by_name({dataset: cx.neurons}, [_query(t) for t in types])
    out = {}
    for t in types:
        rows = cw[(cw["query"] == t) & (cw["source"] == "name")]
        sets = {}
        for s, full in (("L", "left"), ("R", "right")):
            sets[s] = np.sort(cx.index.index_of(rows.loc[rows["side"] == full, "id"].to_numpy()))
        if not (len(sets["L"]) and len(sets["R"])):
            raise ValueError(f"readout {t}: missing side")
        out[t] = sets
    return out


@dataclass
class Readouts:
    """Readout sets of one connectome, with the symmetry-gate result of every type.

    dataset: crosswalk dataset name (``flywire783`` / ``banc``). waive: types allowed to drive a command although
    they failed the gate (recorded in :meth:`record`)."""

    cx: Connectome
    dataset: str
    types: tuple = READOUT_TYPES
    waive: tuple = ()
    sets: dict = field(init=False)
    gate: dict = field(init=False)
    n: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sets = locate(self.cx, self.dataset, self.types)
        cache = qa.input_synapses(self.cx)
        self.gate = {t: qa.symmetry_gate(self.cx, self.sets[t]["L"], self.sets[t]["R"], name=t, _cache=cache)
                     for t in self.types}
        self.n = np.array([[len(self.sets[t][s]) for s in SIDES] for t in self.types])

    # ── gate ────────────────────────────────────────────────────────────────
    def usable(self, t: str) -> bool:
        return self.gate[t].passed or t in self.waive

    def require(self, t: str) -> None:
        if not self.usable(t):
            g = self.gate[t]
            raise GateFailure(f"readout {t} failed the symmetry gate (total log2 R/L {g.log2_total:+.2f}, "
                              f"excitatory {g.log2_exc:+.2f}, threshold {g.threshold}); name it in `waive` to "
                              "use it as a labelled exception")

    def usable_types(self, among=STEER_TYPES) -> tuple:
        return tuple(t for t in among if t in self.types and self.usable(t))

    # ── engine groups and rates ─────────────────────────────────────────────
    def group_names(self) -> dict:
        """{engine group name: neuron indices} to register with the engine."""
        return {f"ro_{t}_{s}": self.sets[t][s] for t in self.types for s in SIDES}

    def hz(self, engine, window_ms: float) -> np.ndarray:
        """(n_types, 2) mean firing rate per neuron (Hz) over the last ``window_ms``, columns L, R."""
        names = list(self.group_names())
        r = engine.rates(names, window_ms)
        return np.array([[float(np.mean(r[f"ro_{t}_{s}"])) for s in SIDES] for t in self.types])

    def counts(self, per_neuron: np.ndarray) -> np.ndarray:
        """(n_types, 2) integer spike counts from a per-neuron count vector (replay / tests)."""
        return np.array([[int(per_neuron[self.sets[t][s]].sum()) for s in SIDES] for t in self.types])

    def rates(self, counts, dt):
        """Counts in an interval of ``dt`` s -> Hz per neuron."""
        return counts / (np.maximum(self.n, 1) * dt)

    def row(self, t: str) -> int:
        return self.types.index(t)

    def record(self) -> dict:
        """The gate table and waivers, for the run record (CLAUDE.md rule 5: the result is reported)."""
        return dict(threshold=qa.SYMMETRY_THRESHOLD, waived=list(self.waive),
                    gate={t: g.as_dict() for t, g in self.gate.items()}, n=self.n.tolist(),
                    types=list(self.types))
