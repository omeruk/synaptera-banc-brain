"""Common connectome schema shared by every dataset adapter (FlyWire v783, BANC, ...).

Ported from synaptera@c94796d:flight/groups.py (the root_id <-> row-index mapping and the "groups are built
through the id -> index mapping, never by slicing" rule). The legacy keyword group rule, the hand-made DN
group set and neuron_positions of that file are NeuroFly adaptations and are not carried over.

Tables
------
neurons : one row per neuron, in the dataset's *engine order* (row position == neuron index in the engine).
    Columns NEURON_COLUMNS. Adapters may append source-specific extra columns after them (for example
    FlyWire's ``known_nt``); the schema only checks the required ones.
edges : one row per directed neuron pair.
    Columns EDGE_COLUMNS. ``pre`` / ``post`` are neuron *ids* (dataset-native int64), ``n_syn`` is the
    unsigned synapse count, ``sign`` is +1 / -1 (0 = silenced). The signed weight in synapse units is
    ``sign * n_syn``; the sign rule that produced ``sign`` is recorded in ``Connectome.meta``.

The mapping from ids to engine indices is :class:`IdIndex`; a sign rule or a group selection never relies
on position arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

NEURON_COLUMNS = ("id", "cell_type", "cell_type_alt", "cell_class", "super_class", "side", "region",
                  "nt", "nt_conf", "soma_x", "soma_y", "soma_z", "status")
EDGE_COLUMNS = ("pre", "post", "n_syn", "sign")

SIDES = ("left", "right", "center", "unknown")
REGIONS = ("brain", "SEZ", "neck", "VNC")      # NaN = not assigned by the adapter
SIDE_UNKNOWN = "unknown"


def normalise_side(values) -> pd.Series:
    """Map raw side labels to SIDES: strip + lower-case; anything that is not left/right/center -> unknown."""
    s = pd.Series(values).astype(str).str.strip().str.lower()
    s = s.where(s.isin(("left", "right", "center")), SIDE_UNKNOWN)
    return s


class IdIndex:
    """Vectorised neuron id -> engine index map (row position of the neurons table)."""

    def __init__(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ids must be a 1-D integer array")
        ids = ids.astype(np.int64, copy=False)
        order = np.argsort(ids, kind="stable")
        srt = ids[order]
        if len(srt) > 1 and (srt[1:] == srt[:-1]).any():
            raise ValueError("duplicate neuron ids")
        self._ids, self._order, self._sorted = ids, order, srt

    def __len__(self):
        return len(self._ids)

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    def index_of(self, ids, *, missing: str = "raise") -> np.ndarray:
        """Engine index of each id (same shape). missing: "raise" (KeyError) or "-1" (index -1)."""
        if missing not in ("raise", "-1"):
            raise ValueError(missing)
        ids = np.asarray(ids, dtype=np.int64)
        if len(self._sorted) == 0:
            found = np.zeros(ids.shape, bool)
            out = np.full(ids.shape, -1, np.int64)
        else:
            pos = np.minimum(np.searchsorted(self._sorted, ids), len(self._sorted) - 1)
            found = self._sorted[pos] == ids
            out = np.where(found, self._order[pos], -1)
        if missing == "raise" and not found.all():
            raise KeyError(f"{int((~found).sum())} ids not in the index, e.g. {ids[~found][:3].tolist()}")
        return out

    def group(self, ids) -> np.ndarray:
        """Strictly increasing engine indices of the ids that are present (absent ids are ignored)."""
        idx = self.index_of(ids, missing="-1")
        return np.unique(idx[idx >= 0])

    def ids_of(self, indices) -> np.ndarray:
        return self._ids[np.asarray(indices, dtype=np.int64)]


def validate_neurons(neurons: pd.DataFrame) -> None:
    miss = [c for c in NEURON_COLUMNS if c not in neurons.columns]
    if miss:
        raise ValueError(f"neurons table lacks columns {miss}")
    if not np.issubdtype(neurons["id"].dtype, np.integer):
        raise ValueError("neurons.id must be integer")
    if not neurons["id"].is_unique:
        raise ValueError("neurons.id is not unique")
    bad = set(neurons["side"].unique()) - set(SIDES)
    if bad:
        raise ValueError(f"neurons.side has values outside {SIDES}: {sorted(bad)}")
    reg = neurons["region"].dropna()
    if len(reg) and not reg.isin(REGIONS).all():
        raise ValueError(f"neurons.region has values outside {REGIONS}")


def validate_edges(edges: pd.DataFrame, index: IdIndex | None = None) -> None:
    miss = [c for c in EDGE_COLUMNS if c not in edges.columns]
    if miss:
        raise ValueError(f"edges table lacks columns {miss}")
    if len(edges) and (edges["n_syn"].to_numpy() <= 0).any():
        raise ValueError("edges.n_syn must be positive")
    if not edges["sign"].isin((-1, 0, 1)).all():
        raise ValueError("edges.sign must be in {-1, 0, +1}")
    if index is not None:
        index.index_of(edges["pre"].to_numpy())
        index.index_of(edges["post"].to_numpy())


@dataclass
class Connectome:
    """A dataset on the common schema. ``meta`` carries the source files, their hashes and the sign rule."""

    name: str
    neurons: pd.DataFrame
    edges: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_neurons(self.neurons)
        self.neurons = self.neurons.reset_index(drop=True)
        self.index = IdIndex(self.neurons["id"].to_numpy())
        validate_edges(self.edges, self.index)

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    def edge_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """(pre, post) engine indices of every edge."""
        return (self.index.index_of(self.edges["pre"].to_numpy()),
                self.index.index_of(self.edges["post"].to_numpy()))

    def signed_weights(self) -> np.ndarray:
        """sign * n_syn (synapse units), int64, edge order."""
        return (self.edges["sign"].to_numpy(np.int64) * self.edges["n_syn"].to_numpy(np.int64))

    def select(self, mask) -> np.ndarray:
        """Engine indices of the neurons where ``mask`` (bool array over the neurons table) is True."""
        return np.flatnonzero(np.asarray(mask, dtype=bool))

    def by_type(self, types: Iterable[str], *, use_alt: bool = False) -> np.ndarray:
        """Engine indices of the neurons whose cell_type (or cell_type_alt) is in ``types`` (exact match)."""
        col = "cell_type_alt" if use_alt else "cell_type"
        return self.select(self.neurons[col].isin(set(types)).to_numpy())
