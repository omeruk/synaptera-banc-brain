"""FlyWire v783 DN spikes -> BANC v3 DN Poisson rates (docs/07_HYBRID_INTERFACE.md section 3; prereg/09).

Owner's answers to docs/07 section 5 (docs/05, after PR #12): matching by exact ``cell_type`` first, then by family
(type without a single trailing ``_<letter>``), else silent; side to same side (``center`` only to ``center``);
per-neuron mean rate of the matched FlyWire neurons (no total-spike conservation); every BANC neuron of a unit gets an
independent Poisson process of rate ``g * rho_u``; latency one coupling interval; everything recorded.

Rate of unit u for coupling interval k (``latency_intervals`` L, ``window_intervals`` W, interval length dt):

    rho_u(k) = sum_{j = k-L-W+1 .. k-L} sum_{i in FlyWire(u)} c[j, i] / (|FlyWire(u)| * W * dt)

``c[j, i]``: spike count of FlyWire DN i in interval j; intervals before 0 count as 0 (zero padding). The newest
interval used is k - L, so with L = 1 the VNC never sees spikes of the interval it is in (CLAUDE.md rule 4). L = 0 is
the ACAUSAL variant and needs ``acausal=True``.

Shuffled control: ``source`` maps a target family to the FlyWire family whose rate it receives (a fixed permutation in
the prereg); the target keeps its side. ``gain_by`` = ``"target"`` or ``"source"`` chooses whose gain applies.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SIDES = ("left", "right", "center")


def family(cell_type) -> str:
    """Type without a single trailing ``_<letter>`` (as scripts/f3_dn_crosswalk.py:family)."""
    return re.sub(r"_[a-z]$", "", str(cell_type))


@dataclass(frozen=True)
class BridgeConfig:
    families: tuple                      # BANC DN families that receive a brain rate; every other DN is silent
    gains: dict = field(default_factory=dict)   # family -> g (default 1)
    interval_ms: float = 5.0
    latency_intervals: int = 1
    window_intervals: int = 5
    source: dict = field(default_factory=dict)  # target family -> source family (shuffled control); default identity
    gain_by: str = "target"                     # "target" | "source"
    acausal: bool = False
    label: str = "BRAIN"

    def __post_init__(self):
        if self.latency_intervals < 0 or (self.latency_intervals == 0 and not self.acausal):
            raise ValueError("latency 0 is the ACAUSAL variant (acausal=True); negative latency is impossible")
        if self.window_intervals < 1:
            raise ValueError("window_intervals must be >= 1")
        if self.gain_by not in ("target", "source"):
            raise ValueError("gain_by must be 'target' or 'source'")
        if any(g < 0 for g in self.gains.values()):
            raise ValueError("negative gain")
        unknown = set(self.source) - set(self.families)
        if unknown:
            raise ValueError(f"source map for families outside the set: {sorted(unknown)}")

    def record(self) -> dict:
        d = asdict(self)
        d["families"] = list(self.families)
        d["variant"] = "ACAUSAL" if self.latency_intervals == 0 else None
        return d


def _table(df: pd.DataFrame) -> pd.DataFrame:
    need = {"cell_type", "side"}
    if not need <= set(df.columns):
        raise ValueError(f"need columns {need}")
    t = df[["cell_type", "side"]].copy()
    t["cell_type"] = t["cell_type"].astype(str)
    t["side"] = t["side"].astype(str)
    t["family"] = t["cell_type"].map(family)
    t["pos"] = np.arange(len(t))
    return t


class DNBridge:
    """Mapping table and rate transform.

    ``fw``: FlyWire DN table (``cell_type``, ``side``), one row per column of the count arrays passed to
    :meth:`rates`. ``banc``: BANC DN table (``cell_type``, ``side`` and ``index`` = engine index of the BANC neuron).
    Units are built for the BANC neurons whose family is in ``config.families``."""

    def __init__(self, fw: pd.DataFrame, banc: pd.DataFrame, config: BridgeConfig):
        self.config = config
        self.fw = _table(fw)
        b = _table(banc)
        b["index"] = np.asarray(banc["index"], dtype=np.int64)
        self.banc = b
        self.units = []
        self.unmatched_banc = []
        keyed = {}
        for _, r in b[b["family"].isin(config.families)].iterrows():
            tgt_fam = r["family"]
            src_fam = config.source.get(tgt_fam, tgt_fam)
            same_side = self.fw["side"] == r["side"]
            if src_fam == tgt_fam:
                exact = self.fw[same_side & (self.fw["cell_type"] == r["cell_type"])]
                if len(exact):
                    key, match, src = ("type", r["cell_type"], r["side"]), "type", exact
                else:
                    src = self.fw[same_side & (self.fw["family"] == tgt_fam)]
                    key, match = ("family", tgt_fam, r["side"]), "family"
            else:
                src = self.fw[same_side & (self.fw["family"] == src_fam)]
                key, match = ("shuffled", tgt_fam, src_fam, r["side"]), "shuffled"
            if len(src) == 0:
                self.unmatched_banc.append(dict(index=int(r["index"]), cell_type=r["cell_type"], side=r["side"]))
                continue
            gfam = src_fam if config.gain_by == "source" else tgt_fam
            if key not in keyed:
                keyed[key] = dict(target_family=tgt_fam, source_family=src_fam, side=r["side"], match=match,
                                  key=list(key), fw_pos=src["pos"].to_numpy(np.int64),
                                  fw_types=sorted(set(src["cell_type"])), banc_index=[], banc_types=[],
                                  gain=float(config.gains.get(gfam, 1.0)), gain_family=gfam)
                self.units.append(keyed[key])
            keyed[key]["banc_index"].append(int(r["index"]))
            keyed[key]["banc_types"].append(r["cell_type"])
        for u in self.units:
            u["banc_index"] = np.asarray(u["banc_index"], dtype=np.int64)
        used = np.zeros(len(self.fw), bool)
        for u in self.units:
            used[u["fw_pos"]] = True
        self.fw_used = used
        fam_fw = self.fw["family"].isin(set(config.source.get(f, f) for f in config.families)).to_numpy()
        self.unmatched_fw = self.fw[fam_fw & ~used][["cell_type", "side"]].to_dict("records")

    # ── targets ────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def target_index(self) -> np.ndarray:
        """Engine indices of the BANC neurons that receive a rate, in the column order of :meth:`rates`."""
        if not self.units:
            return np.zeros(0, np.int64)
        return np.concatenate([u["banc_index"] for u in self.units])

    def unit_of_target(self) -> np.ndarray:
        """Unit number of every column of :meth:`rates`."""
        return np.concatenate([np.full(len(u["banc_index"]), k) for k, u in enumerate(self.units)]) \
            if self.units else np.zeros(0, np.int64)

    # ── transform ──────────────────────────────────────────────────────────────────────────────────────────────
    def unit_rates(self, counts) -> np.ndarray:
        """(n_intervals, n_units) gain-free per-neuron rate rho_u(k) of the interval each value is APPLIED in
        (latency and window included)."""
        c = np.asarray(counts, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != len(self.fw):
            raise ValueError(f"counts must be (n_intervals, {len(self.fw)})")
        n = c.shape[0]
        L, W, dt = self.config.latency_intervals, self.config.window_intervals, self.config.interval_ms * 1e-3
        out = np.zeros((n, len(self.units)))
        for k, u in enumerate(self.units):
            s = c[:, u["fw_pos"]].sum(1)
            cs = np.concatenate([[0.0], np.cumsum(s)])
            hi = np.arange(n) - L + 1                       # exclusive end of the window in interval units
            lo = np.maximum(hi - W, 0)
            hi = np.clip(hi, 0, n)
            lo = np.minimum(lo, hi)
            out[:, k] = (cs[hi] - cs[lo]) / (len(u["fw_pos"]) * W * dt)
        return out

    def rates(self, counts) -> np.ndarray:
        """(n_intervals, n_targets) Poisson rate (Hz) of every BANC target neuron in every interval."""
        ur = self.unit_rates(counts)
        g = np.array([u["gain"] for u in self.units])
        return (ur * g)[:, self.unit_of_target()] if self.units else np.zeros((ur.shape[0], 0))

    # ── record ─────────────────────────────────────────────────────────────────────────────────────────────────
    def transfer(self, counts) -> dict:
        """Spike accounting of one record: FlyWire DN spikes used by a unit, and those of the configured source
        families that found no unit (the loss, per type)."""
        c = np.asarray(counts, dtype=np.int64).sum(0)
        src = set(self.config.source.get(f, f) for f in self.config.families)
        fam_fw = self.fw["family"].isin(src).to_numpy()
        lost = {}
        for _, r in self.fw[fam_fw & ~self.fw_used].iterrows():
            lost[r["cell_type"]] = lost.get(r["cell_type"], 0) + int(c[r["pos"]])
        tot_fam = int(c[fam_fw].sum())
        return dict(spikes_all_dn=int(c.sum()), spikes_used=int(c[self.fw_used].sum()),
                    spikes_source_families=tot_fam, spikes_lost_by_type=lost,
                    fraction_lost=(float(sum(lost.values()) / tot_fam) if tot_fam else 0.0),
                    fraction_all_dn_used=(float(c[self.fw_used].sum() / c.sum()) if c.sum() else 0.0))

    def record(self) -> dict:
        return dict(config=self.config.record(), label=self.config.label,
                    units=[dict(target_family=u["target_family"], source_family=u["source_family"], side=u["side"],
                                match=u["match"], fw_types=u["fw_types"], n_fw=int(len(u["fw_pos"])),
                                fw_pos=u["fw_pos"].tolist(), banc_types=u["banc_types"],
                                n_banc=int(len(u["banc_index"])), banc_index=u["banc_index"].tolist(),
                                gain=u["gain"], gain_family=u["gain_family"]) for u in self.units],
                    unmatched_banc=self.unmatched_banc, unmatched_fw=self.unmatched_fw,
                    statistic="per-neuron mean over the matched FlyWire neurons (no total-spike conservation)",
                    resynthesis="independent Poisson process per BANC neuron; spike-timing correlations are lost")
