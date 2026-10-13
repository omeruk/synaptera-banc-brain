"""Per-neuron input normalisation of BANC to the FlyWire in-synapse level (model variant ``innorm_homolog``,
prereg/05_banc_v3_and_innorm.md section 2.1; origin P, computed from data only).

For every postsynaptic BANC neuron with a FlyWire homolog type the factor is
``s = mean FlyWire input synapses per neuron of the type / mean BANC input synapses per neuron of the type``, both
counted on the graphs the engine runs. One factor per neuron multiplies all its incoming weights (excitatory and
inhibitory alike, so the neuron's E/I balance is kept). Neurons without a homolog get the median ``s`` of their super
class; ``s`` is clipped to ``clip``.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

CLIP = (0.25, 4.0)
BANC_SUPER_CLASS = {"optic_lobe_intrinsic": "optic", "central_brain_intrinsic": "central"}
# named critical-type entries of connectome/crosswalk.py (BANC name -> FlyWire name)
NAMED = ((re.compile(r"^MN9$"), "CB0701"), (re.compile(r"^LB3[a-z]?$"), "LB3"))
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def in_synapses(cx) -> np.ndarray:
    """Input synapses (sum of ``n_syn`` over incoming pairs) per neuron, in engine index order."""
    _, post = cx.edge_endpoints()
    return np.bincount(post, weights=cx.edges["n_syn"].to_numpy(np.float64), minlength=cx.n_neurons)


def banc_type_key(cell_type) -> object:
    """FlyWire type name that a BANC primary cell type stands for (identity unless a named entry applies)."""
    if cell_type is None or (isinstance(cell_type, float) and np.isnan(cell_type)) or pd.isna(cell_type):
        return None
    for rx, name in NAMED:
        if rx.match(str(cell_type)):
            return name
    return str(cell_type)


def _summary(s: np.ndarray) -> dict:
    if len(s) == 0:
        return dict(n=0)
    q = np.quantile(s, QUANTILES)
    return dict(n=int(len(s)), mean=float(s.mean()), **{f"q{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, q)})


def innorm_factors(fx, bx, clip=CLIP):
    """(s, report): ``s`` has one positive factor per neuron of ``bx`` (engine index order)."""
    fin, bin_ = in_synapses(fx), in_synapses(bx)
    ftype = fx.neurons["cell_type"].astype(object).to_numpy()
    fkey = pd.Series(ftype)
    bkey = pd.Series([banc_type_key(t) for t in bx.neurons["cell_type"].astype(object).to_numpy()], dtype=object)
    f_mean = pd.Series(fin).groupby(fkey.to_numpy()).mean()
    b_mean = pd.Series(bin_)[bkey.notna().to_numpy()].groupby(bkey[bkey.notna()].to_numpy()).mean()
    both = f_mean.index.intersection(b_mean.index)
    ok = (f_mean[both] > 0) & (b_mean[both] > 0)
    s_type = (f_mean[both] / b_mean[both])[ok]
    s_raw = bkey.map(s_type).to_numpy(dtype=np.float64)          # NaN: no homolog
    has = ~np.isnan(s_raw)
    sc = bx.neurons["super_class"].replace(BANC_SUPER_CLASS).astype(object).to_numpy()
    glob = float(np.median(s_raw[has]))
    med = {c: float(np.median(s_raw[has & (sc == c)])) for c in pd.unique(sc[pd.notna(sc)]) if (has & (sc == c)).any()}
    s = s_raw.copy()
    for i in np.flatnonzero(~has):
        s[i] = med.get(sc[i], glob)
    lo, hi = clip
    low, high = s < lo, s > hi
    s = np.clip(s, lo, hi)
    per_class = {}
    for c in pd.unique(sc[pd.notna(sc)]):
        m = sc == c
        per_class[str(c)] = dict(all=_summary(s[m]), homolog=_summary(s[m & has]), fallback=_summary(s[m & ~has]),
                                 n_clipped_low=int((low & m).sum()), n_clipped_high=int((high & m).sum()),
                                 class_median_for_fallback=med.get(c, glob))
    report = dict(
        n_neurons=int(len(s)), n_homolog=int(has.sum()), n_fallback=int((~has).sum()),
        n_homolog_types=int(len(s_type)), n_clipped_low=int(low.sum()), n_clipped_high=int(high.sum()),
        frac_clipped=float((low | high).mean()), global_median_s=glob, clip=list(clip),
        all=_summary(s), homolog=_summary(s[has]), fallback=_summary(s[~has]), by_super_class=per_class,
        named={t: (float(s_type[t]) if t in s_type.index else None) for _, t in NAMED},
        s_sha256=hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest())
    return s, report


def scaled_view(cx, s):
    """Light shim of ``cx`` whose ``edges['n_syn']`` is multiplied by ``s[post]`` (for the antennal-lobe eigenvalue
    of ``connectome.qa_report.al_structure``); the Connectome itself is not modified."""
    from types import SimpleNamespace
    _, post = cx.edge_endpoints()
    e = cx.edges.copy()
    e["n_syn"] = e["n_syn"].to_numpy(np.float64) * np.asarray(s, np.float64)[post]
    pre_post = (cx.index.index_of(cx.edges["pre"].to_numpy()), post)
    return SimpleNamespace(n_neurons=cx.n_neurons, edges=e, edge_endpoints=lambda: pre_post)
