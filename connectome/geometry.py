"""Neuron geometry from the common schema: arbor centroids and dominant neuropil per neuron.

Pattern taken from synaptera@c94796d:scripts/prepare_neuron_geometry.py (a one-off cache for the video and the
boundary-layer column rule). The reductions are pure functions of arrays so they work for any dataset; the
file formats of the FlyWire Codex downloads are handled by the adapters, not here.

Axes (FlyWire, checked on neuropil medians in v1): x grows towards the fly's right, y ventral, z posterior.
"""
from __future__ import annotations

import numpy as np

from connectome.schema import IdIndex


def arbor_centroids(index: IdIndex, syn_neuron_ids: np.ndarray, syn_xyz: np.ndarray, fallback_xyz=None,
                    fallback_ok=None) -> dict:
    """Mean synapse position per neuron (all synapses it takes part in).

    syn_neuron_ids: (m,) neuron id of each synapse end (pre and post ends stacked by the caller);
    syn_xyz: (m, 3). Neurons without synapses take ``fallback_xyz`` where ``fallback_ok``. Returns
    {xyz (n, 3) float32 with NaN where unknown, n_syn (n,), source (n,) int8: 1 synapses, 2 fallback, 0 none}."""
    n = len(index)
    idx = index.index_of(syn_neuron_ids, missing="-1")
    ok = idx >= 0
    cnt = np.bincount(idx[ok], minlength=n)
    xyz = np.full((n, 3), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(3):
            xyz[:, j] = np.bincount(idx[ok], weights=np.asarray(syn_xyz, float)[ok, j], minlength=n) / cnt
    has = cnt > 0
    source = np.where(has, 1, 0).astype(np.int8)
    if fallback_xyz is not None:
        use = ~has & np.asarray(fallback_ok, bool)
        xyz[use] = np.asarray(fallback_xyz, float)[use]
        source[use] = 2
    return dict(xyz=xyz.astype(np.float32), n_syn=cnt.astype(np.int32), source=source)


def dominant_neuropil(index: IdIndex, neuron_ids, neuropil_names, synapse_counts) -> dict:
    """Dominant neuropil (share of input + output synapses) per engine index.

    neuron_ids: (k,) ids of the rows of ``synapse_counts`` (k, n_neuropil), columns named ``neuropil_names`` with
    a hemisphere suffix ``_L`` / ``_R`` (no suffix = central). Returns {neuropils (base names), dominant (index
    into neuropils, -1 none), side ("L" / "R" / "C" / ""), frac}."""
    import re
    names = list(neuropil_names)
    base = [re.sub(r"_(L|R)$", "", nm) for nm in names]
    side = np.array([nm[-1] if re.search(r"_(L|R)$", nm) else "C" for nm in names])
    bases = sorted(set(base))
    cnt = np.asarray(synapse_counts, float)
    rows = index.index_of(neuron_ids, missing="-1")
    ok = rows >= 0
    n = len(index)
    dom = np.full(n, -1, np.int16)
    dside = np.full(n, "", "<U1")
    frac = np.zeros(n, np.float32)
    s = cnt.sum(1)
    j = cnt.argmax(1)
    has = ok & (s > 0)
    dom[rows[has]] = [bases.index(base[k]) for k in j[has]]
    dside[rows[has]] = side[j[has]]
    frac[rows[has]] = cnt[has, j[has]] / s[has]
    return dict(neuropils=np.array(bases), dominant=dom, side=dside, frac=frac)
