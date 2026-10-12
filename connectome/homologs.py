"""Homolog selection by connectivity profile, on the common schema.

Ported from synaptera@c94796d:flight/groups.py (partner_profiles, select_sugar_homologs and the shared
k-nearest-neighbour vote of select_leg_sugar_grns). The functions take a :class:`connectome.schema.Connectome`
instead of file paths, so the same code runs on FlyWire v783 and BANC.

Method (unchanged from Synaptera v1): per neuron, synapse counts to / from every partner cell type, split by
partner side relative to the neuron (ipsi / contra) and direction (in / out), L2-normalised -> a
mirror-invariant profile. A query neuron is labelled positive when the majority of its k most cosine-similar
labelled neurons are positive; accuracy is reported by leave-one-out on the labelled pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from connectome.schema import SIDE_UNKNOWN, Connectome

SUGAR_KNN = 5          # k of the kNN vote (configs/assumptions.json: connectome.sugar_knn)
_CHUNK = 2048          # queries per similarity block


def partner_profiles(ids, cx: Connectome) -> np.ndarray:
    """L2-normalised partner-type profiles, one row per id (row order = ``ids``).

    Partner type = cell_type, else cell_type_alt. A partner without a type contributes nothing. A partner
    counts as ipsilateral only when both sides are known and equal (an unknown side is contralateral, as in
    Synaptera v1)."""
    ids = np.asarray(ids, dtype=np.int64)
    rows = cx.index.index_of(ids)
    nrn = cx.neurons
    ptype = nrn["cell_type"].fillna(nrn["cell_type_alt"]).to_numpy(object)
    side = nrn["side"].to_numpy(object)
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.float64)
    wanted = np.zeros(cx.n_neurons, bool)
    wanted[rows] = True

    parts = []
    for direction, me, other in (("out", pre, post), ("in", post, pre)):
        m = wanted[me]
        me_i, ot_i, wt = me[m], other[m], w[m]
        pt = ptype[ot_i]
        ok = ~pd.isna(pt)
        ipsi = (side[ot_i] == side[me_i]) & (side[me_i] != SIDE_UNKNOWN)
        feat = pd.Series([f"{direction}|{t}|{'i' if i else 'c'}" for t, i in zip(pt[ok], ipsi[ok])], dtype=object)
        parts.append(pd.DataFrame({"row": me_i[ok], "feat": feat.to_numpy(), "w": wt[ok]}))
    f = pd.concat(parts, ignore_index=True)
    mat = f.pivot_table(index="row", columns="feat", values="w", aggfunc="sum", fill_value=0)
    x = mat.reindex(rows).fillna(0).to_numpy(float)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def knn_positive_fraction(pool: np.ndarray, positive: np.ndarray, queries: np.ndarray, k: int = SUGAR_KNN,
                          *, exclude_self: bool = False) -> np.ndarray:
    """Fraction of positives among the k pool neurons most cosine-similar to each query (ties: pool order).

    exclude_self: ``queries`` is ``pool`` and each row must not vote for itself (leave-one-out)."""
    pool = np.asarray(pool, float)
    positive = np.asarray(positive, bool)
    queries = np.asarray(queries, float)
    if len(pool) != len(positive):
        raise ValueError("pool and positive differ in length")
    kk = min(k, len(pool) - (1 if exclude_self else 0))
    out = np.empty(len(queries))
    for a in range(0, len(queries), _CHUNK):
        sim = queries[a:a + _CHUNK] @ pool.T
        if exclude_self:
            sim[np.arange(sim.shape[0]), np.arange(a, a + sim.shape[0])] = -np.inf
        top = np.argsort(-sim, axis=1, kind="stable")[:, :kk]
        out[a:a + _CHUNK] = positive[top].mean(axis=1)
    return out


def select_homologs(cx: Connectome, *, cell_type: str, seed_ids, k: int = SUGAR_KNN, labelled_side: str = "left",
                    seed_source: str = "seed") -> tuple[pd.DataFrame, dict]:
    """Seed neurons plus their homologs on the other side(s), by connectivity profile.

    Candidates = every neuron whose cell_type is ``cell_type``. Labels exist on ``labelled_side`` only: there a
    candidate is positive when its id is in ``seed_ids`` (the others are the negative class). A candidate that
    is not on the labelled side is selected when the majority of its k most similar labelled candidates are
    positive. Returns (DataFrame id, side, source, knn_frac; loo) with loo = leave-one-out counts on the
    labelled side: tp, n_positive, fp, n_negative."""
    cand = cx.neurons.loc[cx.neurons["cell_type"] == cell_type, ["id", "side"]].reset_index(drop=True)
    ids = cand["id"].to_numpy(np.int64)
    x = partner_profiles(ids, cx)
    labelled = (cand["side"] == labelled_side).to_numpy()
    positive = np.isin(ids, np.asarray(list(seed_ids), dtype=np.int64))
    li = np.flatnonzero(labelled)
    pool, pos = x[li], positive[li]

    loo = knn_positive_fraction(pool, pos, pool, k, exclude_self=True) > 0.5
    stats = dict(tp=int((loo & pos).sum()), n_positive=int(pos.sum()), fp=int((loo & ~pos).sum()),
                 n_negative=int((~pos).sum()))
    rows = [dict(id=int(i), side=labelled_side, source=seed_source, knn_frac=np.nan)
            for i in ids[labelled & positive]]
    qi = np.flatnonzero(~labelled)
    frac = knn_positive_fraction(pool, pos, x[qi], k)
    for i, fr in zip(qi, frac):
        if fr > 0.5:
            rows.append(dict(id=int(ids[i]), side=str(cand["side"].iloc[i]), source=f"homolog_knn{k}",
                             knn_frac=float(fr)))
    return pd.DataFrame(rows, columns=["id", "side", "source", "knn_frac"]), stats
