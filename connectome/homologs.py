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


def partner_features(ids, cx: Connectome) -> pd.DataFrame:
    """Raw (un-normalised) partner-type synapse counts, one row per id (row order = ``ids``), one column per
    feature ``"<in|out>|<partner type>|<i|c>"``.

    Partner type = cell_type, else cell_type_alt. A partner without a type contributes nothing. A partner
    counts as ipsilateral (i) only when both sides are known and equal (an unknown side is contralateral, c,
    as in Synaptera v1)."""
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
    out = mat.reindex(rows).fillna(0)
    out.index = ids
    return out


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def partner_profiles(ids, cx: Connectome) -> np.ndarray:
    """L2-normalised partner-type profiles (see :func:`partner_features`), one row per id."""
    return _unit_rows(partner_features(ids, cx).to_numpy(float))


def cross_dataset_cosine(ids_a, cx_a: Connectome, ids_b, cx_b: Connectome) -> pd.DataFrame:
    """Cosine similarity of partner-type profiles across two datasets, on the features (direction, partner
    type name, ipsi/contra) that exist in both. Only meaningful where the type vocabularies agree, so the
    share of each neuron's synapses that falls on shared features is returned too.

    Returns a DataFrame indexed by ids_a with one cosine column per ids_b entry (``cos_<id>``), plus
    ``shared_frac_a`` (per row) and a row ``shared_frac_b`` stored in ``attrs``."""
    fa, fb = partner_features(ids_a, cx_a), partner_features(ids_b, cx_b)
    shared = fa.columns.intersection(fb.columns)
    xa, xb = fa[shared].to_numpy(float), fb[shared].to_numpy(float)
    cos = _unit_rows(xa) @ _unit_rows(xb).T
    out = pd.DataFrame(cos, index=fa.index, columns=[f"cos_{i}" for i in fb.index])
    tot_a, tot_b = fa.to_numpy(float).sum(axis=1), fb.to_numpy(float).sum(axis=1)
    out["shared_frac_a"] = np.divide(xa.sum(axis=1), tot_a, out=np.zeros(len(tot_a)), where=tot_a > 0)
    out.attrs["shared_frac_b"] = np.divide(xb.sum(axis=1), tot_b, out=np.zeros(len(tot_b)),
                                           where=tot_b > 0).tolist()
    out.attrs["n_shared_features"] = int(len(shared))
    return out


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


def select_by_profile(cx: Connectome, *, positive_ids, negative_ids, query_ids, k: int = SUGAR_KNN) -> tuple[pd.DataFrame, dict]:
    """Label ``query_ids`` by the kNN vote of a labelled pool (positive / negative ids) on partner-type profiles.

    Returns (DataFrame id, side, cell_type, knn_frac, selected; loo) with loo = leave-one-out counts inside
    the pool: tp, n_positive, fp, n_negative, k."""
    pos = np.asarray(positive_ids, dtype=np.int64)
    neg = np.asarray(negative_ids, dtype=np.int64)
    q = np.asarray(query_ids, dtype=np.int64)
    x = partner_profiles(np.concatenate([pos, neg, q]), cx)
    pool, queries = x[:len(pos) + len(neg)], x[len(pos) + len(neg):]
    is_pos = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    loo = knn_positive_fraction(pool, is_pos, pool, k, exclude_self=True) > 0.5
    stats = dict(tp=int((loo & is_pos).sum()), n_positive=int(len(pos)), fp=int((loo & ~is_pos).sum()),
                 n_negative=int(len(neg)), k=k)
    frac = knn_positive_fraction(pool, is_pos, queries, k)
    nrn = cx.neurons.set_index("id")
    df = pd.DataFrame({"id": q, "side": nrn["side"].reindex(q).astype(str).to_numpy(),
                       "cell_type": nrn["cell_type"].reindex(q).astype(str).to_numpy(), "knn_frac": frac,
                       "selected": frac > 0.5})
    return df, stats


# FlyWire v783 annotation labels of the leg-GRN selection (SPEC_SENSORY_INPUTS 2.4 of v1)
LEG_GRN_SUBCLASS = "SA_VTV_pro_meso_meta"


def select_leg_sugar_grns(cx: Connectome, k: int = SUGAR_KNN, leg_subclass: str = LEG_GRN_SUBCLASS):
    """Sugar-like leg (tarsal) GRNs by connectivity profile.

    Ported from synaptera@c94796d:flight/groups.py:select_leg_sugar_grns. Labelled pool: labellar GRNs of
    cell_sub_class sugar/water (positive) and bitter (negative); query: ``sensory_ascending`` gustatory
    neurons of ``leg_subclass``. A leg GRN is selected when the majority of its k most similar pool neurons
    are sugar/water. Needs the extra columns cell_sub_class, cell_class and super_class (FlyWire annotations);
    a dataset with other labels passes the ids to :func:`select_by_profile` itself."""
    n = cx.neurons
    for c in ("cell_sub_class", "cell_class", "super_class"):
        if c not in n.columns:
            raise ValueError(f"neurons table lacks {c!r}")
    gust = n["cell_class"] == "gustatory"
    sugar = n.loc[gust & (n["super_class"] == "sensory") & (n["cell_sub_class"] == "sugar/water"), "id"]
    bitter = n.loc[gust & (n["super_class"] == "sensory") & (n["cell_sub_class"] == "bitter"), "id"]
    leg = n.loc[gust & (n["super_class"] == "sensory_ascending") & (n["cell_sub_class"] == leg_subclass), "id"]
    return select_by_profile(cx, positive_ids=sugar, negative_ids=bitter, query_ids=leg, k=k)
