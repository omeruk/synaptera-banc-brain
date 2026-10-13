"""Dataset QA computations of phase F1 (numbers for docs/results/f1_banc_qa.md; the script prints no hand-typed number).

Everything works on :class:`connectome.schema.Connectome` objects so the same functions run on FlyWire v783 and BANC:
counts and sign-rule effects, left / right symmetry per cell type with the input-synapse gate, sparseness,
critical-type presence and the antennal-lobe structural check of docs/02 section 1.6.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from connectome import qa
from connectome.schema import Connectome

# LIF arithmetic of docs/02 (Shiu parameters): depolarisation per (synapse x Hz) = tau_syn * w_syn
TAU_S_MS, W_SYN_MV, THRESHOLD_GAP_MV = 5.0, 0.275, 7.0
MEAN_DEPOL_MV_PER_SYN_HZ = TAU_S_MS * W_SYN_MV / 1000.0


def counts(cx: Connectome) -> dict:
    n = cx.neurons
    return dict(neurons=int(cx.n_neurons), left=int((n["side"] == "left").sum()), right=int((n["side"] == "right").sum()),
                typed=float(n["cell_type"].notna().mean()), pairs=int(len(cx.edges)),
                synapses=int(cx.edges["n_syn"].sum()), median_syn_per_pair=float(cx.edges["n_syn"].median()),
                super_classes=n["super_class"].value_counts(dropna=False).astype(int).to_dict())


def sign_effects(cx: Connectome) -> dict:
    """Neurons / outgoing synapses by sign (+1, -1, 0 = silent, no outgoing synapses) of the connectome's edge signs."""
    pre, _ = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.int64)
    s = cx.edges["sign"].to_numpy(np.int64)
    out = {}
    for name, v in (("excitatory", 1), ("inhibitory", -1), ("silent", 0)):
        m = s == v
        out[name] = dict(neurons=int(len(np.unique(pre[m]))), synapses=int(w[m].sum()))
    return out


def type_symmetry(cx: Connectome, threshold: float = qa.SYMMETRY_THRESHOLD, min_per_side: int = 1) -> pd.DataFrame:
    """One row per cell type with >= min_per_side neurons on each side: input synapses (total, excitatory) per side,
    log2(R/L) of both, and the gate (|log2| < threshold for both)."""
    total, exc = qa.input_synapses(cx)
    n = cx.neurons
    df = pd.DataFrame({"type": n["cell_type"].to_numpy(object), "side": n["side"].to_numpy(object),
                       "sc": n["super_class"].to_numpy(object), "tot": total, "exc": exc})
    df = df[df["type"].notna() & df["side"].isin(["left", "right"])]
    g = df.groupby(["type", "side"]).agg(n=("tot", "size"), tot=("tot", "sum"), exc=("exc", "sum"),
                                         sc=("sc", "first")).unstack("side")
    g = g.dropna(subset=[("n", "left"), ("n", "right")])
    out = pd.DataFrame({"type": g.index, "n_left": g[("n", "left")].astype(int).to_numpy(),
                        "n_right": g[("n", "right")].astype(int).to_numpy(), "super_class": g[("sc", "left")].to_numpy(),
                        "in_left": g[("tot", "left")].to_numpy(), "in_right": g[("tot", "right")].to_numpy(),
                        "exc_left": g[("exc", "left")].to_numpy(), "exc_right": g[("exc", "right")].to_numpy()})
    out = out[(out["n_left"] >= min_per_side) & (out["n_right"] >= min_per_side)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log2_total"] = np.log2(out["in_right"] / out["in_left"])
        out["log2_exc"] = np.log2(out["exc_right"] / out["exc_left"])
    out["pass"] = (out["log2_total"].abs() < threshold) & (out["log2_exc"].abs() < threshold)
    return out.reset_index(drop=True)


def symmetry_summary(t: pd.DataFrame) -> dict:
    def block(d):
        a = d["log2_total"].abs()
        return dict(n_types=int(len(d)), pass_fraction=float(d["pass"].mean()) if len(d) else float("nan"),
                    median_abs_log2_total=float(a.median()) if len(d) else float("nan"),
                    frac_abs_log2_gt_0p5=float((a > 0.5).mean()) if len(d) else float("nan"),
                    frac_abs_log2_gt_1=float((a > 1.0).mean()) if len(d) else float("nan"))
    out = dict(all=block(t), n_ge10_per_side=block(t[(t["n_left"] >= 10) & (t["n_right"] >= 10)]),
               single_pair=block(t[(t["n_left"] == 1) & (t["n_right"] == 1)]))
    out["by_super_class"] = {sc: block(g) for sc, g in t.groupby("super_class") if len(g) >= 5}
    return out


def side_totals(cx: Connectome) -> dict:
    """Neuron counts and input synapses per super class and side, R/L ratios (the optic lobe included)."""
    total, _ = qa.input_synapses(cx)
    n = cx.neurons
    out = {}
    for sc, g in n.groupby("super_class"):
        l, r = g["side"] == "left", g["side"] == "right"
        il, ir = float(total[g.index[l]].sum()), float(total[g.index[r]].sum())
        out[sc] = dict(n_left=int(l.sum()), n_right=int(r.sum()), in_left=il, in_right=ir,
                       ratio_in_right_over_left=(ir / il) if il > 0 else float("nan"))
    return out


def sparseness(cx: Connectome, group, min_syn: int = 0) -> dict:
    """Mean incoming edges / synapses per neuron for the neurons of each group label (``group``: array over the
    neurons table). ``min_syn``: keep only pairs with >= min_syn synapses (the Shiu graph keeps all pairs, min 1;
    the BANC export has min 3: see :func:`pair_threshold_profile`)."""
    _, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.int64)
    keep = w >= min_syn
    in_e = np.bincount(post[keep], minlength=cx.n_neurons)
    in_s = np.bincount(post[keep], weights=w[keep], minlength=cx.n_neurons)
    g = pd.Series(np.asarray(group, dtype=object))
    out = {}
    for lab in g.dropna().unique():
        m = (g == lab).to_numpy()
        out[str(lab)] = dict(n=int(m.sum()), in_edges_mean=float(in_e[m].mean()), in_syn_mean=float(in_s[m].mean()),
                             syn_per_edge=float(in_s[m].sum() / max(in_e[m].sum(), 1)))
    out["__all__"] = dict(n=cx.n_neurons, in_edges_mean=float(in_e.mean()), in_syn_mean=float(in_s.mean()),
                          syn_per_edge=float(in_s.sum() / max(in_e.sum(), 1)))
    return out


def pair_threshold_profile(cx: Connectome, thresholds=(1, 3, 5)) -> dict:
    """Synapses-per-pair profile of the graph as it is: minimum, fraction of pairs below 5, and the pairs / synapses
    that remain at each threshold (pair level, neuropils summed)."""
    w = cx.edges["n_syn"].to_numpy(np.int64)
    out = dict(pairs=int(len(w)), synapses=int(w.sum()), min_syn_per_pair=int(w.min()),
               frac_pairs_below_5=float((w < 5).mean()))
    for t in thresholds:
        k = w >= t
        out[f"ge{t}"] = dict(pairs=int(k.sum()), synapses=int(w[k].sum()))
    return out


def count_types(cx: Connectome, patterns: dict, use_alt: bool = True, where=None) -> dict:
    """{label: dict(n, left, right, types)} for regex patterns on cell_type (and cell_type_alt)."""
    n = cx.neurons
    ct = n["cell_type"].astype(str)
    alt = n["cell_type_alt"].astype(str) if use_alt else ct
    out = {}
    for label, pat in patterns.items():
        m = ct.str.contains(pat, regex=True) | (alt.str.contains(pat, regex=True) if use_alt else False)
        if where is not None:
            m &= where
        sub = n[m]
        out[label] = dict(n=int(m.sum()), left=int((sub["side"] == "left").sum()),
                          right=int((sub["side"] == "right").sum()),
                          types=sub["cell_type"].astype(str).value_counts().head(12).to_dict())
    return out


def al_structure(cx: Connectome, alln_mask, alpn_mask) -> dict:
    """The docs/02 section 1.6 check on the antennal-lobe LN + PN subnetwork under the connectome's edge signs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigs
    alln, alpn = np.flatnonzero(alln_mask), np.flatnonzero(alpn_mask)
    inner = np.concatenate([alln, alpn])
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.float64)
    sg = cx.edges["sign"].to_numpy(np.int64)
    pos = np.full(cx.n_neurons, -1)
    pos[inner] = np.arange(len(inner))
    m = (pos[pre] >= 0) & (pos[post] >= 0)
    n_in = len(inner)
    exc_s, inh_s = float(w[m & (sg > 0)].sum()), float(w[m & (sg < 0)].sum())
    A_signed = coo_matrix((w[m] * sg[m], (pos[post[m]], pos[pre[m]])), shape=(n_in, n_in)).tocsr()
    A_exc = coo_matrix((np.where(sg[m] > 0, w[m], 0.0), (pos[post[m]], pos[pre[m]])), shape=(n_in, n_in)).tocsr()
    def lead(A):
        if A.shape[0] < 4:                         # ARPACK needs k < N - 1
            return float(np.max(np.real(np.linalg.eigvals(A.toarray()))))
        return float(np.real(eigs(A.astype(float), k=1, which="LR", return_eigenvectors=False)[0]))
    ev_s, ev_e = lead(A_signed), lead(A_exc)
    syn_hz_thr = THRESHOLD_GAP_MV / MEAN_DEPOL_MV_PER_SYN_HZ
    rec_e = np.bincount(pos[post[m & (sg > 0)]], weights=w[m & (sg > 0)], minlength=n_in)
    rec_i = np.bincount(pos[post[m & (sg < 0)]], weights=w[m & (sg < 0)], minlength=n_in)
    pre_sign = np.zeros(cx.n_neurons, np.int64)
    pre_sign[pre] = sg
    return dict(
        n_alln=int(len(alln)), n_alpn=int(len(alpn)),
        alln_sign=dict(excitatory=int((pre_sign[alln] > 0).sum()), inhibitory=int((pre_sign[alln] < 0).sum())),
        alpn_sign=dict(excitatory=int((pre_sign[alpn] > 0).sum()), inhibitory=int((pre_sign[alpn] < 0).sum())),
        internal_syn=dict(excitatory=exc_s, inhibitory=inh_s, excitatory_fraction=exc_s / max(exc_s + inh_s, 1.0)),
        neurons_with_exc_rec_ge_threshold_at_100Hz=int((rec_e * 100 >= syn_hz_thr).sum()),
        neurons_with_net_rec_ge_threshold_at_100Hz=int(((rec_e - rec_i) * 100 >= syn_hz_thr).sum()),
        of=int(n_in), threshold_syn_hz=float(syn_hz_thr),
        leading_eig_exc_syn=ev_e, leading_eig_signed_syn=ev_s,
        self_sustaining_mean_rate_hz=float(THRESHOLD_GAP_MV / (ev_s * MEAN_DEPOL_MV_PER_SYN_HZ)) if ev_s > 0 else None)
