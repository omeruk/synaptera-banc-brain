"""Graph controls and the DEV subnet, as Connectome -> Connectome transforms.

Ported from synaptera@c94796d:flight/brain.py (dev_subnet_keep / write_dev_files, shuffled_connectome) with
the review fixes M3 and L6:
  * M3: the v1 null permuted the postsynaptic column over all edges, which duplicated (pre, post) pairs and
    changed the activity level (2.5x the DN spikes), confounding "wiring" with "activity". Here duplicates are
    merged, the shuffle can be restricted to a stratum (e.g. the post neuron's super_class), and the activity
    level is matched separately by ``engine.lif_brian2.match_weight_scale``.
  * L6: a control is applied to whatever graph it is given (so ``shuffle_targets(dev_subnet(...))`` works); the
    ``dev_subnet`` flag stays in ``meta`` and the engine warns about the combination instead of silently
    ignoring the control.
Every transform returns a new Connectome and records itself in ``meta``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from connectome.schema import Connectome
from connectome.sign_rules import NeuronSignRule

DEV_MIN_SYN = 10     # --dev-subnet partner threshold in v1 (configs/assumptions.json: connectome.dev_min_syn)


def _merge_duplicates(pre, post, n_syn, sign):
    """Collapse repeated (pre, post) pairs. Signed: net signed weight (sign 0 only if every copy is silent,
    pairs whose net weight is zero are dropped). Returns (pre, post, n_syn, sign|None, n_merged, n_dropped)."""
    df = pd.DataFrame({"pre": pre, "post": post, "n_syn": n_syn})
    if sign is None:
        g = df.groupby(["pre", "post"], sort=True)["n_syn"].sum().reset_index()
        return (g["pre"].to_numpy(), g["post"].to_numpy(), g["n_syn"].to_numpy(), None,
                int(len(df) - len(g)), 0)
    df["w"] = sign * n_syn
    df["active"] = sign != 0
    g = df.groupby(["pre", "post"], sort=True).agg(n=("n_syn", "sum"), w=("w", "sum"), active=("active", "max"))
    g = g.reset_index()
    new_sign = np.where(g["active"], np.sign(g["w"]), 0).astype(np.int64)
    new_n = np.where(g["active"], np.abs(g["w"]), g["n"]).astype(np.int64)
    keep = ~(g["active"].to_numpy() & (g["w"].to_numpy() == 0))
    return (g["pre"].to_numpy()[keep], g["post"].to_numpy()[keep], new_n[keep], new_sign[keep],
            int(len(df) - len(g)), int((~keep).sum()))


def shuffle_targets(cx: Connectome, seed: int, stratum: str | None = None) -> Connectome:
    """Null graph: every edge keeps its presynaptic neuron, synapse count and sign; its postsynaptic target is
    permuted among the edges (within the post neuron's ``stratum`` class when given, else over all edges).
    Out-degree, out-synapse sums and per-neuron in-degree (before merging duplicates) are preserved exactly;
    which neuron is contacted is destroyed. Duplicated pairs are merged (see _merge_duplicates)."""
    pre, post = cx.edge_endpoints()
    n_syn = cx.edges["n_syn"].to_numpy(np.int64)
    sign = cx.edges["sign"].to_numpy(np.int64) if cx.signed else None
    rng = np.random.default_rng(seed)
    if stratum is None:
        codes = np.zeros(cx.n_neurons, np.int64)
    else:
        codes = pd.factorize(cx.neurons[stratum].astype(object).where(cx.neurons[stratum].notna(), "<none>"))[0]
    edge_stratum = codes[post]
    new_post = post.copy()
    for g in np.unique(edge_stratum):
        m = np.flatnonzero(edge_stratum == g)
        new_post[m] = post[m][rng.permutation(len(m))]
    p, q, w, s, n_merged, n_dropped = _merge_duplicates(pre, new_post, n_syn, sign)
    ids = cx.index.ids
    edges = pd.DataFrame({"pre": ids[p], "post": ids[q], "n_syn": w})
    if s is not None:
        edges["sign"] = s
    ctl = dict(name="shuffle_targets", seed=int(seed), stratum=stratum, n_edges_before=int(len(pre)),
               n_edges_after=int(len(edges)), n_merged=n_merged, n_dropped_zero_net=n_dropped)
    meta = dict(cx.meta, control=ctl)
    return Connectome(cx.name, cx.neurons, edges, meta)


def ablation_rule(neurons) -> NeuronSignRule:
    """Ablation control: the outgoing synapses of ``neurons`` (engine indices) are silenced. Apply with
    ``sign_rules.apply_sign_rules`` (``cx`` keeps the synapse list)."""
    idx = np.unique(np.asarray(neurons, dtype=np.int64))
    return NeuronSignRule("ablation", idx, np.zeros(len(idx), np.int8), dict(n_ablated=int(len(idx))))


def dev_subnet(cx: Connectome, core, min_syn: int = DEV_MIN_SYN) -> Connectome:
    """DEVELOPMENT ONLY (never a result): the ``core`` neurons plus their 1-hop partners connected to a core
    neuron by >= ``min_syn`` synapses, with the edges among the kept neurons. ``meta["dev_subnet"]`` is set and
    ``meta["global_index"]`` maps local -> original engine index (spikes are stored with global indices)."""
    pre, post = cx.edge_endpoints()
    strong = cx.edges["n_syn"].to_numpy() >= min_syn
    is_core = np.zeros(cx.n_neurons, bool)
    is_core[np.asarray(core, dtype=np.int64)] = True
    keep = is_core.copy()
    keep[post[is_core[pre] & strong]] = True
    keep[pre[is_core[post] & strong]] = True
    glob = np.flatnonzero(keep)
    ids = cx.index.ids
    sub_edges = cx.edges[keep[pre] & keep[post]].reset_index(drop=True)
    meta = dict(cx.meta, dev_subnet=True, dev_min_syn=int(min_syn), global_index=glob,
                dev_note="DEV subnet: NOT a result")
    if "global_index" in cx.meta:                   # nested subnet: keep mapping to the original indices
        meta["global_index"] = np.asarray(cx.meta["global_index"])[glob]
    return Connectome(cx.name, cx.neurons.iloc[glob].reset_index(drop=True), sub_edges, meta)
