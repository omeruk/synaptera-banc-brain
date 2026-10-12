"""Sign variants of a connectome as one edge-table transform.

Pattern taken from synaptera@c94796d:scripts/make_nt_silent.py, make_nt_impute.py, make_nt_literature.py and
the three ``_silence_modulatory`` / ``_apply_nt_impute`` / ``_apply_nt_literature`` methods of flight/brain.py,
which edited the Brian2 synapse weights after the network was built. Here a variant is a
:class:`NeuronSignRule` (which neurons, which sign) applied to the ``sign`` column of the edge table *before*
any engine exists; the engine only ever sees ``Connectome.signed_weights()``.

Semantics (identical to v1):
  * a rule acts on the outgoing edges of its neurons only; synapse counts and the synapse list never change;
  * sign 0 silences (weight 0 on the fast path), +1 / -1 sets ``|w| * sign``;
  * an edge that is already silent (sign 0) is never revived by a +1 / -1 rule;
  * rules are applied in order; ``silent`` cannot be combined with ``impute`` / ``literature`` (v1: "excludes
    nt_silent"); ``impute`` + ``literature`` is variant N2.
Every non-reference variant is a *model variant*: the applied rule and its effect are recorded in
``Connectome.meta["sign_rules"]`` (CLAUDE.md rule 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from connectome.schema import Connectome

NT_MODULATORY = ("DA", "SER", "OCT")     # slow modulators (configs/assumptions.json: connectome.nt_modulatory)
SILENT_VARIANTS = ("broad", "narrow")
EXCLUSIVE_WITH_SILENT = ("impute", "literature")


@dataclass(frozen=True)
class NeuronSignRule:
    """Sign override for the outgoing edges of ``neurons`` (strictly increasing engine indices)."""

    name: str                     # "silent" | "impute" | "literature"
    neurons: np.ndarray
    sign: np.ndarray              # +1 / -1 / 0 per neuron (0 = silence)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.neurons) != len(self.sign):
            raise ValueError("neurons and sign differ in length")
        if len(self.neurons) > 1 and not (np.diff(self.neurons) > 0).all():
            raise ValueError("neurons must be strictly increasing")
        if not np.isin(self.sign, (-1, 0, 1)).all():
            raise ValueError("sign must be in {-1, 0, +1}")

    def excluding(self, protected) -> "NeuronSignRule":
        """The rule without the ``protected`` neurons (e.g. Poisson-driven input neurons, which v1 never silenced)."""
        keep = ~np.isin(self.neurons, np.asarray(protected, dtype=np.int64))
        info = dict(self.info, n_protected_kept=int((~keep).sum()))
        return NeuronSignRule(self.name, self.neurons[keep], self.sign[keep], info)


def apply_sign_rule(edge_sign: np.ndarray, pre: np.ndarray, n_syn: np.ndarray, n_neurons: int,
                    rule: NeuronSignRule) -> tuple[np.ndarray, dict]:
    """Return (new edge sign, effect report). ``pre`` holds engine indices of the presynaptic neurons."""
    member = np.zeros(n_neurons, bool)
    member[rule.neurons] = True
    sign_of = np.zeros(n_neurons, np.int8)
    sign_of[rule.neurons] = rule.sign
    sel = np.flatnonzero(member[pre])
    old = edge_sign[sel].astype(np.int64)
    new = sign_of[pre[sel]].astype(np.int64)
    if rule.name != "silent":
        new = np.where(old == 0, 0, new)          # a silent edge stays silent
    changed = old != new
    w = n_syn[sel].astype(np.int64)
    out = edge_sign.copy()
    out[sel] = new
    report = dict(
        rule=rule.name, n_neurons=int(len(rule.neurons)), n_edges=int(len(sel)),
        n_edges_changed=int(changed.sum()),
        n_neurons_changed=int(len(np.unique(pre[sel[changed]]))),
        syn_exc_to_inh=int(w[(old > 0) & (new < 0)].sum()), syn_inh_to_exc=int(w[(old < 0) & (new > 0)].sum()),
        syn_exc_to_zero=int(w[(old > 0) & (new == 0)].sum()), syn_inh_to_zero=int(w[(old < 0) & (new == 0)].sum()),
        syn_exc=int(w[old > 0].sum()), syn_inh=int(w[old < 0].sum()))
    report.update(rule.info)
    return out, report


def apply_sign_rules(cx: Connectome, rules) -> Connectome:
    """New Connectome whose edge signs follow ``rules`` in order; ``meta["sign_rules"]`` lists the reports."""
    if not cx.signed:
        raise ValueError(f"{cx.name}: an unsigned graph has no base sign to modify")
    rules = list(rules)
    names = [r.name for r in rules]
    if "silent" in names and any(n in EXCLUSIVE_WITH_SILENT for n in names):
        raise ValueError("silent excludes impute and literature (v1 rule)")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate rule in {names}")
    pre, _ = cx.edge_endpoints()
    n_syn = cx.edges["n_syn"].to_numpy(np.int64)
    sign = cx.edges["sign"].to_numpy(np.int64)
    reports = list(cx.meta.get("sign_rules", []))
    for r in rules:
        sign, rep = apply_sign_rule(sign, pre, n_syn, cx.n_neurons, r)
        reports.append(rep)
    edges = cx.edges.copy()
    edges["sign"] = sign
    meta = dict(cx.meta, sign_rules=reports)
    return Connectome(cx.name, cx.neurons, edges, meta)


# ── silent: modulatory / no-prediction neurons -> weight 0 on the fast path (v1 --nt-modulatory-silent) ─────────
def silent_rule(cx: Connectome, variant: str = "broad", modulatory=NT_MODULATORY) -> NeuronSignRule:
    """Neurons whose transmitter prediction is DA / SER / OCT, or missing (``broad``) / DA, SER, OCT only
    (``narrow``). Reads ``neurons.nt`` / ``nt_conf`` (Codex prediction). A missing prediction must have
    nt_conf == 0, as in v1. Protect Poisson-driven input neurons with :meth:`NeuronSignRule.excluding`."""
    if variant not in SILENT_VARIANTS:
        raise ValueError(variant)
    nt = cx.neurons["nt"]
    missing = nt.isna().to_numpy()
    conf = cx.neurons["nt_conf"].to_numpy(float)
    if missing.all():
        raise ValueError("neurons.nt is empty: load the transmitter predictions first")
    if (conf[missing] != 0).any():
        raise ValueError("a missing nt prediction with a non-zero score")
    sel = nt.isin(modulatory).to_numpy()
    if variant == "broad":
        sel = sel | missing
    idx = np.flatnonzero(sel)
    by_nt = {t: int((nt.to_numpy()[idx] == t).sum()) for t in modulatory}
    by_nt["none"] = int(missing[idx].sum())
    return NeuronSignRule("silent", idx, np.zeros(len(idx), np.int8),
                          dict(variant=variant, by_nt=by_nt, modulatory=list(modulatory)))


# ── impute: neurons without a prediction take the synapse-weighted majority sign of their cell type ──────────
# v1 MODEL VARIANT N1 (--nt-impute). Not the published model.
PRED_SIGN_SHIU = {"ACH": 1, "DA": 1, "SER": 1, "OCT": 1, "GABA": -1, "GLUT": -1}   # connectome.pred_sign_shiu
IMPUTE_COLUMNS = ("id", "cell_type", "published_sign", "imputed_sign", "n_peers", "peer_syn_pos", "peer_syn_neg")


def _published_sign(cx: Connectome):
    """(out-synapse count, sign) per neuron; sign 0 for neurons without outgoing synapses. The sign of a
    presynaptic neuron must be unique (true for every rule that is a function of the presynaptic neuron)."""
    pre, _ = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.int64)
    s = cx.edges["sign"].to_numpy(np.int64)
    n = cx.n_neurons
    out_syn = np.bincount(pre, weights=w, minlength=n).astype(np.int64)
    lo = np.full(n, 2, np.int64)
    hi = np.full(n, -2, np.int64)
    np.minimum.at(lo, pre, s)
    np.maximum.at(hi, pre, s)
    has = out_syn > 0
    if not (lo[has] == hi[has]).all():
        raise ValueError("the sign of a presynaptic neuron is not unique")
    return out_syn, np.where(has, lo, 0)


def impute_table(cx: Connectome, pred_sign: dict | None = None) -> pd.DataFrame:
    """One row per neuron without a transmitter prediction (IMPUTE_COLUMNS).

    Rule (v1, fixed before any run, whole brain): a predicted neuron votes with the sign of its PREDICTED
    transmitter under ``pred_sign`` (default PRED_SIGN_SHIU: ACh, DA, SER, OCT -> +1; GABA, GLUT -> -1),
    weighted by its number of output synapses; an empty neuron takes the sign of the weighted majority of the
    predicted neurons of the SAME cell_type. No cell_type, no predicted peer with outputs, a zero sum or no
    outputs of its own -> imputed_sign 0 (keeps its published sign).

    ``pred_sign`` must cover every transmitter label of the dataset; an unknown label raises, so that
    transmitters without an agreed sign (BANC: HIST, TYR) cannot be mapped by default."""
    pred_sign = PRED_SIGN_SHIU if pred_sign is None else pred_sign
    nt = cx.neurons["nt"]
    empty = nt.isna().to_numpy()
    conf = cx.neurons["nt_conf"].to_numpy(float)
    if (conf[empty] != 0).any():
        raise ValueError("a missing nt prediction with a non-zero score")
    unknown = sorted(set(nt.dropna().unique()) - set(pred_sign))
    if unknown:
        raise ValueError(f"transmitters without a sign in pred_sign: {unknown}")
    out_syn, published = _published_sign(cx)
    has = out_syn > 0
    pred = nt.map(pred_sign).fillna(0).to_numpy(np.int64)
    ctype = cx.neurons["cell_type"]
    df = pd.DataFrame({"idx": np.arange(cx.n_neurons), "cell_type": ctype.to_numpy(object), "empty": empty,
                       "published": published, "pos": out_syn * (pred > 0), "neg": out_syn * (pred < 0)})
    peers = df[(~df["empty"]) & df["cell_type"].notna() & has]
    agg = peers.groupby("cell_type").agg(n_peers=("idx", "size"), peer_syn_pos=("pos", "sum"),
                                         peer_syn_neg=("neg", "sum"))
    out = df[df["empty"]].join(agg, on="cell_type")
    for c in ("n_peers", "peer_syn_pos", "peer_syn_neg"):
        out[c] = out[c].fillna(0).astype(np.int64)
    imputed = np.sign((out["peer_syn_pos"] - out["peer_syn_neg"]).to_numpy()).astype(np.int64)
    imputed[out["published"].to_numpy() == 0] = 0     # no outputs of its own: nothing to change
    return pd.DataFrame({"id": cx.neurons["id"].to_numpy()[out["idx"].to_numpy()],
                         "cell_type": out["cell_type"].fillna("").to_numpy(),
                         "published_sign": out["published"].to_numpy(), "imputed_sign": imputed,
                         "n_peers": out["n_peers"].to_numpy(), "peer_syn_pos": out["peer_syn_pos"].to_numpy(),
                         "peer_syn_neg": out["peer_syn_neg"].to_numpy()}, columns=list(IMPUTE_COLUMNS))


def impute_rule_from_table(cx: Connectome, table: pd.DataFrame) -> NeuronSignRule:
    """Rule from an :func:`impute_table` result (or v1's data/nt_impute_783.csv with ``root_id`` for ``id``)."""
    t = table.rename(columns={"root_id": "id"})
    t = t[t["imputed_sign"] != 0]
    idx = cx.index.index_of(t["id"].to_numpy())
    o = np.argsort(idx)
    return NeuronSignRule(
        "impute", idx[o], t["imputed_sign"].to_numpy(np.int8)[o],
        dict(n_imputable=int(len(t)),
             rule="MODEL VARIANT N1: neurons without a transmitter prediction take the synapse-weighted "
                  "majority sign of the predicted neurons of the same cell_type; w = |w| * imputed_sign"))


def impute_rule(cx: Connectome, pred_sign: dict | None = None) -> NeuronSignRule:
    return impute_rule_from_table(cx, impute_table(cx, pred_sign))
