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

import re
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
    labels: np.ndarray | None = None   # optional per-neuron label (cell type); reported for changed neurons

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
        labels = None if self.labels is None else self.labels[keep]
        return NeuronSignRule(self.name, self.neurons[keep], self.sign[keep], info, labels)


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
        name=rule.name, n_neurons=int(len(rule.neurons)), n_edges=int(len(sel)),
        n_edges_changed=int(changed.sum()),
        n_neurons_changed=int(len(np.unique(pre[sel[changed]]))),
        syn_exc_to_inh=int(w[(old > 0) & (new < 0)].sum()), syn_inh_to_exc=int(w[(old < 0) & (new > 0)].sum()),
        syn_exc_to_zero=int(w[(old > 0) & (new == 0)].sum()), syn_inh_to_zero=int(w[(old < 0) & (new == 0)].sum()),
        syn_exc=int(w[old > 0].sum()), syn_inh=int(w[old < 0].sum()))
    if rule.labels is not None:
        changed_pre = np.unique(pre[sel[changed]])
        report["labels_changed"] = sorted(set(rule.labels[np.searchsorted(rule.neurons, changed_pre)].tolist()))
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
    conf = np.nan_to_num(cx.neurons["nt_conf"].to_numpy(float))   # missing prediction: 0 (FlyWire) or NaN (BANC)
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
    conf = np.nan_to_num(cx.neurons["nt_conf"].to_numpy(float))   # missing prediction: 0 (FlyWire) or NaN (BANC)
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


# ── literature: fast transmitter of the cell type from the literature (v1 --nt-literature) ───────────────────
FAST_NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}   # connectome.fast_nt_sign
LITERATURE_COLUMNS = ("cell_type", "n", "known_nt", "fast_nt", "sign_lit", "source", "confidence", "model_sign",
                      "n_change")
# Verified in the primary text in v1 (REPORT_SENSORY_A2 2.2); appended to the annotation source of that type.
LITERATURE_EXTRA_SOURCES = {
    "lLN1_bc": "Eckstein et al. 2024 (Cell 187:2574; discussion: broad unilateral 'ALl1 dorsal' LNs likely "
               "cholinergic, lLN1_bc mispredicted DA/5-HT); Huang et al. 2010 (Neuron 67:1021)"}
_LEGACY_CONFIDENCE = {"orta": "medium", "y\u00fcksek": "high"}   # labels stored in v1 data/nt_literature_783.csv


def fast_nt(known_nt) -> tuple:
    """Fast transmitters named in a ``known_nt`` string (";" or "," separated; "-negative" entries, peptides,
    NO and amines are ignored): a sorted tuple of FAST_NT_SIGN keys."""
    if not isinstance(known_nt, str):
        return ()
    toks = {t.strip() for t in re.split(r"[;,]", known_nt)}
    return tuple(sorted(t for t in toks if t in FAST_NT_SIGN))


def literature_confidence(source: str) -> str:
    """"medium" when the only evidence is scRNA-seq, lineage- / tract-based, or a morphological match to
    Shang et al. 2007 cholinergic eLNs; else "high" (reported only, not used by the rule)."""
    s = source.lower()
    if "shang et al., 2007" in s or "lineage" in s or "tract based" in s:
        return "medium"
    if not any(k in s for k in ("immuno", "tapin", "easi-fish", "rnai", "intersection", "mcfo", "marcm")):
        return "medium"
    return "high"


def _type_labels(cx: Connectome) -> pd.Series:
    n = cx.neurons
    return n["cell_type"].fillna(n["cell_type_alt"])


def literature_table(cx: Connectome, types=None, extra_sources=None) -> pd.DataFrame:
    """One row per cell type with a literature transmitter (LITERATURE_COLUMNS).

    Needs the extra columns ``known_nt`` and ``known_nt_source`` on the neurons table (FlyWire annotations,
    Schlegel et al. 2024). Sign rule (fixed before any run): exactly one fast transmitter in known_nt, shared
    by every neuron of the type -> its sign (ACh +1, GABA -1, Glu -1); none, several, or neurons that disagree
    -> 0 (no literature sign; untouched). ``types``: restrict to these cell types (v1: the types of the AL
    persistent state); placeholder groups without a type ("[class]") are skipped."""
    for c in ("known_nt", "known_nt_source"):
        if c not in cx.neurons.columns:
            raise ValueError(f"neurons table lacks the extra column {c!r}")
    extra_sources = LITERATURE_EXTRA_SOURCES if extra_sources is None else extra_sources
    typ = _type_labels(cx)
    _, published = _published_sign(cx)
    want = sorted(set(typ.dropna())) if types is None else sorted(set(types))
    rows = []
    for t in want:
        if t.startswith("["):
            continue
        m = (typ == t).to_numpy()
        kn = cx.neurons.loc[m, "known_nt"].dropna()
        if kn.empty:
            continue
        fs = {fast_nt(k) for k in kn}
        f = next(iter(fs)) if len(fs) == 1 else ()
        sign = FAST_NT_SIGN[f[0]] if len(f) == 1 and len(kn) == int(m.sum()) else 0
        src = "; ".join(sorted(set(cx.neurons.loc[m, "known_nt_source"].dropna())))
        if t in extra_sources:
            src += "; " + extra_sources[t]
        ms = published[m]
        rows.append(dict(cell_type=t, n=int(m.sum()), known_nt=" | ".join(sorted(set(kn))),
                         fast_nt="/".join(f) if f else "", sign_lit=sign,
                         source="Schlegel et al. 2024 known_nt: " + src, confidence=literature_confidence(src),
                         model_sign=",".join(str(x) for x in sorted(set(ms.tolist()))),
                         n_change=int(((ms != sign) & (sign != 0)).sum())))
    return pd.DataFrame(rows, columns=list(LITERATURE_COLUMNS))


def literature_rule(cx: Connectome, table: pd.DataFrame) -> NeuronSignRule:
    """Rule from a :func:`literature_table` (or v1's data/nt_literature_783.csv): neurons of the types with
    sign_lit != 0 get that sign on all outgoing synapses; other types are untouched."""
    lit = table.loc[table["sign_lit"] != 0].set_index("cell_type")["sign_lit"]
    typ = _type_labels(cx)
    sel = typ.isin(lit.index).to_numpy()
    idx = np.flatnonzero(sel)
    t = typ.to_numpy(object)[idx]
    return NeuronSignRule(
        "literature", idx, lit.reindex(t).to_numpy(np.int8), labels=t,
        info=dict(n_types=int(len(set(t.tolist()))),
                  rule="w = |w| * sign_lit on every outgoing synapse of the neurons of types with "
                       "sign_lit != 0 (ACh +1, GABA -1, Glu -1; Schlegel et al. 2024 known_nt)"))


# ── base sign of an unsigned graph: the Shiu rule, and the named BANC variants ───────────────────────────────
# Decision (prereg/01_banc_sign_rule.md): reference arm = the Shiu rule exactly as in v1: GABA and Glu -> -1,
# every other transmitter (ACh, DA, SER, OCT, HIST, TYR) and a missing prediction -> +1. Variants are flags:
#   hist_inhibitory  HIST -> -1 (histamine acts through HisCl chloride channels, e.g. photoreceptors)
#   nt_impute        missing prediction -> sign of the cell type's synapse-weighted majority (v1 N1)
INHIBITORY_NT = ("GABA", "GLUT")            # connectome.inhibitory_nt
BANC_PRED_SIGN = {**PRED_SIGN_SHIU, "HIST": 1, "TYR": 1}   # Shiu mapping extended; votes of the impute rule
SIGN_VARIANTS = ("hist_inhibitory", "nt_impute")


def shiu_neuron_sign(nt: pd.Series) -> np.ndarray:
    """+1 / -1 per neuron under the Shiu rule (inhibitory transmitters -1; everything else, incl. missing, +1)."""
    return np.where(nt.isin(INHIBITORY_NT).to_numpy(), -1, 1).astype(np.int8)


def nt_counts(cx: Connectome) -> dict:
    """Per transmitter label: neurons and outgoing synapses (label "none" = missing prediction)."""
    lab = cx.neurons["nt"].fillna("none").to_numpy(object)
    pre, _ = cx.edge_endpoints()
    out_syn = np.bincount(pre, weights=cx.edges["n_syn"].to_numpy(np.float64), minlength=cx.n_neurons)
    res = {}
    for t in sorted(set(lab.tolist())):
        m = lab == t
        res[t] = dict(neurons=int(m.sum()), out_synapses=int(out_syn[m].sum()))
    return res


def assign_signs(cx: Connectome, variants=()) -> Connectome:
    """Signed Connectome from an unsigned one: Shiu rule, then the requested named variants.

    The rule name and the counts per transmitter are recorded in ``meta`` (``sign_rule``, ``nt_counts``,
    ``sign_variants``) for the QA report. Variants are model variants, never the reference."""
    unknown = sorted(set(variants) - set(SIGN_VARIANTS))
    if unknown:
        raise ValueError(f"unknown sign variants {unknown}; choose from {SIGN_VARIANTS}")
    if cx.signed:
        raise ValueError(f"{cx.name} already has signs")
    pre, _ = cx.edge_endpoints()
    neuron_sign = shiu_neuron_sign(cx.neurons["nt"])
    edges = cx.edges.copy()
    edges["sign"] = neuron_sign[pre].astype(np.int64)
    meta = dict(cx.meta, sign_rule="shiu_v1", signed=True, nt_counts=nt_counts(cx), sign_variants=list(variants))
    out = Connectome(cx.name, cx.neurons, edges, meta)
    rules = []
    if "hist_inhibitory" in variants:
        idx = np.flatnonzero((cx.neurons["nt"] == "HIST").to_numpy())
        rules.append(NeuronSignRule("hist_inhibitory", idx, np.full(len(idx), -1, np.int8),
                                    dict(rule="MODEL VARIANT: HIST -> -1 (HisCl chloride channel)")))
    if "nt_impute" in variants:
        rules.append(impute_rule(out, BANC_PRED_SIGN))
    return apply_sign_rules(out, rules) if rules else out
