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
