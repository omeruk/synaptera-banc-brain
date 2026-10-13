import numpy as np
import pandas as pd
import pytest

from connectome import qa_report as Q
from connectome.schema import Connectome
from toy import edges_table, neurons_table


def _cx():
    rows = [dict(id=i, cell_type=t, side=s, super_class="central") for i, (t, s) in enumerate(
        [("A", "left"), ("A", "right"), ("B", "left"), ("B", "right"), ("C", "left")])]
    n = neurons_table(rows)
    # inputs: A_R gets 2x the synapses of A_L (fails), B balanced (passes); C has no partner
    e = edges_table([(4, 0, 10), (4, 1, 20), (4, 2, 15), (4, 3, 15)])
    e["sign"] = [1, 1, 1, -1]
    return Connectome("t", n, e)


def test_counts_and_sign_effects():
    cx = _cx()
    c = Q.counts(cx)
    assert c["neurons"] == 5 and c["pairs"] == 4 and c["synapses"] == 60 and c["left"] == 3
    s = Q.sign_effects(cx)
    assert s["excitatory"]["synapses"] == 45 and s["inhibitory"]["synapses"] == 15


def test_type_symmetry_gate():
    t = Q.type_symmetry(_cx()).set_index("type")
    assert set(t.index) == {"A", "B"}
    assert t.loc["A", "log2_total"] == pytest.approx(1.0) and not t.loc["A", "pass"]
    assert t.loc["B", "log2_total"] == pytest.approx(0.0)
    # B: excitatory input 15 on the left, 0 on the right -> the excitatory gate fails even though totals match
    assert np.isinf(t.loc["B", "log2_exc"]) or t.loc["B", "log2_exc"] < -0.5 or not t.loc["B", "pass"]
    s = Q.symmetry_summary(t.reset_index())
    assert s["all"]["n_types"] == 2 and s["all"]["pass_fraction"] == 0.0


def test_sparseness_threshold_and_groups():
    cx = _cx()
    g = np.array(["x", "x", "y", "y", "y"], dtype=object)
    sp = Q.sparseness(cx, g)
    assert sp["x"]["in_edges_mean"] == 1.0 and sp["__all__"]["in_syn_mean"] == pytest.approx(12.0)
    sp5 = Q.sparseness(cx, g, min_syn=16)
    assert sp5["__all__"]["in_edges_mean"] == pytest.approx(0.2)


def test_al_structure_toy_eigenvalue():
    # two-neuron loop: LN0 -> PN1 (3 syn, +), PN1 -> LN0 (3 syn, +): leading eigenvalue 3
    n = neurons_table([dict(id=0, cell_type="l", side="left"), dict(id=1, cell_type="p", side="left"),
                       dict(id=2, cell_type="o", side="left")])
    e = edges_table([(0, 1, 3), (1, 0, 3), (2, 0, 5)])
    cx = Connectome("al", n, e)
    r = Q.al_structure(cx, np.array([True, False, False]), np.array([False, True, False]))
    assert r["leading_eig_exc_syn"] == pytest.approx(3.0) and r["internal_syn"]["excitatory_fraction"] == 1.0
    assert r["n_alln"] == 1 and r["n_alpn"] == 1
    assert r["threshold_syn_hz"] == pytest.approx(7.0 / (5 * 0.275 / 1000))


def test_count_types_patterns():
    cx = _cx()
    r = Q.count_types(cx, {"AB": r"^(A|B)$", "C": r"^C$"}, use_alt=False)
    assert r["AB"]["n"] == 4 and r["AB"]["left"] == 2 and r["C"]["n"] == 1
