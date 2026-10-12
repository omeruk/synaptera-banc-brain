import numpy as np
import pandas as pd
import pytest

from connectome import sign_rules as SR
from connectome.schema import Connectome
from conftest import ROOT
from toy import edges_table, neurons_table


def _cx():
    """6 neurons: 0 ACH, 1 DA, 2 OCT, 3 no prediction, 4 GABA, 5 SER. Every neuron i -> (i+1) % 6, 10 syn;
    neuron 0 also -> 2 (4 syn)."""
    nt = ["ACH", "DA", "OCT", np.nan, "GABA", "SER"]
    n = neurons_table([dict(id=100 + i, cell_type=f"T{i % 3}", side="left") for i in range(6)])
    n["nt"] = nt
    n["nt_conf"] = [0.9, 0.8, 0.7, 0.0, 0.9, 0.6]
    tr = [(100 + i, 100 + (i + 1) % 6, 10) for i in range(6)] + [(100, 102, 4)]
    e = edges_table(tr)
    e["sign"] = [1, 1, 1, 1, -1, 1, 1]          # neuron 4 (GABA) is inhibitory
    return Connectome("toy", n, e)


def test_silent_variants_select_expected_neurons():
    cx = _cx()
    assert SR.silent_rule(cx, "narrow").neurons.tolist() == [1, 2, 5]
    broad = SR.silent_rule(cx, "broad")
    assert broad.neurons.tolist() == [1, 2, 3, 5]
    assert broad.info["by_nt"] == {"DA": 1, "SER": 1, "OCT": 1, "none": 1}
    with pytest.raises(ValueError):
        SR.silent_rule(cx, "all")


def test_silent_zeroes_outgoing_edges_only_and_keeps_synapse_list():
    cx = _cx()
    out = SR.apply_sign_rules(cx, [SR.silent_rule(cx, "narrow")])
    pre, _ = cx.edge_endpoints()
    s = out.edges["sign"].to_numpy()
    assert (s[np.isin(pre, [1, 2, 5])] == 0).all()
    np.testing.assert_array_equal(s[~np.isin(pre, [1, 2, 5])], cx.edges["sign"].to_numpy()[~np.isin(pre, [1, 2, 5])])
    pd.testing.assert_frame_equal(out.edges[["pre", "post", "n_syn"]], cx.edges[["pre", "post", "n_syn"]])
    rep = out.meta["sign_rules"][0]
    assert rep["n_edges"] == 3 and rep["n_edges_changed"] == 3 and rep["syn_exc_to_zero"] == 30
    assert rep["variant"] == "narrow" and rep["syn_exc"] == 30


def test_original_connectome_is_not_modified():
    cx = _cx()
    before = cx.edges["sign"].copy()
    SR.apply_sign_rules(cx, [SR.silent_rule(cx)])
    pd.testing.assert_series_equal(cx.edges["sign"], before)
    assert "sign_rules" not in cx.meta


def test_protected_neurons_are_kept():
    cx = _cx()
    r = SR.silent_rule(cx, "broad").excluding([3, 5])
    assert r.neurons.tolist() == [1, 2] and r.info["n_protected_kept"] == 2


def test_missing_prediction_needs_zero_score_and_predictions_must_be_loaded():
    cx = _cx()
    bad = cx.neurons.copy()
    bad.loc[3, "nt_conf"] = 0.5
    with pytest.raises(ValueError, match="non-zero score"):
        SR.silent_rule(Connectome("b", bad, cx.edges))
    empty = cx.neurons.assign(nt=np.nan, nt_conf=np.nan)
    with pytest.raises(ValueError, match="empty"):
        SR.silent_rule(Connectome("e", empty, cx.edges))


def test_unsigned_graph_refused():
    cx = _cx()
    u = Connectome("u", cx.neurons, cx.edges.drop(columns="sign"))
    with pytest.raises(ValueError, match="unsigned"):
        SR.apply_sign_rules(u, [])


def test_rule_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        SR.NeuronSignRule("silent", np.array([2, 1]), np.zeros(2, np.int8))
    with pytest.raises(ValueError, match="sign"):
        SR.NeuronSignRule("impute", np.array([1]), np.array([2]))


def test_v1_silent_table_counts_via_real_neurons():
    """data/nt_modulatory_silent_783.csv of v1: 20,719 broad / 1,677 narrow (with the Codex nt_type), rebuilt
    from that table into the FlyWire neuron table (the CSV holds exactly the selected neurons)."""
    from connectome.adapters import flywire783 as fw
    t = pd.read_csv(ROOT / "data" / "nt_modulatory_silent_783.csv")
    n = fw.load_neurons()
    nt = t.set_index("root_id")
    n["nt"] = n["id"].map(nt["nt_type"].replace("none", np.nan)).to_numpy(object)
    # neurons outside the table carry an (unknown, non-modulatory) prediction: mark them "ACH"
    n.loc[~n["id"].isin(nt.index), "nt"] = "ACH"
    n["nt_conf"] = np.where(n["nt"].isna(), 0.0, 1.0)
    cx = Connectome("fw", n, pd.DataFrame({"pre": [], "post": [], "n_syn": [], "sign": []}, dtype=np.int64))
    assert len(SR.silent_rule(cx, "broad").neurons) == 20_719
    assert len(SR.silent_rule(cx, "narrow").neurons) == 1_677


# ── impute ───────────────────────────────────────────────────────────────────
def _impute_cx():
    """Cell type 'T': predicted ACH (out 30), GABA (out 10), empty E1 (out 5, published +1), empty E2 (no outputs).
    Cell type 'U': predicted GABA only (out 7) -> empty E3 (out 3) becomes -1. Empty E4: no cell type.
    Cell type 'V': two predicted neurons tie (ACH 10 / GABA 10) -> empty E5 imputed 0."""
    rows = [("T", "ACH"), ("T", "GABA"), ("T", np.nan), ("T", np.nan), ("U", "GABA"), ("U", np.nan),
            (np.nan, np.nan), ("V", "ACH"), ("V", "GABA"), ("V", np.nan), ("W", "ACH")]
    n = neurons_table([dict(id=i, cell_type=ct, side="left") for i, (ct, _) in enumerate(rows)])
    n["nt"] = [nt for _, nt in rows]
    n["nt_conf"] = [0.0 if isinstance(nt, float) else 0.9 for _, nt in rows]
    out = {0: 30, 1: 10, 2: 5, 4: 7, 5: 3, 6: 2, 7: 10, 8: 10, 9: 4, 10: 1}
    tr, sg = [], []
    for i, w in out.items():
        tr.append((i, 10, w))                       # everyone projects to the (target-only) neuron 10
        sg.append(-1 if rows[i][1] == "GABA" else 1)
    e = edges_table(tr)
    e["sign"] = sg
    return Connectome("imp", n, e)


def test_impute_table_votes_and_zero_cases():
    cx = _impute_cx()
    t = SR.impute_table(cx).set_index("id")
    assert list(SR.impute_table(cx).columns) == list(SR.IMPUTE_COLUMNS)
    assert set(t.index) == {2, 3, 5, 6, 9}
    assert t.loc[2, ["imputed_sign", "n_peers", "peer_syn_pos", "peer_syn_neg"]].tolist() == [1, 2, 30, 10]
    assert t.loc[3, "imputed_sign"] == 0 and t.loc[3, "published_sign"] == 0   # no outputs -> unchanged
    assert t.loc[5, "imputed_sign"] == -1
    assert t.loc[6, "imputed_sign"] == 0 and t.loc[6, "n_peers"] == 0           # no cell type
    assert t.loc[9, "imputed_sign"] == 0 and t.loc[9, "peer_syn_pos"] == 10     # tie


def test_impute_uses_predicted_transmitter_not_published_sign():
    cx = _impute_cx()
    e = cx.edges.copy()
    e.loc[e["pre"] == 0, "sign"] = -1            # published sign of the ACH neuron flipped (the 473 v1 cases)
    t = SR.impute_table(Connectome("imp2", cx.neurons, e)).set_index("id")
    assert t.loc[2, "peer_syn_pos"] == 30        # still votes as ACH (+)


def test_impute_apply_changes_only_empty_neurons_and_keeps_zero_edges():
    cx = _impute_cx()
    out = SR.apply_sign_rules(cx, [SR.impute_rule(cx)])
    changed = np.flatnonzero(out.edges["sign"].to_numpy() != cx.edges["sign"].to_numpy())
    pre, _ = cx.edge_endpoints()
    assert set(pre[changed]) == {5}              # E3: +1 published -> -1 imputed; E1 already +1 = +1
    rep = out.meta["sign_rules"][0]
    assert rep["syn_exc_to_inh"] == 3 and rep["n_neurons_changed"] == 1 and rep["n_imputable"] == 2
    # an edge that is silent stays silent
    z = cx.edges.copy()
    z.loc[z["pre"] == 5, "sign"] = 0
    out2 = SR.apply_sign_rules(Connectome("z", cx.neurons, z), [SR.NeuronSignRule("impute", np.array([5]), np.array([-1], np.int8))])
    assert (out2.edges.loc[out2.edges["pre"] == 5, "sign"] == 0).all()


def test_impute_unknown_transmitter_must_be_mapped():
    cx = _impute_cx()
    n = cx.neurons.copy()
    n.loc[0, "nt"] = "HIST"
    with pytest.raises(ValueError, match="HIST"):
        SR.impute_table(Connectome("h", n, cx.edges))
    t = SR.impute_table(Connectome("h", n, cx.edges), pred_sign={**SR.PRED_SIGN_SHIU, "HIST": -1})
    assert t.set_index("id").loc[2, "peer_syn_neg"] == 40


def test_silent_excludes_impute_but_impute_literature_compose():
    cx = _impute_cx()
    with pytest.raises(ValueError, match="excludes"):
        SR.apply_sign_rules(cx, [SR.NeuronSignRule("silent", np.array([0]), np.zeros(1, np.int8)),
                                 SR.NeuronSignRule("impute", np.array([1]), np.array([1], np.int8))])
    with pytest.raises(ValueError, match="duplicate"):
        r = SR.NeuronSignRule("impute", np.array([1]), np.array([1], np.int8))
        SR.apply_sign_rules(cx, [r, r])


def test_v1_impute_table_published_signs_match_shiu_edges():
    """Real FlyWire: every neuron of data/nt_impute_783.csv has the published sign v1 recorded (the Shiu
    ``Excitatory`` sign of its outgoing edges) and the rule built from the table selects the imputable ones."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    t = pd.read_csv(ROOT / "data" / "nt_impute_783.csv")
    _, published = SR._published_sign(cx)
    idx = cx.index.index_of(t["root_id"].to_numpy())
    np.testing.assert_array_equal(published[idx], t["published_sign"].to_numpy())
    rule = SR.impute_rule_from_table(cx, t)
    assert len(rule.neurons) == int((t["imputed_sign"] != 0).sum())
    out = SR.apply_sign_rules(cx, [rule])
    rep = out.meta["sign_rules"][0]
    ch = (t["imputed_sign"] != 0) & (t["imputed_sign"] != t["published_sign"])
    assert rep["n_neurons_changed"] == int(ch.sum())


# ── literature ───────────────────────────────────────────────────────────────
def _lit_cx():
    """Types: A (2 neurons, both 'acetylcholine'), B (2 neurons, 'gaba' and 'glutamate; gaba' -> disagree),
    C ('gaba-negative; acetylcholine' -> ACh +1, 1 neuron), D (dopamine only), E (no known_nt), '[x]' placeholder."""
    spec = [("A", "acetylcholine"), ("A", "acetylcholine"), ("B", "gaba"), ("B", "glutamate; gaba"),
            ("C", "gaba-negative; acetylcholine"), ("D", "dopamine"), ("E", np.nan), ("[x]", "gaba")]
    n = neurons_table([dict(id=i, cell_type=t, side="left") for i, (t, _) in enumerate(spec)])
    n["known_nt"] = [k for _, k in spec]
    n["known_nt_source"] = ["Foo et al. (immuno)", "Foo et al. (immuno)", "Bar (scRNA-seq)", "Bar (scRNA-seq)",
                            "Baz (immuno, lineage based)", "x", np.nan, "y"]
    tr = [(i, 7, 5) for i in range(7)]
    e = edges_table(tr)
    e["sign"] = [1, -1, 1, 1, -1, 1, 1]            # model: A has +1 and -1; C is inhibitory (wrong)
    return Connectome("lit", n, e)


def test_fast_nt_parsing():
    assert SR.fast_nt("acetylcholine; gaba-negative") == ("acetylcholine",)
    assert SR.fast_nt("gaba, glutamate") == ("gaba", "glutamate")
    assert SR.fast_nt("dopamine; nitric oxide") == () and SR.fast_nt(np.nan) == ()


def test_literature_table_rule():
    cx = _lit_cx()
    t = SR.literature_table(cx).set_index("cell_type")
    assert list(SR.literature_table(cx).columns) == list(SR.LITERATURE_COLUMNS)
    assert "[x]" not in t.index and "E" not in t.index
    assert t.loc["A", "sign_lit"] == 1 and t.loc["A", "n_change"] == 1       # the -1 neuron changes
    assert t.loc["B", "sign_lit"] == 0 and t.loc["B", "n_change"] == 0       # neurons disagree -> untouched
    assert t.loc["C", "sign_lit"] == 1 and t.loc["C", "n_change"] == 1
    assert t.loc["D", "sign_lit"] == 0 and t.loc["D", "fast_nt"] == ""
    assert t.loc["A", "confidence"] == "high" and t.loc["B", "confidence"] == "medium"
    assert t.loc["C", "confidence"] == "medium"                                # lineage based
    assert SR.literature_table(cx, types=["A"]).shape[0] == 1


def test_literature_apply_reports_changed_types():
    cx = _lit_cx()
    out = SR.apply_sign_rules(cx, [SR.literature_rule(cx, SR.literature_table(cx))])
    s = out.edges.set_index("pre")["sign"]
    assert s[0] == 1 and s[1] == 1 and s[4] == 1                               # A and C forced to +1
    assert s[2] == 1 and s[3] == 1                                             # B untouched
    rep = out.meta["sign_rules"][0]
    assert rep["labels_changed"] == ["A", "C"] and rep["n_neurons_changed"] == 2
    assert rep["syn_inh_to_exc"] == 10 and rep["n_types"] == 2


def test_impute_then_literature_is_variant_n2_and_literature_wins():
    cx = _lit_cx()
    imp = SR.NeuronSignRule("impute", np.array([0, 1]), np.array([-1, -1], np.int8))
    lit = SR.literature_rule(cx, SR.literature_table(cx))
    out = SR.apply_sign_rules(cx, [imp, lit])
    s = out.edges.set_index("pre")["sign"]
    assert s[0] == 1 and s[1] == 1             # literature applied last
    assert [r["name"] for r in out.meta["sign_rules"]] == ["impute", "literature"]


def test_v1_literature_table_reproduced_from_annotations():
    """FlyWire v783: the sign rule applied to the annotation known_nt reproduces v1's sign_lit / fast_nt / n for
    every type of data/nt_literature_783.csv."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    ref = pd.read_csv(ROOT / "data" / "nt_literature_783.csv")
    mine = SR.literature_table(cx, types=ref["cell_type"]).set_index("cell_type")
    r = ref.set_index("cell_type")
    assert set(mine.index) == set(r.index)
    assert (mine["sign_lit"] == r["sign_lit"]).all()
    assert (mine["fast_nt"] == r["fast_nt"].fillna("")).all()
    assert (mine["n"] == r["n"]).all()
    legacy = r["confidence"].map(SR._LEGACY_CONFIDENCE)
    assert (mine["confidence"] == legacy).mean() > 0.97          # confidence is report-only
    rule = SR.literature_rule(cx, ref)
    assert len(rule.neurons) == int(r.loc[r["sign_lit"] != 0, "n"].sum())
