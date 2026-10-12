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
