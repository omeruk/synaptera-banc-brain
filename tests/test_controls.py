import numpy as np
import pandas as pd
import pytest

from connectome import controls as C
from connectome import sign_rules as SR
from connectome.schema import Connectome
from toy import edges_table, neurons_table


def _random_cx(n=60, m=400, seed=0, n_class=3):
    rng = np.random.default_rng(seed)
    nrn = neurons_table([dict(id=1000 + i, cell_type=f"T{i % 7}", side="left") for i in range(n)])
    nrn["super_class"] = [f"C{i % n_class}" for i in range(n)]
    pre = rng.integers(0, n, m)
    post = rng.integers(0, n, m)
    pair = {(a, b) for a, b in zip(pre, post)}
    pre, post = np.array([p for p, _ in pair]), np.array([q for _, q in pair])
    sign_of = rng.choice([-1, 1], n)
    e = pd.DataFrame({"pre": 1000 + pre, "post": 1000 + post, "n_syn": rng.integers(1, 20, len(pre)),
                      "sign": sign_of[pre]})
    return Connectome("rnd", nrn, e)


def test_shuffle_preserves_out_structure_and_signs():
    cx = _random_cx()
    sh = C.shuffle_targets(cx, seed=1)
    ctl = sh.meta["control"]
    assert ctl["name"] == "shuffle_targets" and ctl["n_edges_after"] == len(sh.edges)
    # total synapses and per-neuron out-synapse sums are preserved (signs are a function of the presynaptic neuron)
    a = cx.edges.groupby("pre")["n_syn"].sum()
    b = sh.edges.groupby("pre")["n_syn"].sum()
    pd.testing.assert_series_equal(a, b, check_names=False)
    assert cx.edges["n_syn"].sum() == sh.edges["n_syn"].sum()
    assert not sh.edges.duplicated(["pre", "post"]).any()
    assert sh.edges.groupby("pre")["sign"].nunique().max() == 1
    assert "sign_rules" not in sh.meta and cx.meta == {}


def test_shuffle_changes_wiring_and_is_seeded():
    cx = _random_cx()
    s1, s1b, s2 = C.shuffle_targets(cx, 1), C.shuffle_targets(cx, 1), C.shuffle_targets(cx, 2)
    pd.testing.assert_frame_equal(s1.edges, s1b.edges)
    assert not s1.edges.equals(s2.edges)
    orig = set(zip(cx.edges["pre"], cx.edges["post"]))
    kept = set(zip(s1.edges["pre"], s1.edges["post"])) & orig
    assert len(kept) < 0.2 * len(orig)


def test_stratified_shuffle_keeps_post_class_marginals():
    cx = _random_cx(n=90, m=900)
    cls = cx.neurons.set_index("id")["super_class"]
    sh = C.shuffle_targets(cx, 3, stratum="super_class")
    before = cx.edges["post"].map(cls).value_counts().sort_index()
    after_w = sh.edges["post"].map(cls)
    # synapse mass per post class is preserved (duplicates merge but never cross classes)
    mass_b = cx.edges.assign(c=cx.edges["post"].map(cls)).groupby("c")["n_syn"].sum()
    mass_a = sh.edges.assign(c=after_w).groupby("c")["n_syn"].sum()
    pd.testing.assert_series_equal(mass_b, mass_a, check_names=False)
    assert before.sum() >= len(sh.edges)


def test_merge_duplicates_signed_and_silent():
    p, q, w, s, merged, dropped = C._merge_duplicates(
        np.array([0, 0, 1, 1, 2, 2]), np.array([5, 5, 5, 5, 6, 6]), np.array([3, 4, 2, 2, 7, 1]),
        np.array([1, 1, 1, -1, 0, 0]))
    out = {(a, b): (n, sg) for a, b, n, sg in zip(p, q, w, s)}
    assert out == {(0, 5): (7, 1), (2, 6): (8, 0)}          # (1, 5) nets to zero -> dropped
    assert merged == 3 and dropped == 1


def test_ablation_and_dev_subnet():
    cx = _random_cx(n=40, m=300)
    out = SR.apply_sign_rules(cx, [C.ablation_rule([3, 3, 7])])
    pre, _ = cx.edge_endpoints()
    assert (out.edges.loc[np.isin(pre, [3, 7]), "sign"] == 0).all()
    assert out.meta["sign_rules"][0]["name"] == "ablation" and out.meta["sign_rules"][0]["n_ablated"] == 2

    sub = C.dev_subnet(cx, core=[0, 1], min_syn=10)
    g = sub.meta["global_index"]
    assert sub.meta["dev_subnet"] and np.isin([0, 1], g).all() and sub.n_neurons == len(g) < cx.n_neurons
    assert sub.neurons["id"].tolist() == cx.neurons["id"].to_numpy()[g].tolist()
    # kept neurons are core or a >=10-synapse partner of a core neuron
    pre_i, post_i = cx.edge_endpoints()
    strong = cx.edges["n_syn"].to_numpy() >= 10
    partners = set(post_i[np.isin(pre_i, [0, 1]) & strong]) | set(pre_i[np.isin(post_i, [0, 1]) & strong])
    assert set(g) == partners | {0, 1}
    # sub-edges are exactly the original edges among kept neurons
    both = np.isin(pre_i, g) & np.isin(post_i, g)
    assert len(sub.edges) == int(both.sum())
    # the control composes on a subnet and the flag survives (L6: engine warns, nothing is silently ignored)
    sh = C.shuffle_targets(sub, 0)
    assert sh.meta["dev_subnet"] and sh.meta["control"]["name"] == "shuffle_targets"
