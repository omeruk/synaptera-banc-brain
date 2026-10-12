import numpy as np
import pandas as pd

from connectome import homologs as H
from connectome.schema import Connectome
from toy import edges_table, lb3_like, neurons_table


def test_profiles_are_unit_norm_and_mirror_invariant():
    cx, pos, _ = lb3_like()
    ids = cx.neurons.loc[cx.neurons["cell_type"] == "X", "id"].to_numpy()
    x = H.partner_profiles(ids, cx)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)
    # a left and a right neuron that connect to the same-type ipsilateral partner have identical profiles
    left_pos = x[list(ids).index(0)]
    right_a = x[list(ids).index(50)]
    np.testing.assert_allclose(left_pos, right_a, atol=1e-12)
    # ...and differ from a neuron that targets the other partner type
    assert float(left_pos @ x[list(ids).index(3)]) < 0.5


def test_unknown_side_counts_as_contralateral():
    n = neurons_table([dict(id=1, cell_type="X", side="left"), dict(id=2, cell_type="P", side="unknown"),
                       dict(id=3, cell_type="P", side="left")])
    cx = Connectome("t", n, edges_table([(1, 2, 1), (1, 3, 1)]))
    # two features: out|P|c (unknown-side partner) and out|P|i (same-side partner), equal weight
    x = H.partner_profiles([1], cx)[0]
    assert sorted(np.round(x[x > 0], 6).tolist()) == [round(1 / np.sqrt(2), 6)] * 2


def test_select_homologs_toy():
    cx, pos, right_a = lb3_like()
    df, loo = H.select_homologs(cx, cell_type="X", seed_ids=pos, k=3, seed_source="seed")
    assert loo == dict(tp=3, n_positive=3, fp=0, n_negative=3)
    assert set(df.loc[df.source == "seed", "id"]) == set(pos)
    assert set(df.loc[df.source != "seed", "id"]) == set(right_a)
    assert df.loc[df.source != "seed", "knn_frac"].eq(1.0).all()
    assert (df.loc[df.source != "seed", "side"] == "right").all()


def test_knn_leave_one_out_excludes_self():
    pool = np.eye(4)
    positive = np.array([1, 0, 0, 0], bool)
    # with self excluded, row 0 cannot vote for itself: its 3 neighbours are all negative
    f = H.knn_positive_fraction(pool, positive, pool, k=1, exclude_self=True)
    assert f[0] == 0.0
    f_self = H.knn_positive_fraction(pool, positive, pool, k=1, exclude_self=False)
    assert f_self[0] == 1.0


def test_knn_tie_break_is_pool_order():
    pool = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
    positive = np.array([False, True, True])
    # query equidistant to pool 0 and 1: stable order picks pool 0 first
    assert H.knn_positive_fraction(pool, positive, np.array([[1.0, 0]]), k=1)[0] == 0.0


def test_sugar_homologs_reproduce_v1_table():
    """FlyWire v783: seeds = Shiu LB3 (left); result = data/sugar_grn_783.csv of Synaptera v1 (36 neurons)."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    df, loo = H.select_homologs(cx, cell_type="LB3", seed_ids=fw.SHIU_SUGAR_IDS, seed_source="shiu2023")
    ref = pd.read_csv(fw.REPO_ROOT / "data" / "sugar_grn_783.csv")
    assert sorted(df["id"]) == sorted(ref["root_id"])
    assert loo == dict(tp=19, n_positive=20, fp=2, n_negative=44)
    merged = df.merge(ref, left_on="id", right_on="root_id", suffixes=("", "_ref"))
    assert (merged["source"] == merged["source_ref"]).all() and (merged["side"] == merged["side_ref"]).all()
