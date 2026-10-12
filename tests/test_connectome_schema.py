import numpy as np
import pandas as pd
import pytest

from connectome.schema import IdIndex, normalise_side, validate_edges, validate_neurons
from toy import edges_table, neurons_table


def test_idindex_roundtrip_and_missing():
    ids = np.array([900, 5, 77, 31], dtype=np.int64)
    ix = IdIndex(ids)
    np.testing.assert_array_equal(ix.index_of([77, 900, 5]), [2, 0, 1])
    np.testing.assert_array_equal(ix.ids_of([3, 1]), [31, 5])
    np.testing.assert_array_equal(ix.index_of([5, 4, 1000], missing="-1"), [1, -1, -1])
    with pytest.raises(KeyError):
        ix.index_of([5, 4])


def test_idindex_group_is_sorted_unique_and_ignores_absent():
    ix = IdIndex([10, 20, 30, 40, 50])
    np.testing.assert_array_equal(ix.group([50, 20, 20, 99, 10]), [0, 1, 4])
    assert ix.group([]).size == 0 and ix.group([99]).size == 0


def test_idindex_rejects_duplicates_and_empty_ok():
    with pytest.raises(ValueError):
        IdIndex([1, 2, 2])
    ix = IdIndex(np.zeros(0, dtype=np.int64))
    assert len(ix) == 0 and ix.index_of([], missing="-1").size == 0
    np.testing.assert_array_equal(ix.index_of([3], missing="-1"), [-1])


def test_normalise_side():
    s = normalise_side([" Left", "RIGHT", "center", None, np.nan, "mid"])
    assert s.tolist() == ["left", "right", "center", "unknown", "unknown", "unknown"]


def test_validate_neurons_and_edges():
    n = neurons_table([dict(id=1, cell_type="A", side="left"), dict(id=2, cell_type="B", side="right")])
    validate_neurons(n)
    with pytest.raises(ValueError, match="not unique"):
        validate_neurons(neurons_table([dict(id=1, side="left"), dict(id=1, side="left")]))
    with pytest.raises(ValueError, match="side"):
        validate_neurons(n.assign(side=["left", "up"]))
    with pytest.raises(ValueError, match="lacks"):
        validate_neurons(n.drop(columns=["nt"]))
    e = edges_table([(1, 2, 3)])
    validate_edges(e, IdIndex(n["id"]))
    with pytest.raises(ValueError, match="n_syn"):
        validate_edges(e.assign(n_syn=0))
    with pytest.raises(ValueError, match="sign"):
        validate_edges(e.assign(sign=2))
    with pytest.raises(KeyError):
        validate_edges(edges_table([(1, 3, 1)]), IdIndex(n["id"]))


def test_connectome_indices_follow_row_order_not_id_order():
    from connectome.schema import Connectome
    n = neurons_table([dict(id=30, cell_type="A", side="left"), dict(id=10, cell_type="B", side="right"),
                       dict(id=20, cell_type="A", side="right")])
    cx = Connectome("t", n, edges_table([(30, 10, 2), (10, 20, 4)], sign=-1))
    pre, post = cx.edge_endpoints()
    np.testing.assert_array_equal(pre, [0, 1])
    np.testing.assert_array_equal(post, [1, 2])
    np.testing.assert_array_equal(cx.signed_weights(), [-2, -4])
    np.testing.assert_array_equal(cx.by_type(["A"]), [0, 2])
    assert isinstance(cx.neurons, pd.DataFrame) and cx.n_neurons == 3
