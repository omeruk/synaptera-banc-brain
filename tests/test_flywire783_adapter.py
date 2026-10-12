"""FlyWire v783 adapter on the real Shiu files (skipped when brain_model/ is absent)."""
import numpy as np
import pandas as pd
import pytest

from connectome.adapters import flywire783 as fw
from connectome.schema import NEURON_COLUMNS

N_NEURONS = 138_639
N_SYN = 15_091_983
SUM_EXC_X_CONN = 10_496_512


@pytest.fixture(scope="module")
def cx():
    return fw.load_connectome(hash_files=False)


def test_completeness_rows_are_engine_order(cx):
    ids = fw.load_root_ids()
    assert len(ids) == N_NEURONS and len(np.unique(ids)) == N_NEURONS
    np.testing.assert_array_equal(cx.neurons["id"].to_numpy(), ids)
    assert list(cx.neurons.columns[:len(NEURON_COLUMNS)]) == list(NEURON_COLUMNS)


def test_edges_match_parquet_indices_and_shiu_signs(cx):
    con = pd.read_parquet(fw.BRAIN_DIR / fw.FILES["connectivity"],
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    assert len(cx.edges) == N_SYN
    pre, post = cx.edge_endpoints()
    np.testing.assert_array_equal(pre, con["Presynaptic_Index"].to_numpy())
    np.testing.assert_array_equal(post, con["Postsynaptic_Index"].to_numpy())
    np.testing.assert_array_equal(cx.signed_weights(), con["Excitatory x Connectivity"].to_numpy())
    assert int(cx.signed_weights().sum()) == SUM_EXC_X_CONN


def test_sign_is_unique_per_presynaptic_neuron(cx):
    pre, _ = cx.edge_endpoints()
    s = cx.edges["sign"].to_numpy()
    lo = np.full(cx.n_neurons, 2)
    hi = np.full(cx.n_neurons, -2)
    np.minimum.at(lo, pre, s)
    np.maximum.at(hi, pre, s)
    has = hi > -2
    assert (lo[has] == hi[has]).all()


def test_sides_and_types_present(cx):
    assert set(cx.neurons["side"]) <= {"left", "right", "center", "unknown"}
    assert (cx.neurons["cell_type"] == "LB3").sum() >= 56
    assert cx.neurons["region"].eq("brain").all()


def test_descending_view_matches_v1_dn_table(cx):
    """The DN view equals brain_model/descending_neurons.csv (v1) as a set of (root_id, cell_type, top_nt, side)."""
    dn = fw.descending_neurons(cx.neurons)
    ref = pd.read_csv(fw.BRAIN_DIR / "descending_neurons.csv")
    assert len(dn) == len(ref) == 1299
    assert list(dn.columns) == list(ref.columns) == list(fw.LEGACY_DN_COLUMNS)
    m = dn.merge(ref, on="root_id", suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == 1299
    assert (m["cell_type"].astype(str) == m["cell_type_ref"].astype(str)).all()
    assert (m["top_nt"].astype(str) == m["top_nt_ref"].astype(str)).all()
    # v1 kept the raw side label (left / right / center); the schema normalises the same values
    assert (m["side"] == m["side_ref"].astype(str).str.strip().str.lower()
            .where(lambda s: s.isin(["left", "right", "center"]), "unknown")).all()
    assert dn["side"].value_counts().to_dict() == {"right": 646, "left": 645, "center": 8}
