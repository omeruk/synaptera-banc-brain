"""BANC adapter: neuron table (the Codex export in banc/ is data; tests skip without it)."""
import numpy as np
import pandas as pd
import pytest

from connectome.adapters import banc
from connectome.schema import NEURON_COLUMNS, validate_neurons
from conftest import ROOT

NEURONS = ROOT / "banc" / "neurons-banc.csv.gz"


@pytest.fixture(scope="module")
def neurons():
    return banc.load_neurons(NEURONS)


def test_neuron_table_on_common_schema(neurons):
    validate_neurons(neurons)
    assert len(neurons) == 158_262
    assert list(neurons.columns[:len(NEURON_COLUMNS)]) == list(NEURON_COLUMNS)
    assert neurons["region"].isna().all()           # assigned in F1, not here
    assert neurons["side"].value_counts().to_dict() == {"right": 82_502, "left": 75_760}


def test_descending_list_matches_first_look(neurons):
    dn = banc.descending_neurons(neurons)
    assert len(dn) == 1_316
    assert dn["side"].value_counts().to_dict() == {"right": 665, "left": 651}
    assert (neurons.loc[neurons["super_class"] == "descending", "id"].to_numpy() == dn["root_id"].to_numpy()).all()


def test_key_flight_types_present(neurons):
    ct = neurons["cell_type"].astype(str)
    assert (ct == "MN9").sum() == 2
    assert (ct == "DNp01").sum() == 2 and (ct == "DNa02").sum() == 2
    assert ct.str.match(r"^DNg02(_|$)").sum() >= 25


def test_toy_mapping_without_data(tmp_path):
    raw = pd.DataFrame({"Root ID": [1, 2], "Primary Cell Type": ["DNp01", np.nan],
                        "Alternative Cell Type(s)": [np.nan, np.nan], "Class": ["x", "y"],
                        "Super Class": ["descending", "sensory"], "Soma side": ["Left", np.nan],
                        "Predicted NT type": ["ACH", "HIST"], "Predicted NT confidence": [0.9, 0.5],
                        "Sub Class": [np.nan] * 2, "Body Part": [np.nan] * 2, "Function": [np.nan] * 2,
                        "Flow": ["efferent", "afferent"], "Verified NT type": [np.nan] * 2,
                        "Nerve": [np.nan] * 2, "Hemilineage": [np.nan] * 2})
    p = tmp_path / "n.csv.gz"
    raw.to_csv(p, index=False)
    n = banc.load_neurons(p)
    validate_neurons(n)
    assert n["side"].tolist() == ["left", "unknown"]
    dn = banc.descending_neurons(n)
    assert dn["root_id"].tolist() == [1] and dn["top_nt"].tolist() == ["ACH"]
