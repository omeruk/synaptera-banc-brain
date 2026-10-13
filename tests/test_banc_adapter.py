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


# ── F1: regions, signed connectome, data sources ─────────────────────────────
def _toy_files(tmp_path):
    n = pd.DataFrame({"Root ID": [1, 2, 3, 4, 5], "Primary Cell Type": ["A", "A", "H", "G", np.nan],
                      "Alternative Cell Type(s)": [np.nan] * 5, "Class": ["x"] * 5,
                      "Super Class": ["central_brain_intrinsic"] * 5, "Soma side": ["left", "right", "left", "right", "left"],
                      "Predicted NT type": ["ACH", "GLUT", "HIST", "GABA", np.nan],
                      "Predicted NT confidence": [0.9, 0.8, 0.7, 0.9, np.nan],
                      "Sub Class": [np.nan] * 5, "Body Part": [np.nan] * 5, "Function": [np.nan] * 5,
                      "Flow": [np.nan] * 5, "Verified NT type": [np.nan] * 5, "Nerve": [np.nan] * 5,
                      "Hemilineage": [np.nan] * 5})
    c = pd.DataFrame({"pre_root_id": [1, 1, 2, 3, 4, 5], "post_root_id": [2, 2, 3, 4, 1, 1],
                      "neuropil": ["ME_R", "ME_L", "VNC_T1_ProNm_L", "GNG", "AVLP_R", "ME_R"],
                      "syn_count": [3, 4, 5, 6, 7, 2], "nt_type": [np.nan] * 6})
    n.to_csv(tmp_path / "n.csv.gz", index=False)
    c.to_csv(tmp_path / "c.csv.gz", index=False)
    return tmp_path / "n.csv.gz", tmp_path / "c.csv.gz"


def test_region_of_neuropil_rules():
    r = banc.region_of_neuropil
    assert [r(x) for x in ("VNC_T2_MesoNm_R", "ABDNM_L", "IntTct", "LegNp_T2_L", "WTct_UTct_T2_R", "ANm")] == ["VNC"] * 6
    assert [r(x) for x in ("GNG", "PRW", "SAD", "FLA_L", "CAN_R")] == ["SEZ"] * 5
    assert r("cervical_connective") == "neck"
    assert [r(x) for x in ("ME_R", "AL_L", "FB", "MB_CA_R")] == ["brain"] * 4
    assert banc.is_optic("ME_R") and banc.is_optic("LOP_L") and not banc.is_optic("AL_L")


def test_signed_connectome_reference_variants_and_regions(tmp_path):
    n, c = _toy_files(tmp_path)
    cx = banc.load_connectome(n, c, hash_files=False)
    assert cx.signed and cx.meta["sign_rule"] == "shiu_v1" and cx.n_neurons == 5
    # pairs summed over neuropils: (1,2) = 3 + 4
    e = cx.edges.set_index(["pre", "post"])
    assert e.loc[(1, 2), "n_syn"] == 7 and len(e) == 5
    # reference rule: GLUT / GABA -1; ACH, HIST and the missing prediction +1
    sign = cx.edges.groupby("pre")["sign"].first().to_dict()
    assert sign == {1: 1, 2: -1, 3: 1, 4: -1, 5: 1}
    assert cx.meta["nt_counts"]["HIST"]["neurons"] == 1 and cx.meta["nt_counts"]["none"]["neurons"] == 1
    # regions: neuron 1 (ME_R, ME_L, AVLP_R, ME_R -> brain), neuron 3 (VNC / GNG tie -> weights 5 vs 6+...)
    reg = cx.neurons.set_index("id")["region"]
    assert reg[1] == "brain" and reg[2] in ("brain", "VNC")
    h = banc.load_connectome(n, c, variants=("hist_inhibitory",), hash_files=False)
    assert h.edges.groupby("pre")["sign"].first().to_dict()[3] == -1 and h.meta["sign_variants"] == ["hist_inhibitory"]
    assert h.meta["sign_rules"][0]["name"] == "hist_inhibitory"


def test_min_syn_threshold_is_an_explicit_recorded_option(tmp_path):
    n, c = _toy_files(tmp_path)
    full = banc.load_connectome(n, c, hash_files=False)
    assert full.meta["min_syn"] is None and full.meta["pairs_kept"] == full.meta["pairs_before_threshold"] == 5
    th = banc.load_connectome(n, c, hash_files=False, min_syn=6)
    assert th.meta["min_syn"] == 6 and th.meta["pairs_before_threshold"] == 5
    assert th.meta["pairs_kept"] == len(th.edges) < 5 and (th.edges["n_syn"] >= 6).all()
    assert th.meta["synapses_kept"] == int(th.edges["n_syn"].sum())
    assert th.n_neurons == full.n_neurons          # neurons are kept; only pairs are dropped


@pytest.mark.skipif(not (ROOT / "banc" / "neurons-banc.csv.gz").exists(), reason="needs banc/")
def test_data_sources_hashes_match_local_files():
    import yaml

    from connectome.adapters import file_sha256
    src = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())["sources"]
    checked = 0
    for name, d in src.items():
        p = ROOT / d["path"]
        if p.exists():
            assert file_sha256(p) == d["sha256"], name
            checked += 1
    assert checked >= 2
