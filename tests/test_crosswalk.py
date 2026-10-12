import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from connectome import crosswalk as cwk
from connectome import homologs as H
from connectome.schema import Connectome
from conftest import ROOT
from toy import edges_table, neurons_table


def _fly():
    return neurons_table([dict(id=1, cell_type="CB0701", side="left"), dict(id=2, cell_type="CB0701", side="right"),
                          dict(id=3, cell_type="DNp01", side="left"), dict(id=4, cell_type=np.nan, side="right"),
                          dict(id=5, cell_type="LB3", side="left")]).assign(cell_type_alt=[np.nan] * 3 + ["DNp01", np.nan])


def _banc():
    n = neurons_table([dict(id=11, cell_type="MN9", side="left"), dict(id=12, cell_type="LB3a", side="left"),
                       dict(id=13, cell_type="LB3b", side="right"), dict(id=14, cell_type="b1", side="left"),
                       dict(id=15, cell_type="b2", side="right"), dict(id=16, cell_type="x", side="right")])
    n["body_part"] = [np.nan, np.nan, np.nan, "wing", "wing", "haltere"]
    n["super_class"] = ["motor", "sensory", "sensory", "motor", "motor", "sensory_ascending"]
    return n


def test_name_alias_alt_and_filter_rows():
    cw = cwk.crosswalk_by_name({"flywire783": _fly(), "banc": _banc()})
    assert list(cw.columns) == list(cwk.CROSSWALK_COLUMNS)
    mn9 = cw[cw["query"] == "MN9"]
    assert sorted(mn9.loc[mn9.dataset == "flywire783", "id"]) == [1, 2]        # CB0701 alias
    assert mn9.loc[mn9.dataset == "banc", "id"].tolist() == [11]
    dn = cw[(cw["query"] == "DNp01") & (cw["dataset"] == "flywire783")].set_index("id")["source"]
    assert dn.to_dict() == {3: "name", 4: "name_alt"}
    lb3 = cw[cw["query"] == "LB3"]
    assert sorted(lb3.loc[lb3.dataset == "banc", "id"]) == [12, 13] and lb3.loc[lb3.dataset == "flywire783", "id"].tolist() == [5]
    wing = cw[cw["query"] == "wing_MN"]
    assert sorted(wing["id"]) == [14, 15] and set(wing["dataset"]) == {"banc"} and set(wing["source"]) == {"filter"}
    assert cw[cw["query"] == "haltere_afferent"]["id"].tolist() == [16]        # tuple value = any of
    assert "flywire783" not in set(cw[cw["query"].isin(["wing_MN", "haltere_MN"])]["dataset"])


def test_summary_counts():
    cw = cwk.crosswalk_by_name({"banc": _banc()})
    s = cwk.crosswalk_summary(cw).set_index(["query", "dataset"])
    assert s.loc[("wing_MN", "banc"), ["n", "n_left", "n_right"]].tolist() == [2, 1, 1]


def test_cross_dataset_cosine_toy():
    # dataset A: neuron 1 -> P (ipsi, 3 syn), Q (contra, 1 syn); dataset B: neuron 9 does the same, 8 only P.
    na = neurons_table([dict(id=1, cell_type="M", side="left"), dict(id=2, cell_type="P", side="left"),
                        dict(id=3, cell_type="Q", side="right")])
    nb = neurons_table([dict(id=9, cell_type="M", side="left"), dict(id=8, cell_type="M", side="left"),
                        dict(id=20, cell_type="P", side="left"), dict(id=21, cell_type="Q", side="right"),
                        dict(id=22, cell_type="Z", side="left")])
    a = Connectome("a", na, edges_table([(1, 2, 3), (1, 3, 1)]))
    b = Connectome("b", nb, edges_table([(9, 20, 3), (9, 21, 1), (8, 20, 5), (8, 22, 5)]).drop(columns="sign"))
    cos = H.cross_dataset_cosine([1], a, [9, 8], b)
    assert cos.loc[1, "cos_9"] == pytest.approx(1.0)
    assert 0.0 < cos.loc[1, "cos_8"] < 1.0
    assert cos.loc[1, "shared_frac_a"] == pytest.approx(1.0)
    assert cos.attrs["shared_frac_b"] == pytest.approx([1.0, 0.5])    # neuron 8: half its weight is on type Z


def test_unsigned_connectome_refuses_signed_weights():
    n = neurons_table([dict(id=1, side="left"), dict(id=2, side="left")])
    cx = Connectome("u", n, edges_table([(1, 2, 3)]).drop(columns="sign"))
    assert not cx.signed
    with pytest.raises(ValueError, match="sign"):
        cx.signed_weights()


NEEDS_DATA = pytest.mark.skipif(not (ROOT / "banc" / "neurons-banc.csv.gz").exists()
                                or not (ROOT / "brain_model" / "flywire_annotations.tsv").exists(),
                                reason="needs banc/ and brain_model/")


@NEEDS_DATA
def test_real_crosswalk_counts():
    from connectome.adapters import banc, flywire783 as fw
    cw = cwk.crosswalk_by_name({"flywire783": fw.load_neurons(), "banc": banc.load_neurons(ROOT / "banc" / "neurons-banc.csv.gz")})
    s = cwk.crosswalk_summary(cw).set_index(["query", "dataset"])["n"]
    for q in ("DNp01", "DNa02", "DNp15", "DNp07", "MN9", "TTMn"):
        assert s[(q, "banc")] == 2, q
    assert s[("MN9", "flywire783")] == 2 and s[("DNg02", "flywire783")] == 25
    assert s[("wing_MN", "banc")] == 64 and s[("haltere_MN", "banc")] == 27


@NEEDS_DATA
def test_script_refuses_overwrite_and_writes_meta(tmp_path):
    out = tmp_path / "cw.csv"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "make_crosswalk.py"), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    meta = json.loads((tmp_path / "cw.csv.meta.json").read_text())
    assert "banc_neurons" in meta["inputs"] and len(meta["inputs"]["banc_neurons"]) == 64
    before = out.read_bytes()
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode != 0 and "exists" in (r2.stderr + r2.stdout)
    assert out.read_bytes() == before
    assert len(pd.read_csv(out)) > 100
