"""Per-neuron input normalisation (prereg/05 section 2.1) on toy graphs."""
import numpy as np
import pandas as pd
import pytest

from connectome import innorm
from connectome.schema import Connectome, NEURON_COLUMNS


def _neurons(ids, types, sc):
    d = {c: [None] * len(ids) for c in NEURON_COLUMNS}
    d.update(id=np.asarray(ids, np.int64), cell_type=types, super_class=sc, side=["left"] * len(ids),
             top_nt=["ACH"] * len(ids))
    df = pd.DataFrame(d)
    for c in NEURON_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def _cx(name, ids, types, sc, edges):
    e = pd.DataFrame(edges, columns=["pre", "post", "n_syn", "sign"])
    e["pre"], e["post"] = e["pre"].astype(np.int64), e["post"].astype(np.int64)
    n = _neurons(ids, types, sc)
    n["cell_class"] = None
    return Connectome(name, n, e, {})


def test_factors_ratio_fallback_and_clip():
    # FlyWire: type X neurons (1, 2) receive 10 and 30 synapses (mean 20); type Y neuron 3 receives 4 (mean 4)
    fx = _cx("fw", [1, 2, 3, 9], ["X", "X", "Y", "Z"], ["central"] * 4,
             [(9, 1, 10, 1), (9, 2, 30, 1), (9, 3, 4, 1), (1, 9, 1, 1)])
    # BANC: X neurons (11, 12) receive 5 and 5 (mean 5) -> s 4; Y neuron 13 receives 20 -> s 0.2 -> clip 0.25;
    # neuron 14 (type W, no homolog, class central) gets the median of the central homologs; 15 (MN9) maps to Z
    bx = _cx("banc", [11, 12, 13, 14, 15, 19], ["X", "X", "Y", "W", "MN9", None],
             ["central_brain_intrinsic", "central_brain_intrinsic", "central_brain_intrinsic",
              "central_brain_intrinsic", "motor", "glia"],
             [(19, 11, 5, 1), (19, 12, 5, 1), (19, 13, 20, 1), (19, 14, 7, 1), (19, 15, 1, 1)])
    s, rep = innorm.innorm_factors(fx, bx)
    assert s[0] == pytest.approx(4.0) and s[1] == pytest.approx(4.0)          # 20 / 5
    assert s[2] == 0.25                                                       # 4 / 20 = 0.2 -> clipped
    assert s[4] == pytest.approx(4.0) and rep["named"]["CB0701"] is None      # MN9 -> CB0701 not in FlyWire here
    assert rep["n_clipped_low"] == 1 and rep["n_homolog"] == 3 and rep["n_fallback"] == 3
    assert s[3] == pytest.approx(np.median([4.0, 4.0, 0.25]))                 # class median over homologs (clipped later)


def test_named_entries_and_in_synapse_levels():
    fx = _cx("fw", [1, 2], ["CB0701", "LB3"], ["motor", "sensory"], [(1, 2, 8, 1), (2, 1, 6, 1)])
    bx = _cx("banc", [11, 12, 13], ["MN9", "LB3a", "LB3b"], ["motor", "sensory", "sensory"],
             [(11, 12, 2, 1), (12, 11, 3, 1), (13, 12, 2, 1)])
    s, rep = innorm.innorm_factors(fx, bx, clip=(0.01, 100))
    assert s[0] == pytest.approx(6 / 3)                       # MN9 <- CB0701: FlyWire in 6, BANC in 3
    # LB3a (12) receives 2 + 2 = 4, LB3b (13) receives 0: pooled mean 2, FlyWire LB3 receives 8 -> s = 4
    assert s[1] == s[2] == pytest.approx(4.0)
    assert rep["named"]["CB0701"] == pytest.approx(2.0)
    assert rep["named"]["LB3"] == pytest.approx(4.0)


def test_scaled_view_multiplies_by_post_factor():
    fx = _cx("fw", [1, 2], ["A", "B"], ["central"] * 2, [(1, 2, 8, 1)])
    s = np.array([1.0, 0.5])
    v = innorm.scaled_view(fx, s)
    assert v.edges["n_syn"].tolist() == [4.0] and v.n_neurons == 2
    assert fx.edges["n_syn"].tolist() == [8]
