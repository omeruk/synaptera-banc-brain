"""prereg/13 side audit: the side rule, the per-type verdict, the record relabelling and the bridge side check
(no simulation)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import side_audit as A  # noqa: E402

PREREG = (ROOT / "prereg" / "13_side_audit.md").read_text()


def test_nerve_side_parser():
    assert A.nerve_side("left_posterior_dorsal_mesothoracic_nerve") == "left"
    assert A.nerve_side("right_cervical_nerve") == "right"
    assert A.nerve_side("abdominal_nerve_trunk") is None
    assert A.nerve_side(None) is None and A.nerve_side(np.nan) is None
    assert A.nerve_side("leftover_nerve") is None


def test_neuron_and_type_verdicts():
    assert A.neuron_verdict("left", "left") == "match"
    assert A.neuron_verdict("left", "right") == "relabelled"
    assert A.neuron_verdict("left", None) == "undetermined"
    assert A.type_verdict(["match", "match"]) == "consistent"
    assert A.type_verdict(["relabelled", "relabelled"]) == "swapped"
    assert A.type_verdict(["match", "relabelled"]) == "mixed"
    assert A.type_verdict(["undetermined", "undetermined"]) == "undetermined"
    assert A.type_verdict(["match", "undetermined"]) == "consistent_partial"
    assert A.type_verdict(["relabelled", "undetermined"]) == "mixed"
    assert A.type_verdict([]) == "undetermined"


def test_relabel_rates_swaps_and_removes():
    r = {"DLM5|left": 1.0, "DLM5|right": 2.0, "CvN7|left": 3.0, "hg4|left": 4.0, "hg4|right": 5.0}
    out, rm = A.relabel_rates(r, {"DLM5": "swapped", "CvN7": "mixed", "hg4": "consistent"})
    assert out == {"DLM5|right": 1.0, "DLM5|left": 2.0, "hg4|left": 4.0, "hg4|right": 5.0}
    assert rm == ["CvN7"]
    # a type absent from the verdict map is kept as it is
    out, rm = A.relabel_rates({"x|left": 1.0}, {})
    assert out == {"x|left": 1.0} and rm == []


def test_bridge_keeps_sides():
    from coupling.dn_bridge import BridgeConfig, DNBridge
    fw = pd.DataFrame(dict(cell_type=["DNa02", "DNa02"], side=["left", "right"]))
    banc = pd.DataFrame(dict(cell_type=["DNa02", "DNa02"], side=["right", "left"], index=[10, 11]))
    b = DNBridge(fw, banc, BridgeConfig(families=("DNa02",), latency_intervals=1, window_intervals=1))
    c = np.zeros((3, 2))
    c[0, 0] = 1.0                                        # FlyWire left only
    r = b.rates(c)[1]
    side = banc.set_index("index").loc[b.target_index, "side"].to_numpy()
    assert (r[side == "left"] > 0).all() and (r[side == "right"] == 0).all()


def test_constants_match_the_prereg():
    assert A.L4_MIN == 0.99 and ">= 99 %" in PREREG
    assert A.L6_MIN == 0.95 and ">= 95 %" in PREREG
    assert A.T45_FTB == ("T4a", "T5a") and "(T4a + T5a) - (T4b + T5b)" in PREREG
    assert set(A.AZEVEDO_NERVE.values()) == set(A.NERVE_NAMES)
    assert A.SOMA_LIT == {"DLM1-4": "ipsilateral", "DLM5": "contralateral"}
    assert "MN1-4 ipsilateral, MN5 contralateral" in PREREG


@pytest.mark.skipif(not (ROOT / "docs" / "results" / "side_audit.json").exists(), reason="audit not run")
def test_stored_audit_mapping_types_consistent():
    import json
    o = json.loads((ROOT / "docs" / "results" / "side_audit.json").read_text())
    for t in ("b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4"):
        assert o["section1"]["types"][t]["verdict"] == "consistent"
    assert set(o["chain_verdicts"].values()) <= {"OK", "SWAPPED", "UNRESOLVED"}
