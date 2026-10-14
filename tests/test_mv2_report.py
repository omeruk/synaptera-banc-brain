"""prereg/12 constants and the statistics of scripts/mv2_report.py (no simulation)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mv2_common as M  # noqa: E402
import mv2_report as R  # noqa: E402

PREREG = (ROOT / "prereg" / "12_neuron_model_v2.md").read_text()


def test_constants_match_the_prereg():
    assert M.ARMS == ("R", "A", "AC", "ACG") and M.REDUCED_ARMS == ("R", "ACG")
    assert M.T1_SEEDS == tuple(range(300, 350)) and "300-349" in PREREG
    assert M.T3_SEEDS == tuple(range(30, 40)) and "fresh brain seeds 30-39" in PREREG
    assert M.T4_SEEDS == tuple(range(10)) and M.T4_ARMS == ("R", "ACG")
    assert M.T3_N_RANDOM == 19 and M.T3_ARMS[:2] == ("ensemble", "bridge_off") and M.T3_ARMS[-1] == "random_19"
    assert M.T1_MN9_MIN_FRACTION == 0.10 and M.T1_IGNITION_HZ == 1.0
    assert M.T4_FOLLOW_MIN == 0.9 and M.T4_LATENCY_MAX_MS == 1.0
    assert M.RUNTIME_LIMIT_H == 6.0


def test_primary_mapping_is_the_v2_lateral_set():
    import f4_design_numbers as F4
    assert R.SETS["lateral"] == F4.SIGNS_LATERAL_V2
    assert R.SETS["direct"] == F4.SIGNS_DIRECT
    assert {"lateral+iii4(+)", "lateral+iii4(-)"} <= set(R.SETS)
    g1 = json.loads(R.F4_V1.read_text())["gain"]["G1_deg_per_hz"]
    assert f"{g1:.4f}" in PREREG


def test_one_sided_p_conventions():
    assert R.p_one_sided(np.zeros(10)) == 1.0
    assert R.p_one_sided(np.ones(10)) == 0.0
    assert R.p_one_sided(-np.ones(10)) == 1.0
    x = np.array([1.0, 2.0, 0.5, 1.5, -0.2, 1.1, 0.9, 1.3, 0.7, 1.0])
    assert R.p_one_sided(x) < 0.001
    assert R.p_one_sided(-x) > 0.99
    from scipy.stats import ttest_1samp
    assert R.p_one_sided(x) == pytest.approx(ttest_1samp(x, 0).pvalue / 2)


def test_markdown_formatting_helpers():
    assert R.f(None) == "-" and R.f(1.234) == "1.23" and R.f(1.0, 1, True) == "+1.0"
    assert R.pf(1e-6) == "<0.0001" and R.pf(0.0312) == "0.0312"
