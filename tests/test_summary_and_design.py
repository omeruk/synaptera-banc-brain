"""Summary generator (scripts/summary_f0_f3.py) and design numbers (scripts/f4_design_numbers.py): no simulation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import f4_design_numbers as F  # noqa: E402
import summary_f0_f3 as S  # noqa: E402


def test_summary_template_has_no_typed_numbers():
    assert S.stray_numbers(S.TEMPLATE) == []


def test_summary_render_fills_every_placeholder():
    vals = {k: ["1", "2", "3", "4", "5", "6", "7", "8"] for k in S.FACTS}
    text = S.render(S.TEMPLATE, vals)
    assert "{{" not in text and "}}" not in text


def test_summary_stray_number_is_detected():
    assert S.stray_numbers("A typed number 42 here") == ["42"]
    assert S.stray_numbers("F3b-3 and `0.5` and [f3b3_specificity](f3b3_specificity.md) {{c_T.1}}") == []


@pytest.mark.skipif(not (ROOT / "docs" / "results" / "f3b3_specificity.md").exists(), reason="reports missing")
def test_summary_facts_match_reports():
    vals = S.capture()
    assert set(vals) == set(S.FACTS)
    assert (ROOT / "docs" / "results" / "SUMMARY_F0_F3.md").read_text() == S.render(S.TEMPLATE, vals)


def test_mapping_sets_are_consistent():
    assert set(F.SIGNS_DIRECT) <= set(F.SIGNS_LATERAL)
    assert all(F.SIGNS_LATERAL[t] == s for t, s in F.SIGNS_DIRECT.items())
    assert not set(F.SIGNS_LATERAL) & set(F.GATE_FAIL)          # b1 and iii3 fail the symmetry gate
    assert all(v in (-1, 1) for v in F.SIGNS_LATERAL.values())


def test_mapping_v2_follows_the_registered_rule():
    # primary v2 set: signs stated directly by Lindsay et al. 2017, gate-passing; iii1 / iii4 (text vs Fig. 6) excluded
    assert set(F.SIGNS_LATERAL_V2) == {"b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4"}
    assert not set(F.SIGNS_LATERAL_V2) & set(F.GATE_FAIL)
    assert all(F.SIGNS_LATERAL_V2[t] == s for t, s in F.SIGNS_DIRECT.items())
    assert F.SETS_V2["direct"] == F.SIGNS_DIRECT
    for t in ("iii1", "iii4"):
        assert {F.SETS_V2[f"lateral+{t}(+)"][t], F.SETS_V2[f"lateral+{t}(-)"][t]} == {1, -1}
    # v1 is kept unchanged and written to its own file
    assert F.VERSIONS["v1"]["out"] != F.VERSIONS["v2"]["out"]


def _fake(rr_r, rr_l):
    # one type b3 (s = -1): per-neuron rates left / right per stimulus
    def rec(l, r):
        return {"b3|left": l, "b3|right": r}
    return {"x": {"rot_right": {0: rec(*rr_r)}, "rot_left": {0: rec(*rr_l)}, "static": {0: rec(0, 0)}, "grey": {0: rec(0, 0)}}}


def test_syndirectional_index_sign_convention():
    # b3 lowers the ipsilateral amplitude (s = -1). Right side more active under rot_right and left side more active
    # under rot_left: A_R falls under rot_right, so A_L - A_R > 0 (right turn, with the drum) and S > 0.
    rates = _fake((0.0, 10.0), (10.0, 0.0))
    S_ = F.s_index(rates, "x", [0], {"b3": -1})
    assert S_[0] > 0
    # sign_flip equals the opposite of the original
    assert F.s_index(rates, "x", [0], {"b3": +1})[0] == -S_[0]
    # equals - sum s D / 2
    d = F.d_per_type(rates, "x", [0], ["b3"])["b3"][0]
    assert S_[0] == pytest.approx(-(-1) * d / 2.0)


def test_summarise_constant_has_unit_p():
    r = F.summarise(np.zeros(5))
    assert r["p_t"] == 1.0 and r["n_pos"] == 0 and r["n_neg"] == 0
