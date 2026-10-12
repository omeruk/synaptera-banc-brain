"""F0 acceptance helpers: S1 / S2 criteria and the v1 comparison, checked on the v1 record itself."""
import pytest

from tasks import f0_flight as F

pytestmark = pytest.mark.skipif(not F.V1_RECORD.exists(), reason="needs the v1 n1 record")


def test_v1_n1_satisfies_s1_and_s2():
    c = F.criteria(F.V1_RECORD)
    assert c["S1"] and c["S2"] and c["tower_contact_steps"] == 0 and c["feeding_steps"] == 180
    assert c["landed_step"] is not None and c["first_feeding_step"] >= c["landed_step"]


def test_comparison_of_a_record_with_itself_is_zero():
    d = F.compare(F.V1_RECORD, F.V1_RECORD)
    assert d["pos_diff_mm"]["max"] == 0.0 and d["phase_identical"] and d["wings_on_identical"]
    assert d["n_steps"] == 300


def test_recorded_terms_cover_all_steps():
    t = F.RecordedTerms(F.V1_RECORD)
    assert t.n == 300 and set(t.at(0)) == set(F.RecordedTerms.FIELDS)
    assert t.meta["calib_steps"] == 10 and t.meta["persist_steps"] == 8
