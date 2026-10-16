"""scripts/verify_report.py: rebuilt JSON is compared value by value, floats within a relative 1e-12, and every float
that needed the tolerance is listed."""
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as VR  # noqa: E402


@pytest.fixture(autouse=True)
def empty_log():
    VR.TOLERATED.clear()
    yield
    VR.TOLERATED.clear()


def test_exact_values_need_no_tolerance():
    d = {"a": [1, 2.5, "x", None, True], "b": {"c": float("nan")}}
    assert VR.json_close("k", d, {"b": {"c": float("nan")}, "a": (1, 2.5, "x", None, True)})
    assert VR.TOLERATED == []


def test_last_digit_difference_is_accepted_and_listed():
    assert VR.json_close("k", {"x": [-63.62019341269978]}, {"x": [-63.62019341269979]})
    (key, path, a, b, rel), = VR.TOLERATED
    assert (key, path, a, b) == ("k", "/x[0]", -63.62019341269978, -63.62019341269979)
    assert 0 < rel <= VR.JSON_REL_TOL


@pytest.mark.parametrize("stored, rebuilt", [
    ({"x": 1.0}, {"x": 1.0 + 1e-9}),            # beyond the tolerance
    ({"x": 0.0}, {"x": 1e-300}),                # relative to zero
    ({"x": 1}, {"x": 1.0}),                     # int vs float
    ({"x": 1.0}, {"x": 1.0, "y": 2}),           # extra key
    ({"x": [1.0]}, {"x": [1.0, 2.0]}),          # length
    ({"x": "a"}, {"x": "b"}),
    ({"x": float("nan")}, {"x": 1.0}),
])
def test_differences_are_mismatches(stored, rebuilt):
    assert not VR.json_close("k", stored, rebuilt)
    assert VR.TOLERATED == []


def test_tolerated_values_of_a_failed_comparison_are_not_listed():
    assert not VR.json_close("k", {"a": 1.0, "b": "x"}, {"a": 1.0 + 1e-15, "b": "y"})
    assert VR.TOLERATED == []
