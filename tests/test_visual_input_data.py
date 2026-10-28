"""docs/figures/input_checks.csv and visual_input.json (visual input of stage A, scripts/readme_visual_input.py): the CSV
schema and row counts, test rows without values, the p0 rows equal to input_checks_batch0, the JSON's internal
consistency, check_visual_input of scripts/verify_report.py (passes on the committed files, fails on perturbed copies
in a temporary folder), and the reimplemented a_in against f3b1_brain.a_in."""
import csv
import json
import shutil
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as V  # noqa: E402

FIG = ROOT / "docs" / "figures"
REC = json.loads((ROOT / "docs" / "results" / "f5_stageA_records.json").read_text())
COLUMNS = ["group", "split", "speed_deg_s", "phase_deg", "passed", "A_in_rot_left", "A_in_rot_right"]


def _rows(path=FIG / "input_checks.csv"):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def test_csv_schema_and_counts():
    cols, rows = _rows()
    assert cols == COLUMNS
    assert len(rows) == REC["completion"]["input_checks_total"]
    for s in ("train", "val", "test"):
        assert sum(r["split"] == s for r in rows) == REC["completion"]["input_checks"][s]["groups"]
    order = {"train": 0, "val": 1, "test": 2}
    keys = [(order[r["split"]], 0 if r["group"] == "p0" else int(r["group"][1:]), int(r["speed_deg_s"])) for r in rows]
    assert keys == sorted(keys)


def test_test_rows_have_no_values():
    _, rows = _rows()
    test = [r for r in rows if r["split"] == "test"]
    assert test and all(r["A_in_rot_left"] == "" and r["A_in_rot_right"] == "" for r in test)
    assert all(r["A_in_rot_left"] and r["A_in_rot_right"] for r in rows if r["split"] != "test")


def test_p0_rows_equal_batch0():
    _, rows = _rows()
    p0 = {f"p0_v{int(r['speed_deg_s']):03d}": r for r in rows if r["group"] == "p0"}
    assert sorted(p0) == sorted(REC["input_checks_batch0"])
    for name, v in REC["input_checks_batch0"].items():
        for st in ("rot_left", "rot_right"):
            assert float(p0[name][f"A_in_{st}"]) == v["A_in_T4aT5a"][st]


def test_json_internal_consistency():
    d = json.loads((FIG / "visual_input.json").read_text())
    assert d["gate"]["passed"] is True and all(c["equal"] for c in d["gate"]["comparisons"])
    assert set(d["mean_rate_hz"]) == set(d["asymmetry"]["values"])
    for e, v in d["neurons"].items():
        assert list(v["per_subtype"]) == d["subtypes"] and sum(v["per_subtype"].values()) == v["t4t5_total"]
    assert len(d["eye_images"]["images"]) > 0 and all(len(x["sha256"]) == 64 for x in d["eye_images"]["images"])


def test_check_passes_on_the_committed_files():
    n, bad = V.check_visual_input()
    assert n > 0 and not bad, bad[:5]


def _copy(tmp_path):
    for f in ("input_checks.csv", "visual_input.json"):
        shutil.copy(FIG / f, tmp_path / f)
    return tmp_path


@pytest.mark.parametrize("edit", ["test_value", "p0_value", "drop_row", "phase"])
def test_check_fails_on_a_perturbed_csv(tmp_path, edit):
    d = _copy(tmp_path)
    cols, rows = _rows(d / "input_checks.csv")
    if edit == "test_value":
        next(r for r in rows if r["split"] == "test")["A_in_rot_left"] = "0.2"
    elif edit == "p0_value":
        r = next(r for r in rows if r["group"] == "p0")
        r["A_in_rot_left"] = repr(float(r["A_in_rot_left"]) * (1 + 1e-9))
    elif edit == "drop_row":
        rows.pop(-1)
    else:
        next(r for r in rows if r["group"] != "p0")["phase_deg"] = "0.0"
    with open(d / "input_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    assert V.check_visual_input(d)[1]


@pytest.mark.parametrize("edit", ["asymmetry", "total", "gate_flag", "window"])
def test_check_fails_on_a_perturbed_json(tmp_path, edit):
    d = _copy(tmp_path)
    j = json.loads((d / "visual_input.json").read_text())
    if edit == "asymmetry":
        key = next(iter(j["asymmetry"]["values"]))
        t = j["subtypes"][0]
        j["asymmetry"]["values"][key][t] *= 1 + 1e-9
    elif edit == "total":
        j["neurons"]["L"]["t4t5_total"] += 1
    elif edit == "gate_flag":
        j["gate"]["comparisons"][0]["equal"] = False
    else:
        j["window"]["window"][0] += 1
    (d / "visual_input.json").write_text(json.dumps(j))
    assert V.check_visual_input(d)[1]


def test_a_in_equals_f3b1_brain():
    from f3b1_brain import a_in
    vals = [(0.0, 0.0), (1.5, 0.25), (12.0, 30.5), (np.float32(3.1), np.float32(2.7)), (2e-3, 7.0), (40.0, 40.0)]
    for rl, rr in vals:
        assert V._a_in(rl, rr) == a_in(rl, rr)
