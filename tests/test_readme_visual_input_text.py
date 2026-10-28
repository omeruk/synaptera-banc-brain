"""The "What the fly sees" numbers of the README: each new formula of check_readme in scripts/verify_report.py on small
hand-made inputs (every SD the sample SD, ddof = 1), the CSV source of check_readme failing closed (missing file,
missing column, empty selection), each new formula used by docs/readme_numbers.json, and the readme_text.py side
refusing an empty selection."""
import csv
import json
import statistics
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as V  # noqa: E402

SPEEDS = V._readme_const("scripts/f5_stageA_records.py", "SPEEDS")
COLS = ("A_in_rot_left", "A_in_rot_right")
ROWS = [dict(group=f"s{i:03d}", split=sp, speed_deg_s=str(v), phase_deg="0.0", passed="True",
             A_in_rot_left=repr(0.1 * k + 0.01 * i), A_in_rot_right=repr(-0.1 * k - 0.02 * i))
        for k, v in enumerate(SPEEDS, 1) for i, sp in enumerate(("train", "train", "val", "test"))]
for r in ROWS:
    if r["split"] == "test":
        r["A_in_rot_left"] = r["A_in_rot_right"] = ""


def _vals(v, col):
    return [float(r[col]) for r in ROWS if r["split"] in ("train", "val") and int(r["speed_deg_s"]) == v]


def _expected_csv():
    out = {}
    for v in SPEEDS:
        out[f"train + val seed groups at {v} deg/s (count of rows)"] = len(_vals(v, COLS[0]))
        for col in COLS:
            sfx = f"{col} over the train + val rows at {v} deg/s"
            out[f"mean of {sfx}"] = statistics.mean(_vals(v, col))
            out[f"sample SD (ddof = 1) of {sfx}"] = statistics.stdev(_vals(v, col))
        out[f"min of A_in_rot_left over the train + val rows at {v} deg/s"] = min(_vals(v, "A_in_rot_left"))
        out[f"max of A_in_rot_right over the train + val rows at {v} deg/s"] = max(_vals(v, "A_in_rot_right"))
    out["largest sample SD (ddof = 1) of A_in over speeds and directions, train + val rows"] = max(
        statistics.stdev(_vals(v, c)) for v in SPEEDS for c in COLS)
    out["train + val seed groups per speed (equal at every speed)"] = len(_vals(SPEEDS[0], COLS[0]))
    return out


CSV_EXPECTED = _expected_csv()
ASYM = {"rot_left_v060_p0": {"T4a": 0.3, "T4b": -0.5}, "rot_right_v060_p0": {"T4a": -0.28, "T4b": 0.45},
        "rot_left_v120_p0": {"T4a": 0.4, "T4b": -0.6}, "rot_right_v120_p0": {"T4a": -0.41, "T4b": 0.58}}
JSON_CASES = {
    "GATE_SPEEDS[0]": ([(60, 120)], 60),
    "GATE_SPEEDS[1]": ([(60, 120)], 120),
    "SPEEDS[0]": ([(30, 60, 90, 120)], 30),
    "SPEEDS[1]": ([(30, 60, 90, 120)], 60),
    "SPEEDS[2]": ([(30, 60, 90, 120)], 90),
    "SPEEDS[3]": ([(30, 60, 90, 120)], 120),
    "min of SPEEDS": ([(60, 30, 120)], 30),
    "max of SPEEDS": ([(60, 30, 120)], 120),
    "lead-in mean of eye L, equal for every key": ([{"a": {"L": 0.5, "R": 0.7}, "b": {"L": 0.5, "R": 0.7}}], 0.5),
    "lead-in mean of eye R, equal for every key": ([{"a": {"L": 0.5, "R": 0.7}, "b": {"L": 0.5, "R": 0.7}}], 0.7),
    "min over eyes and subtypes": ([{"L": {"T4a": 3.0, "T4b": 9.0}, "R": {"T4a": 2.5, "T4b": 7.0}}], 2.5),
    "max over eyes and subtypes": ([{"L": {"T4a": 3.0, "T4b": 9.0}, "R": {"T4a": 2.5, "T4b": 7.0}}], 9.0),
    "largest |asymmetry(rot_left) + asymmetry(rot_right)| over subtypes and speeds": ([ASYM], abs(-0.5 + 0.45)),
    "largest |asymmetry| over subtypes and keys": ([ASYM], 0.6),
    "fraction x 100 (not rounded)": ([0.0225], 2.25),
}


def test_every_new_formula_is_registered_and_used():
    used = {e["source"].get("formula") for e in json.loads((ROOT / "docs" / "readme_numbers.json").read_text())["entries"]}
    for name in [*CSV_EXPECTED, *JSON_CASES]:
        assert name in V.FORMULAS, name
        assert name in used, name


@pytest.mark.parametrize("name", sorted(CSV_EXPECTED))
def test_csv_formula(name):
    got = V.FORMULAS[name]([ROWS], None)
    assert got == pytest.approx(CSV_EXPECTED[name], rel=1e-12, abs=1e-15)


def test_sd_is_the_sample_sd():
    v, col = SPEEDS[0], COLS[0]
    got = V.FORMULAS[f"sample SD (ddof = 1) of {col} over the train + val rows at {v} deg/s"]([ROWS], None)
    assert got == pytest.approx(statistics.stdev(_vals(v, col))) and got != pytest.approx(statistics.pstdev(_vals(v, col)))


@pytest.mark.parametrize("name", sorted(JSON_CASES))
def test_json_formula(name):
    args, want = JSON_CASES[name]
    assert V.FORMULAS[name](args, None) == pytest.approx(want)


def test_lead_in_formula_refuses_keys_that_differ():
    with pytest.raises(ValueError):
        V.FORMULAS["lead-in mean of eye L, equal for every key"]([{"a": {"L": 0.5}, "b": {"L": 0.6}}], None)


def _entry(path, formula, string):
    return dict(string=string, files=[], kind="data",
                source=dict(file=str(path), keys=["rows"], formula=formula, format="{:.3f}"))


@pytest.mark.parametrize("case", ["missing_file", "missing_column", "empty_selection"])
def test_csv_source_fails_closed(tmp_path, monkeypatch, case):
    v = SPEEDS[1]
    formula = f"mean of A_in_rot_left over the train + val rows at {v} deg/s"
    rows = [dict(r) for r in ROWS]
    cols = list(rows[0])
    if case == "missing_column":
        cols.remove("A_in_rot_left")
        rows = [{k: x for k, x in r.items() if k in cols} for r in rows]
    elif case == "empty_selection":
        rows = [r for r in rows if int(r["speed_deg_s"]) != v]
    path = tmp_path / "input_checks.csv"
    if case != "missing_file":
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    string = "{:.3f}".format(statistics.mean(_vals(v, "A_in_rot_left")))
    nums = tmp_path / "readme_numbers.json"
    nums.write_text(json.dumps(dict(entries=[_entry(path, formula, string)])))
    monkeypatch.setattr(V, "README_NUMBERS", nums)
    n, bad = V.check_readme()
    assert any(b[0] == f"readme:data:{string}" for b in bad), bad


def test_csv_source_passes_on_good_rows(tmp_path, monkeypatch):
    v = SPEEDS[1]
    path = tmp_path / "input_checks.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    string = "{:.3f}".format(statistics.mean(_vals(v, "A_in_rot_left")))
    nums = tmp_path / "readme_numbers.json"
    nums.write_text(json.dumps(dict(entries=[_entry(path, f"mean of A_in_rot_left over the train + val rows at {v} "
                                                          f"deg/s", string)])))
    monkeypatch.setattr(V, "README_NUMBERS", nums)
    n, bad = V.check_readme()
    assert not any(b[0].startswith("readme:data:") for b in bad), bad


def test_readme_text_refuses_an_empty_selection():
    import readme_text as RT
    with pytest.raises(SystemExit):
        RT._ic_sel([r for r in ROWS if int(r["speed_deg_s"]) != SPEEDS[0]], SPEEDS[0])
