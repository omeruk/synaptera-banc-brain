"""The training-data tables of the README (Training section): each new formula of check_readme in
scripts/verify_report.py on small hand-made inputs, each one used by docs/readme_numbers.json, the formulas failing
closed (runs that differ, no runs, a report without the key), and the readme_text.py helpers (the decimals of the
learning-rate grid, the figure range of the docs/FIGURES.md intro)."""
import json
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import readme_text as RT  # noqa: E402
import verify_report as V  # noqa: E402

TEST_RUNS = {"true": {"t0": {"n_test_trials": 12}, "t1": {"n_test_trials": 12}}, "mlp": {"t0": {"n_test_trials": 12}}}
STEPS = {"t0": {"selected_step": 50}, "t1": {"selected_step": 400}, "t2": {"selected_step": 100},
         "t3": {"selected_step": 200}}
CASES = {
    "n_test_trials, equal for every run": ([TEST_RUNS], 12),
    "rotation stimuli of TRAIN_CONDITIONS x len(HELD_OUT) x test_seeds": ([7], 2 * 2 * 7),
    "n_test_trials (equal for every run) + rotation stimuli of TRAIN_CONDITIONS x len(HELD_OUT) x test_seeds":
        ([TEST_RUNS, 7], 12 + 2 * 2 * 7),
    "median of selected_step over the runs": ([STEPS], 150),
    "min of selected_step over the runs": ([STEPS], 50),
    "max of selected_step over the runs": ([STEPS], 400),
}


def test_the_held_out_factor_is_two_stimuli_at_two_speeds():
    cond = V._readme_const("train/f5_stageA.py", "TRAIN_CONDITIONS")
    assert sorted({s for s, _ in cond if s.startswith("rot")}) == ["rot_left", "rot_right"]
    assert sorted(V._readme_const("train/f5_stageA.py", "HELD_OUT").values()) == [30, 90]


def test_every_new_formula_is_registered_and_used():
    used = {e["source"].get("formula") for e in json.loads((ROOT / "docs" / "readme_numbers.json").read_text())["entries"]}
    for name in CASES:
        assert name in V.FORMULAS, name
        assert name in used, name


@pytest.mark.parametrize("name", sorted(CASES))
def test_formula(name):
    args, want = CASES[name]
    assert V.FORMULAS[name](args, None) == want


def test_median_of_an_odd_count():
    steps = {k: v for k, v in STEPS.items() if k != "t1"}
    assert V.FORMULAS["median of selected_step over the runs"]([steps], None) == 100


def test_formulas_agree_with_readme_text():
    assert RT._held_out_trials(7) == V.FORMULAS["rotation stimuli of TRAIN_CONDITIONS x len(HELD_OUT) x test_seeds"](
        [7], None)
    assert RT._n_test_trials(TEST_RUNS) == V.FORMULAS["n_test_trials, equal for every run"]([TEST_RUNS], None)


def test_runs_that_differ_fail_closed():
    runs = json.loads(json.dumps(TEST_RUNS))
    runs["mlp"]["t0"]["n_test_trials"] = 11
    with pytest.raises(ValueError):
        V.FORMULAS["n_test_trials, equal for every run"]([runs], None)
    with pytest.raises(SystemExit):
        RT._n_test_trials(runs)


@pytest.mark.parametrize("name", ["median", "min", "max"])
def test_no_runs_fail_closed(name):
    with pytest.raises(ValueError):
        V.FORMULAS[f"{name} of selected_step over the runs"]([{}], None)


@pytest.mark.parametrize("case", ["runs_differ", "missing_key"])
def test_check_readme_reports_a_mismatch(tmp_path, monkeypatch, case):
    runs = json.loads(json.dumps(TEST_RUNS))
    if case == "runs_differ":
        runs["mlp"]["t0"]["n_test_trials"] = 11
    else:
        del runs["mlp"]["t0"]["n_test_trials"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(runs=runs, test_seeds=7)))
    entry = dict(string="12", files=[], kind="result",
                 source=dict(file=str(report), keys=["runs"], formula="n_test_trials, equal for every run", format="{}"))
    nums = tmp_path / "readme_numbers.json"
    nums.write_text(json.dumps(dict(entries=[entry])))
    monkeypatch.setattr(V, "README_NUMBERS", nums)
    _, bad = V.check_readme()
    assert any(b[0] == "readme:result:12" for b in bad), bad


def test_check_readme_passes_on_good_runs(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(dict(runs=TEST_RUNS, test_seeds=7)))
    entry = dict(string="40", files=[], kind="result",
                 source=dict(file=str(report), keys=["runs", "test_seeds"], format="{}",
                             formula="n_test_trials (equal for every run) + rotation stimuli of TRAIN_CONDITIONS x "
                                     "len(HELD_OUT) x test_seeds"))
    nums = tmp_path / "readme_numbers.json"
    nums.write_text(json.dumps(dict(entries=[entry])))
    monkeypatch.setattr(V, "README_NUMBERS", nums)
    _, bad = V.check_readme()
    assert not any(b[0].startswith("readme:result:") for b in bad), bad


def test_lr_decimals():
    assert RT.lr_decimals([0.0899, 0.0600, 0.0520]) == RT.LR_MIN_DECIMALS
    assert RT.lr_decimals([0.27777816, 0.27778373, 0.27780817]) == 6
    with pytest.raises(SystemExit):
        RT.lr_decimals([0.5, 0.5])


def test_bridge_off_needs_more_decimals_in_the_committed_grid():
    vs = json.loads((ROOT / "docs" / "results" / "f5_stageA_validation.json").read_text())
    d = {a: RT.lr_decimals([x["final_val_loss"] for x in g]) for a, g in vs["lr_grid"].items()}
    assert d["bridge_off"] > RT.LR_MIN_DECIMALS and all(v == RT.LR_MIN_DECIMALS for a, v in d.items() if a != "bridge_off")


def test_readme_figures():
    assert RT.readme_figures("*Figure 0 | a.*\n\n**Figure 2 | c.**\n**Figure 1 | b.**\n") == "0-2"
    assert RT.readme_figures("**Figure 1 | a.**\n**Figure 3 | c.**\n") == "1 and 3"
    assert RT.readme_figures("**Figure 4 | a.**\n") == "4"
    assert RT.readme_figures("text with Figure 9 | inside a line\n**Figure 1 | a.**\n") == "1"
    with pytest.raises(SystemExit):
        RT.readme_figures("no captions")


def test_figures_md_intro_matches_the_readme():
    readme = (ROOT / "README.md").read_text()
    gallery = (ROOT / "docs" / "FIGURES.md").read_text()
    assert f"Figures {RT.readme_figures(readme)} are in the [README](../README.md)." in gallery
