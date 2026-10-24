"""scripts/f5_stageA_report.py, scripts/f5_stageA_preflight.py and the verify_report hook, on synthetic test.json
fixtures and tiny synthetic runs in a temporary directory (never the real runs): the exact sign-flip p, Holm, both
guards, the final / provisional rule, the refusals and the pre-flight's model checks."""
import hashlib
import itertools
import json
import math
import sys

import numpy as np
import pytest
import torch

from conftest import ROOT
from test_f5_stageA import synthetic_setup, train, write_world
from train import f5_stageA as A

sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_preflight as P  # noqa: E402
import f5_stageA_report as R  # noqa: E402
import verify_report as VR  # noqa: E402

LR = {"true": 0.01, "shuffled": 0.01, "derange": 0.01, "mlp": 0.003, "linear": 0.001, "bridge_off": 0.001}
N_TEST = len(A.splits()["test"])


@pytest.fixture(autouse=True)
def _deterministic():
    yield
    torch.use_deterministic_algorithms(False)


# ── synthetic test.json fixtures ────────────────────────────────────────────────────────────────────────────────────
def default_r2(arm, t):
    return {"true": 0.95 + 0.002 * t, "shuffled": 0.90 + 0.001 * t, "derange": 0.5, "mlp": 0.99,
            "linear": 0.6, "bridge_off": -0.01, "untrained": -0.5}[arm]


def write_runs(root, r2=default_r2, bridge_off_sign=0.5, samples=None, skip=()):
    """lr_selected.json, DONE, best.pt (random bytes) and test.json of every registered run under root."""
    split_sha = A.split_sha256()
    for a, lr in LR.items():
        (root / a).mkdir(parents=True, exist_ok=True)
        (root / a / "lr_selected.json").write_text(json.dumps(dict(arm=a, lr=lr)))
    reg = [(a, t, LR[a]) for a in LR for t in A.TRAIN_SEEDS[a]] + [("untrained", 0, None)]
    for a, t, lr in reg:
        run = A.run_dir_of(a, t, lr, root)
        run.mkdir(parents=True, exist_ok=True)
        s, c = (samples or {}).get((a, t), (800 * (t + 1), False))
        (run / "DONE").write_text(json.dumps(dict(arm=a, training_seed=t, lr=lr, smoke=False, replicate=False,
                                                  eval_set="validation", selected_step=50 * (t + 1), steps=2000,
                                                  final_step=2000, samples_to_set_accuracy=dict(samples=s,
                                                                                                censored=c))))
        (run / "best.pt").write_bytes(np.random.default_rng([t, len(a)]).bytes(64))
        if (a, t) in skip:
            continue
        sign = bridge_off_sign if a == "bridge_off" else 1.0
        x = r2(a, t)

        def grp(nt, nr, rr, sg):
            return dict(loss=0.01, sign_accuracy=sg, r2=rr, n_trials=nt, n_rotation=nr, records_sha256="0" * 64)
        te = dict(stage="A", arm=a, training_seed=t, lr=lr, selected_step=50 * (t + 1), split_sha256=split_sha,
                  device="cpu", threads=A.TEST_THREADS,
                  best_sha256=hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
                  groups=dict(training_speeds=grp(6 * N_TEST, 4 * N_TEST, x, sign),
                              held_out_primary_90=grp(2 * N_TEST, 2 * N_TEST, x - 0.1, sign),
                              held_out_secondary_30=grp(2 * N_TEST, 2 * N_TEST, x - 0.3, sign)))
        (run / "test.json").write_text(json.dumps(te))
    return root


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "A"
    monkeypatch.setattr(R, "RUNS", root)
    monkeypatch.setattr(R, "EXPLORATORY", tmp_path / "A_exploratory")
    monkeypatch.setattr(R, "OUT_JSON", tmp_path / "out" / "f5_stageA_report.json")
    monkeypatch.setattr(R, "OUT_MD", tmp_path / "out" / "f5_stageA_report.md")
    (tmp_path / "out").mkdir()
    return root


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────────────
def brute_force(d):
    obs = sum(d) / len(d)
    ms = [sum(s * x for s, x in zip(sg, d)) / len(d) for sg in itertools.product((1, -1), repeat=len(d))]
    return sum(m >= obs - 1e-12 for m in ms) / len(ms), sum(abs(m) >= abs(obs) - 1e-12 for m in ms) / len(ms)


def test_sign_flip_known_p():
    d = [0.01 * (k + 1) for k in range(10)]
    t = R.sign_flip(d)
    assert (t["n_patterns"], t["count_greater"], t["p_greater"]) == (1024, 1, 1 / 1024)
    assert t["p_two_sided"] == 2 / 1024
    d[0] = -0.01                                     # flipping the smallest negative one only raises the mean
    assert R.sign_flip(d)["count_greater"] == 2
    d = [0.01, 0.02, 0.03, 0.04, -0.05, 0.06, -0.07, 0.08, 0.09, -0.1]
    t = R.sign_flip(d)
    assert (t["p_greater"], t["p_two_sided"]) == brute_force(d)
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.normal(0.01, 0.03, 10).tolist()
        t = R.sign_flip(d)
        assert (t["p_greater"], t["p_two_sided"]) == brute_force(d)
    assert R.sign_flip([0.0] * 10)["p_greater"] == R.sign_flip([0.0] * 10)["p_two_sided"] == 1.0
    assert R.sign_flip([-0.01 * (k + 1) for k in range(10)])["p_greater"] == 1.0


@pytest.mark.parametrize("p, adj", [
    ({"A": 0.01, "B": 1.0, "C": 1.0}, {"A": 0.03, "B": 1.0, "C": 1.0}),               # B, C not run
    ({"A": 3 / 1024, "B": 1.0, "C": 1.0}, {"A": 9 / 1024, "B": 1.0, "C": 1.0}),
    ({"A": 0.4, "B": 1.0, "C": 1.0}, {"A": 1.0, "B": 1.0, "C": 1.0}),                 # capped at 1
    ({"A": 0.02, "B": 0.01, "C": 0.04}, {"A": 0.04, "B": 0.03, "C": 0.04}),           # step-down maximum
    ({"A": 0.03, "B": 0.011, "C": 0.5}, {"A": 0.06, "B": 0.033, "C": 0.5}),
])
def test_holm(p, adj):
    assert R.holm(p) == pytest.approx(adj, abs=1e-15)


def test_stage_A_holm_with_B_and_C_not_run_is_three_p():
    for k in range(1, 1025):
        assert R.holm({"A": k / 1024, "B": 1.0, "C": 1.0})["A"] == min(1.0, 3 * k / 1024)


def test_ceiling_guard():
    assert R.ceiling_guard([0.985] * 10, [0.99] * 10)["triggered"]
    assert R.ceiling_guard([0.98] * 10, [0.98] * 10)["triggered"]                      # >=
    assert not R.ceiling_guard([0.99] * 10, [0.97] * 10)["triggered"]
    assert not R.ceiling_guard([0.975] * 10, [0.945] * 10)["triggered"]


def test_leakage_guard_binomial_interval():
    for n in (20, 160, 161):
        cdf = list(itertools.accumulate(math.comb(n, k) / 2 ** n for k in range(n + 1)))
        lo, hi = next(k for k, c in enumerate(cdf) if c >= 0.025), next(k for k, c in enumerate(cdf) if c >= 0.975)
        iv = R.binomial_interval(n)
        assert (iv["lo_count"], iv["hi_count"]) == (lo, hi)
    iv = R.binomial_interval(160)
    assert (iv["lo"], iv["hi"]) == (68 / 160, 92 / 160)
    assert not R.leakage_guard([0.5] * 5, 160)["triggered"]
    assert not R.leakage_guard([68 / 160] * 5, 160)["triggered"]                       # on the boundary: inside
    assert not R.leakage_guard([92 / 160] * 5, 160)["triggered"]
    assert R.leakage_guard([67 / 160] * 5, 160)["triggered"]
    assert R.leakage_guard([0.6] * 5, 160)["triggered"]
    assert R.leakage_guard([0.4, 0.4, 0.4, 0.4, 0.4], 160)["triggered"]


def _t(mean_d, p2):
    return dict(mean_d=mean_d, p_two_sided=p2)


NO = dict(triggered=False)
YES = dict(triggered=True)


def test_final_or_provisional():
    v = R.verdict(3 / 1024, _t(0.02, 2 / 1024), NO, NO)
    assert (v["verdict"], v["status"], v["qualifiers"]) == (R.VERDICT_HELPS, "final", [])
    v = R.verdict(0.3, _t(0.01, 0.2), NO, NO)
    assert (v["verdict"], v["status"], v["shuffled_better"]) == (R.VERDICT_NONE, "provisional", False)
    assert R.verdict(0.3, _t(0.01, 0.2), NO, NO, not_run=())["status"] == "final"      # every stage has a p
    assert R.verdict(0.051, _t(0.01, 0.2), NO, NO)["verdict"] == R.VERDICT_NONE
    assert R.verdict(0.0499, _t(0.01, 0.2), NO, NO)["verdict"] == R.VERDICT_HELPS


def test_shuffled_better_ceiling_and_leak():
    v = R.verdict(1.0, _t(-0.02, 2 / 1024), NO, NO)
    assert v["shuffled_better"] and v["qualifiers"] == [R.SHUFFLED_BETTER] and v["status"] == "provisional"
    assert not R.verdict(1.0, _t(-0.02, 0.06), NO, NO)["shuffled_better"]
    assert not R.verdict(1.0, _t(0.02, 0.01), NO, NO)["shuffled_better"]               # same direction: not it
    v = R.verdict(3 / 1024, _t(0.001, 2 / 1024), YES, NO)
    assert v["verdict"] == R.VERDICT_HELPS and v["qualifiers"] == [R.CEILING_NOTE]
    v = R.verdict(3 / 1024, _t(0.02, 2 / 1024), NO, YES)
    assert (v["verdict"], v["status"]) == ("withheld (leak)", "withheld")


def test_censored_runs_rank_largest():
    rt, rs = R.censored_ranks([800, 32000], [False, True], [32000, 1600], [False, False])
    assert rt.tolist() == [1.0, 4.0] and rs.tolist() == [3.0, 2.0]
    rt, rs = R.censored_ranks([32000], [True], [32000], [True])
    assert rt.tolist() == rs.tolist() == [1.5]


def test_bootstrap_interval_is_fixed_by_its_seed():
    d = np.linspace(-0.01, 0.03, 10)
    a, b = R.bootstrap_ci(d), R.bootstrap_ci(d)
    assert a == b and a["lo"] < a["mean"] < a["hi"] and a["mean"] == pytest.approx(d.mean())
    m = d[np.random.default_rng(14).integers(0, 10, (10_000, 10))].mean(1)
    assert (a["lo"], a["hi"]) == tuple(np.percentile(m, [2.5, 97.5]))


# ── the report on fixtures ──────────────────────────────────────────────────────────────────────────────────────────
def test_report_true_better_is_a_final_helps_verdict(runs):
    write_runs(runs)
    d = R.build()
    pr = d["primary"]
    assert d["n_runs"] == 56 and pr["test"]["p_greater"] == 1 / 1024
    assert pr["holm"]["adjusted"] == {"A": 3 / 1024, "B": 1.0, "C": 1.0}
    assert (pr["verdict"], pr["status"], pr["qualifiers"]) == (R.VERDICT_HELPS, "final", [])
    assert pr["d"] == [default_r2("true", t) - default_r2("shuffled", t) for t in range(10)]
    assert not pr["guards"]["ceiling"]["triggered"] and not pr["guards"]["leakage"]["triggered"]
    assert pr["guards"]["leakage"]["interval"]["n"] == 4 * N_TEST
    c = d["secondary"]["comparisons"]
    assert set(c) == {"true_vs_derange_r2", "true_vs_mlp_r2", "true_vs_linear_r2", "true_vs_shuffled_samples",
                      "true_vs_shuffled_g90_sign_accuracy", "true_vs_shuffled_g90_r2",
                      "true_vs_shuffled_g30_sign_accuracy", "true_vs_shuffled_g30_r2"}
    assert c["true_vs_derange_r2"]["test"]["p_greater"] == 1 / 1024
    assert c["true_vs_mlp_r2"]["test"]["p_greater"] == 1.0
    assert d["learning"]["arms"]["true"] == dict(n_seeds=10, n_meeting=10, learned=True)
    assert d["learning"]["arms"]["bridge_off"]["learned"] is None
    assert [x["id"] for x in d["limitations"]] == list("abcde")
    assert d["exploratory"]["run"] is False


def test_report_no_advantage_is_provisional_and_shuffled_better(runs):
    write_runs(runs, r2=lambda a, t: default_r2("true", t) - 0.05 - 0.001 * t if a == "true" else default_r2(a, t))
    pr = R.build()["primary"]
    assert pr["test"]["p_greater"] == 1.0 and pr["test"]["p_two_sided"] == 2 / 1024
    assert (pr["verdict"], pr["status"], pr["shuffled_better"]) == (R.VERDICT_NONE, "provisional", True)


def test_report_mixed_signs_p_and_ceiling(runs):
    d_t = [0.004, 0.003, -0.001, 0.002, 0.005, 0.001, -0.002, 0.006, 0.0005, 0.0015]
    write_runs(runs, r2=lambda a, t: 0.981 + d_t[t] if a == "true" else 0.985 if a == "shuffled" else 0.5)
    pr = R.build()["primary"]
    p1, p2 = brute_force(pr["d"])
    assert (pr["test"]["p_greater"], pr["test"]["p_two_sided"]) == (p1, p2)
    assert pr["holm"]["adjusted"]["A"] == min(1.0, 3 * p1)
    assert pr["guards"]["ceiling"]["triggered"] and R.CEILING_NOTE in pr["qualifiers"]


def test_report_leak_withholds_the_verdict(runs):
    write_runs(runs, bridge_off_sign=0.7)
    pr = R.build()["primary"]
    assert pr["guards"]["leakage"]["triggered"] and pr["verdict"] == "withheld (leak)"
    assert pr["test"]["p_greater"] == 1 / 1024                                         # p is not changed


def test_samples_comparison_ranks_censored_largest(runs):
    samples = {("true", t): (800, False) for t in range(10)}
    samples.update({("shuffled", t): (32000, True) for t in range(10)})
    write_runs(runs, samples=samples)
    c = R.build()["secondary"]["comparisons"]["true_vs_shuffled_samples"]
    assert c["rank_true"] == [5.5] * 10 and c["rank_shuffled"] == [15.5] * 10
    assert c["test"]["p_greater"] == 1 / 1024


def test_refuses_without_every_test_json(runs):
    write_runs(runs, skip={("bridge_off", 3)})
    with pytest.raises(SystemExit, match="1 of 56 registered runs have no test.json"):
        R.build()
    with pytest.raises(SystemExit):
        R.main([])
    assert not R.OUT_JSON.exists() and not R.OUT_MD.exists()


def test_refuses_a_test_json_of_another_checkpoint(runs):
    write_runs(runs)
    (A.run_dir_of("true", 2, 0.01, runs) / "best.pt").write_bytes(b"another checkpoint")
    with pytest.raises(SystemExit, match="does not match"):
        R.build()


def test_never_overwrites_without_flag(runs):
    write_runs(runs)
    R.main([])
    first = R.OUT_MD.read_text()
    assert R.summary(json.loads(R.OUT_JSON.read_text())) in first
    with pytest.raises(SystemExit, match="overwrite"):
        R.main([])
    R.main(["--overwrite"])
    assert R.OUT_MD.read_text() == first


def test_markdown_states_limitations_and_follow_up(runs, tmp_path):
    write_runs(runs)
    d = R.build()
    md = R.render(d)
    for x in d["limitations"]:
        assert f"{x['id']}. {x['text']}" in md
    assert "has not been run." in md and "Status: **final**" in md
    (tmp_path / "A_exploratory" / "true" / "t0_lr0.03").mkdir(parents=True)
    (tmp_path / "A_exploratory" / "true" / "t0_lr0.03" / "DONE").write_text("{}")
    assert R.exploratory_state()["run"] is True


# ── verify_report hook ──────────────────────────────────────────────────────────────────────────────────────────────
def test_verify_hook_on_a_fixture_report(runs, monkeypatch):
    write_runs(runs)
    R.main([])
    monkeypatch.setattr(VR, "F5RP_JSON", R.OUT_JSON)
    monkeypatch.setattr(VR, "F5RP_MD", R.OUT_MD)
    n, bad = VR.check_f5_stageA_report()
    assert n > 10 and bad == []
    R.OUT_MD.write_text(R.OUT_MD.read_text().replace("Status: **final**", "Status: **provisional**"))
    _, bad = VR.check_f5_stageA_report()
    assert {k for k, _ in bad} == {"f5rp:md_equals_render"}
    R.main(["--overwrite"])
    d = json.loads(R.OUT_JSON.read_text())
    d["primary"]["test"]["p_greater"] = 2 / 1024
    R.OUT_JSON.write_text(json.dumps(d))
    _, bad = VR.check_f5_stageA_report()
    assert {"f5rp:json_equals_rebuild", "f5rp:p_one_sided"} <= {k for k, _ in bad}


def test_verify_hook_is_skipped_while_the_report_does_not_exist():
    src = (ROOT / "scripts" / "verify_report.py").read_text()
    assert "    if F5RP_MD.exists():\n        k, b = check_f5_stageA_report()" in src
    assert VR.F5RP_MD == ROOT / "docs" / "results" / "f5_stageA_report.md"


# ── pre-flight on tiny synthetic runs ───────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def world(tmp_path):
    setup = synthetic_setup()
    split = write_world(tmp_path / "rec", setup)
    return setup, tmp_path / "rec", split


def test_preflight_passes_finished_runs_without_reading_data(world, tmp_path):
    setup, _, sp = world
    root, graphs = tmp_path / "A", tmp_path / "graphs"
    reg = []
    for arm in ("true", "shuffled", "mlp", "bridge_off"):
        run = A.run_dir_of(arm, 0, 0.01, root)
        train(world, run, arm=arm, steps=4)
        reg.append((arm, 0, 0.01, run))
    lines = []
    res = P.preflight(setup, graphs, registered=reg, split_path=sp, log=lines.append)
    assert [r["ok"] for r in res] == [True] * 4, [r["problems"] for r in res]
    assert all(ln.startswith("ok ") for ln in lines)
    assert A.open_record.__name__ == "open_record" and not A.in_final_evaluation()


def test_preflight_reports_each_problem(world, tmp_path):
    setup, _, sp = world
    root, graphs = tmp_path / "A", tmp_path / "graphs"
    run = A.run_dir_of("true", 0, 0.01, root)
    train(world, run, steps=4)
    with P.no_data():
        with pytest.raises(A.SealedSplitError):
            A.open_record(next((world[1] / "brain").iterdir()), sp)
        bad = P.check_run("mlp", 0, 0.01, run, setup, graphs, sp)               # another arm's checkpoint
        assert not bad["ok"] and any("DONE is true" in x for x in bad["problems"])
        assert any(x.startswith("model:") for x in bad["problems"])
        assert not P.check_run("true", 0, 0.01, run, setup, graphs)["ok"]       # another split hash
        (run / "test.lock").write_text("x")
        r = P.check_run("true", 0, 0.01, run, setup, graphs, sp)
        assert not r["ok"] and "already run" in r["problems"][0]
        (run / "test.lock").unlink()
        (run / "best.pt").unlink()
        assert P.check_run("true", 0, 0.01, run, setup, graphs, sp)["problems"] == ["best.pt missing"]
