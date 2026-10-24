"""F5 stage A: the pre-registered test report (prereg/14 section 8 with prereg/20).

    env -u PYTHONPATH python scripts/f5_stageA_report.py          # writes docs/results/f5_stageA_report.{json,md}

Written and committed before any stage-A test evaluation; it is not changed after the first `final-eval`.
Inputs (not committed): of every registered stage-A run (each arm at its selected rate, `untrained` t0; never a
seed-0 grid run at another rate, the prereg/19 replicate or a smoke run) runs/f5/A/<arm>/<run>/test.json, written
by `scripts/f5_train.py final-eval`, with DONE (samples to set accuracy, a validation quantity) and the SHA-256 of
best.pt; runs/f5/A/<arm>/lr_selected.json; prereg/20 section 1 (the limitations, quoted). No record is opened.
Refuses to run unless every registered run has test.json; refuses to overwrite without --overwrite.

- Primary (8.2): d_t = test R2(`true`, t) - test R2(`shuffled`, t) at the training speeds, t = 0-9. One-sided exact
  sign-flip permutation test over all 2^10 sign patterns (H1: mean d > 0), the two-sided test for "shuffled
  better", Holm over stages A, B, C with B and C at p = 1, and whether the verdict is final or provisional
  (prereg/20 section 3).
- Effect size: mean d with a 95 % percentile bootstrap interval over training seeds (`default_rng(14)`, 10,000
  resamples of the seeds with replacement).
- Guards: ceiling (both arms mean R2 >= 0.98); leakage (`bridge_off` mean test sign accuracy outside the
  equal-tailed binomial(n, 0.5) 95 % interval, n = the test rotation trials of one run).
- Secondary (8.3, no Holm, never a verdict): the same test for `true` against `derange`, `mlp`, `linear` on R2, and
  against `shuffled` on samples to set accuracy (pooled ranks, censored runs ranked largest) and on generalisation
  (sign accuracy and R2 at 90 and 30 deg/s).
- Every 8.1 metric per arm and seed, the 8.4 learning rule, the five prereg/20 limitations and the state of the
  prereg/20 exploratory follow-up.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train import f5_stageA as A  # noqa: E402

RUNS = A.OUT
SPLIT_PATH = A.SPLIT_FILE
PREREG20 = ROOT / "prereg" / "20_f5_stageA_limitations_exploratory.md"
EXPLORATORY = A.OUT.parent / "A_exploratory"          # prereg/20 section 2; never read by this report
OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_report.json"
OUT_MD = ROOT / "docs" / "results" / "f5_stageA_report.md"

# ── registered rules (prereg/14 section 8, prereg/20 section 3) ────────────────────────────────────────────────────
TRAINED = ("true", "shuffled", "derange", "mlp", "linear", "bridge_off")
STAGES = ("A", "B", "C")
NOT_RUN = ("B", "C")                 # p = 1 (prereg/14 8.2); a "no advantage" verdict is provisional (prereg/20 s. 3)
ALPHA = 0.05
CEILING_R2 = 0.98
BOOT_SEED, N_BOOT, CI = 14, 10_000, 0.95
LEAK_LEVEL = 0.95
LEARNED_SIGN, LEARNED_MIN_SEEDS = 0.9, 8
GROUPS = {"train": "training_speeds", "g90": "held_out_primary_90", "g30": "held_out_secondary_30"}
VERDICT_HELPS = "The connectome helps in stage A"
VERDICT_NONE = "No advantage in stage A"
SHUFFLED_BETTER = "shuffled better (descriptive)"
CEILING_NOTE = "ceiling: the task does not separate the wirings"
METRICS = ("r2", "sign_accuracy", "loss", "g90_sign_accuracy", "g90_r2", "g30_sign_accuracy", "g30_r2",
           "samples_to_set_accuracy")


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────────────
def sign_flip(d) -> dict:
    """Exact paired sign-flip permutation test of mean d over all 2^n sign patterns (the observed one included).
    p_greater: H1 mean d > 0; p_two_sided: |mean| at least as large. Ties within 1e-12 of the scale of d count."""
    d = np.asarray(d, np.float64)
    n = len(d)
    signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = signs @ d / n
    obs = float(d.mean())
    tol = 1e-12 * max(float(np.abs(d).mean()), np.finfo(float).tiny)
    k1, k2 = int((means >= obs - tol).sum()), int((np.abs(means) >= abs(obs) - tol).sum())
    return dict(n=n, n_patterns=2 ** n, mean_d=obs, count_greater=k1, p_greater=k1 / 2 ** n, count_two_sided=k2,
                p_two_sided=k2 / 2 ** n)


def holm(p: dict) -> dict:
    """Holm step-down adjusted p values of a family {name: p}."""
    names = sorted(p, key=lambda k: (p[k], k))
    m, out, run = len(names), {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def bootstrap_ci(d, seed=BOOT_SEED, n_boot=N_BOOT, level=CI) -> dict:
    """Mean of d with a percentile bootstrap interval over the seeds (resampled with replacement)."""
    d = np.asarray(d, np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), level=level, n_boot=n_boot, rng_seed=seed)


def binomial_interval(n, level=LEAK_LEVEL) -> dict:
    """Equal-tailed interval of binomial(n, 0.5): the (1 - level) / 2 and (1 + level) / 2 quantiles, as counts and as
    fractions of n."""
    from scipy.stats import binom
    lo, hi = (int(binom.ppf(q, n, 0.5)) for q in ((1 - level) / 2, (1 + level) / 2))
    return dict(n=int(n), level=level, lo_count=lo, hi_count=hi, lo=lo / n, hi=hi / n)


def ceiling_guard(r2_true, r2_shuffled) -> dict:
    mt, ms = float(np.mean(r2_true)), float(np.mean(r2_shuffled))
    return dict(threshold=CEILING_R2, mean_r2_true=mt, mean_r2_shuffled=ms,
                triggered=bool(mt >= CEILING_R2 and ms >= CEILING_R2))


def leakage_guard(sign_acc, n_rotation) -> dict:
    m = float(np.mean(sign_acc))
    iv = binomial_interval(n_rotation)
    return dict(mean_sign_accuracy=m, n_seeds=len(sign_acc), interval=iv, triggered=bool(m < iv["lo"] or m > iv["hi"]))


def verdict(p_holm, test, ceiling, leak, not_run=NOT_RUN) -> dict:
    """prereg/14 section 8.2 verdict with its guards; status per prereg/20 section 3 (a "helps" verdict is final, a
    "no advantage" verdict is provisional while any other stage of the Holm family has not been run)."""
    if leak["triggered"]:
        return dict(verdict="withheld (leak)", status="withheld", shuffled_better=None, qualifiers=["leak"])
    helps = p_holm < ALPHA
    better = (not helps) and test["mean_d"] < 0 and test["p_two_sided"] < ALPHA
    q = ([SHUFFLED_BETTER] if better else []) + ([CEILING_NOTE] if ceiling["triggered"] else [])
    status = "final" if helps or not not_run else "provisional"
    return dict(verdict=VERDICT_HELPS if helps else VERDICT_NONE, status=status, shuffled_better=bool(better),
                qualifiers=q)


def censored_ranks(a_val, a_cens, b_val, b_cens):
    """Ranks (1 = fewest samples) of the pooled values of two arms; censored runs are ranked above every uncensored
    one (ties share the mean rank)."""
    from scipy.stats import rankdata
    key = np.array([np.inf if c else float(v) for v, c in zip(list(a_val) + list(b_val), list(a_cens) + list(b_cens))])
    r = rankdata(key)
    return r[:len(a_val)], r[len(a_val):]


def paired(d, label) -> dict:
    d = [float(x) for x in d]
    return dict(label=label, d=d, test=sign_flip(d), effect=bootstrap_ci(d))


# ── inputs ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selected_lrs(runs=None):
    runs = Path(runs or RUNS)
    return {a: json.loads((runs / a / "lr_selected.json").read_text())["lr"] for a in TRAINED}


def registered(runs=None):
    """(arm, seed, lr, run directory) of every registered stage-A run (prereg/14 sections 5-6, prereg/19)."""
    runs = Path(runs or RUNS)
    sel = selected_lrs(runs)
    out = [(a, t, sel[a], A.run_dir_of(a, t, sel[a], runs)) for a in TRAINED for t in A.TRAIN_SEEDS[a]]
    return out + [("untrained", 0, None, A.run_dir_of("untrained", 0, None, runs))]


def run_row(arm, t, lr, run, split_sha, n_test_seeds, runs=None) -> dict:
    """The 8.1 metrics of one run from its test.json (DONE: samples to set accuracy), after consistency checks."""
    done, te = json.loads((run / "DONE").read_text()), json.loads((run / "test.json").read_text())
    if done.get("smoke") or done["eval_set"] != "validation" or done.get("replicate"):
        raise SystemExit(f"{run}: not a registered validation run")
    want = dict(stage=A.STAGE, arm=arm, training_seed=t, lr=lr, selected_step=done["selected_step"],
                split_sha256=split_sha, device="cpu", threads=A.TEST_THREADS, best_sha256=_sha(run / "best.pt"))
    bad = {k: (te.get(k), v) for k, v in want.items() if te.get(k) != v}
    if bad or set(te["groups"]) != set(GROUPS.values()):
        raise SystemExit(f"{run}/test.json does not match the run: {bad or sorted(te['groups'])}")
    g = {k: te["groups"][v] for k, v in GROUPS.items()}
    exp = dict(train=(6 * n_test_seeds, 4 * n_test_seeds), g90=(2 * n_test_seeds,) * 2, g30=(2 * n_test_seeds,) * 2)
    for k, (nt, nr) in exp.items():
        if (g[k]["n_trials"], g[k]["n_rotation"]) != (nt, nr):
            raise SystemExit(f"{run}/test.json: {GROUPS[k]} has {g[k]['n_trials']} trials, expected {nt}")
    s2a = done["samples_to_set_accuracy"]
    return dict(r2=g["train"]["r2"], sign_accuracy=g["train"]["sign_accuracy"], loss=g["train"]["loss"],
                g90_sign_accuracy=g["g90"]["sign_accuracy"], g90_r2=g["g90"]["r2"],
                g30_sign_accuracy=g["g30"]["sign_accuracy"], g30_r2=g["g30"]["r2"],
                samples_to_set_accuracy=s2a["samples"], samples_censored=s2a["censored"],
                n_test_trials=g["train"]["n_trials"], n_test_rotation=g["train"]["n_rotation"],
                selected_step=done["selected_step"], lr=lr, run=str(run.relative_to(runs or RUNS)),
                test_json_sha256=_sha(run / "test.json"), best_sha256=want["best_sha256"])


def limitations(path=None) -> list:
    """The five limitations of prereg/20 section 1, quoted (one line each)."""
    text = Path(path or PREREG20).read_text()
    sec = text.split("## 1. ", 1)[1].split("\n## ", 1)[0]
    items = re.findall(r"^([a-e])\. (.+?)(?=^[a-e]\. |^Every stage-A report|\Z)", sec, re.S | re.M)
    out = [dict(id=k, text=" ".join(v.split())) for k, v in items]
    if [x["id"] for x in out] != list("abcde"):
        raise SystemExit(f"{path or PREREG20}: expected limitations a-e in section 1, found {[x['id'] for x in out]}")
    return out


def exploratory_state(path=None) -> dict:
    p = Path(path or EXPLORATORY)
    ran = p.exists() and any(p.glob("**/DONE"))
    return dict(run=bool(ran), location=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                statement=("The prereg/20 exploratory follow-up (true and shuffled, training seeds 0-4; lr 0.03 for "
                           "2,000 steps and lr 0.01 for 6,000 steps) "
                           + ("has runs; it is exploratory, never part of a verdict, and reported separately."
                              if ran else "has not been run.")))


# ── report ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _mean_sd(x):
    x = np.asarray(x, np.float64)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None, n=len(x))


def build(runs=None, prereg20=None, exploratory=None, split_path=None) -> dict:
    runs = Path(runs or RUNS)
    split_path = Path(split_path or SPLIT_PATH)
    reg = registered(runs)
    missing = [str(r) for *_, r in reg if not (r / "test.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(reg)} registered runs have no test.json (first: {missing[0]}); "
                         "the report runs only after every final-eval")
    split_sha, n_test = A.split_sha256(split_path), len(A.splits(split_path)["test"])
    rows = {}
    for a, t, lr, r in reg:
        rows.setdefault(a, {})[f"t{t}"] = run_row(a, t, lr, r, split_sha, n_test, runs)
    seeds = [f"t{t}" for t in A.TRAIN_SEEDS["true"]]
    tr, sh = rows["true"], rows["shuffled"]

    # primary (8.2)
    d = [tr[s]["r2"] - sh[s]["r2"] for s in seeds]
    test = sign_flip(d)
    p = {"A": test["p_greater"], **{x: 1.0 for x in NOT_RUN}}
    adj = holm(p)
    ceil = ceiling_guard([tr[s]["r2"] for s in seeds], [sh[s]["r2"] for s in seeds])
    bo = rows["bridge_off"]
    nrot = {r["n_test_rotation"] for r in bo.values()}
    if len(nrot) != 1:
        raise SystemExit(f"bridge_off runs differ in their number of test rotation trials: {sorted(nrot)}")
    leak = leakage_guard([r["sign_accuracy"] for r in bo.values()], nrot.pop())
    primary = dict(metric="test R2 at the training speeds (window mean, all trials)", d=d, seeds=seeds, test=test,
                   holm=dict(family=list(STAGES), p=p, adjusted=adj, not_run=list(NOT_RUN)),
                   effect=bootstrap_ci(d), guards=dict(ceiling=ceil, leakage=leak),
                   **verdict(adj["A"], test, ceil, leak))

    # secondary (8.3)
    sec = {f"true_vs_{b}_r2": paired([tr[s]["r2"] - rows[b][s]["r2"] for s in seeds], f"R2(true) - R2({b})")
           for b in ("derange", "mlp", "linear")}
    rt, rs = censored_ranks([tr[s]["samples_to_set_accuracy"] for s in seeds],
                            [tr[s]["samples_censored"] for s in seeds],
                            [sh[s]["samples_to_set_accuracy"] for s in seeds],
                            [sh[s]["samples_censored"] for s in seeds])
    sec["true_vs_shuffled_samples"] = paired(rs - rt, "rank(shuffled) - rank(true) of samples to set accuracy "
                                                      "(pooled ranks, censored largest; > 0 favours true)")
    sec["true_vs_shuffled_samples"].update(rank_true=rt.tolist(), rank_shuffled=rs.tolist())
    for g, deg in (("g90", 90), ("g30", 30)):
        for m in ("sign_accuracy", "r2"):
            sec[f"true_vs_shuffled_{g}_{m}"] = paired([tr[s][f"{g}_{m}"] - sh[s][f"{g}_{m}"] for s in seeds],
                                                      f"{m}(true) - {m}(shuffled) at {deg} deg/s")

    # 8.1 summaries and 8.4 learning
    summ = {a: {m: _mean_sd([r[m] for r in rs_.values()]) for m in METRICS} for a, rs_ in rows.items()}
    for a, rs_ in rows.items():
        summ[a]["n_samples_censored"] = sum(r["samples_censored"] for r in rs_.values())
    learned = {}
    for a, rs_ in rows.items():
        k = sum(r["sign_accuracy"] >= LEARNED_SIGN and r["r2"] > 0 for r in rs_.values())
        learned[a] = dict(n_seeds=len(rs_), n_meeting=int(k),
                          learned=bool(k >= LEARNED_MIN_SEEDS) if len(rs_) == 10 else None)
    return dict(stage=A.STAGE, split="test", test_seeds=n_test, split_sha256=split_sha,
                note=("pre-registered stage-A test report (prereg/14 section 8 with prereg/20); every value from "
                      "the test.json of each registered run, written once by final-eval on the CPU from best.pt"),
                selected_lr=selected_lrs(runs), n_runs=len(reg), runs=rows, summary=summ, primary=primary,
                secondary=dict(note="prereg/14 section 8.3: same test, no Holm, never a verdict", comparisons=sec),
                learning=dict(rule=f"at least {LEARNED_MIN_SEEDS} of 10 training seeds with test sign accuracy >= "
                                   f"{LEARNED_SIGN} and R2 > 0 (prereg/14 section 8.4; None: the arm has not 10 seeds)",
                              arms=learned),
                not_applicable_in_A=["g", "stability", "drift", "imitation R2"],
                limitations=limitations(prereg20), exploratory=exploratory_state(exploratory))


def _f(x, k=4):
    return "n/a" if x is None else f"{x:.{k}f}"


def _ms(d, k=4):
    return f"{d['mean']:.{k}f} ± {_f(d['sd'], k)}" if d["sd"] is not None else f"{d['mean']:.{k}f}"


def _p(t, key="greater"):
    return f"{t[f'p_{key}']:.4f} ({t[f'count_{key}']}/{t['n_patterns']})"


def summary(d):
    pr = d["primary"]
    t, e = pr["test"], pr["effect"]
    q = "; ".join(pr["qualifiers"])
    return (f"Stage A (test split): mean d = R2(true) - R2(shuffled) {e['mean']:.4f} (95 % bootstrap interval "
            f"{e['lo']:.4f} to {e['hi']:.4f}, {t['n']} seeds); one-sided exact p {_p(t)}, Holm-adjusted "
            f"{pr['holm']['adjusted']['A']:.4f}. Verdict: {pr['verdict']} ({pr['status']})" + (f"; {q}" if q else "")
            + ".")


def render(d):
    pr = d["primary"]
    t, e, g = pr["test"], pr["effect"], pr["guards"]
    L = ["# F5 stage A: pre-registered test report", "",
         "Generated by `scripts/f5_stageA_report.py`; checked by `scripts/verify_report.py`. prereg/14 section 8 "
         "with prereg/20. Every metric is from the test split (data seeds 200-239) at the selected checkpoint of "
         "each registered run, evaluated once on the CPU by `final-eval`.", "", summary(d), "",
         "## 1. Primary test (prereg/14 section 8.2)", "",
         f"d_t = test R2(`true`, t) - test R2(`shuffled`, t) at the training speeds, t = 0-{t['n'] - 1}.", "",
         "| seed | R2 true | R2 shuffled | d |", "|---:|---:|---:|---:|"]
    for s, x in zip(pr["seeds"], pr["d"]):
        L.append(f"| {s[1:]} | {_f(d['runs']['true'][s]['r2'])} | {_f(d['runs']['shuffled'][s]['r2'])} | {_f(x)} |")
    h = pr["holm"]
    L += ["", f"- One-sided exact sign-flip permutation test (all {t['n_patterns']} sign patterns, H1: mean d > 0): "
          f"p = {_p(t)}.",
          f"- Two-sided test: p = {_p(t, 'two_sided')}; \"shuffled better\" needs mean d < 0 and p < {ALPHA}: "
          f"{'yes' if pr['shuffled_better'] else 'no' if pr['shuffled_better'] is not None else 'n/a (withheld)'}.",
          f"- Holm over stages {', '.join(h['family'])}; stages not run ({', '.join(h['not_run'])}) have p = 1. "
          "Adjusted p: " + ", ".join(f"{k} {h['adjusted'][k]:.4f}" for k in h["family"]) + ".",
          f"- Effect size: mean d {e['mean']:.4f}, 95 % percentile bootstrap interval {e['lo']:.4f} to {e['hi']:.4f} "
          f"({e['n_boot']:,} resamples of the seeds, `default_rng({e['rng_seed']})`).",
          f"- Ceiling guard: mean R2 `true` {g['ceiling']['mean_r2_true']:.4f}, `shuffled` "
          f"{g['ceiling']['mean_r2_shuffled']:.4f}; both >= {g['ceiling']['threshold']}: "
          f"{'yes' if g['ceiling']['triggered'] else 'no'}.",
          f"- Leakage guard: `bridge_off` mean test sign accuracy {g['leakage']['mean_sign_accuracy']:.4f} over "
          f"{g['leakage']['n_seeds']} seeds; binomial(n = {g['leakage']['interval']['n']}, 0.5) 95 % interval "
          f"{g['leakage']['interval']['lo']:.4f} to {g['leakage']['interval']['hi']:.4f} "
          f"({g['leakage']['interval']['lo_count']} to {g['leakage']['interval']['hi_count']} of "
          f"{g['leakage']['interval']['n']} test rotation trials of one run); outside: "
          f"{'yes (leak)' if g['leakage']['triggered'] else 'no'}.", "",
          f"**Verdict: {pr['verdict']}**" + (f" ({'; '.join(pr['qualifiers'])})" if pr["qualifiers"] else "")
          + f". Status: **{pr['status']}**. "
          + {"final": "A \"helps\" verdict reported before B and C are run is final (prereg/20 section 3)."
             if pr["verdict"] == VERDICT_HELPS else "Every stage of the Holm family has a p value.",
             "provisional": "A \"no advantage\" verdict reported before B and C are run is provisional; it is "
                            "recomputed once B and C have p values or are declared not run (prereg/20 section 3).",
             "withheld": "The leakage guard withholds the verdict (prereg/14 section 8.2)."}[pr["status"]], "",
          "## 2. Limitations recorded before unsealing (prereg/20 section 1)", ""]
    L += [f"{x['id']}. {x['text']}" for x in d["limitations"]]
    L += ["", "## 3. Secondary comparisons (prereg/14 section 8.3; no Holm, never a verdict)", "",
          "| comparison | mean d | 95 % bootstrap interval | one-sided p | two-sided p |", "|---|---:|---:|---:|---:|"]
    for c in d["secondary"]["comparisons"].values():
        L.append(f"| {c['label']} | {c['effect']['mean']:.4f} | {c['effect']['lo']:.4f} to {c['effect']['hi']:.4f} | "
                 f"{_p(c['test'])} | {_p(c['test'], 'two_sided')} |")
    L += ["", "## 4. Metrics per arm and seed (prereg/14 section 8.1)", "",
          "Test split at the selected checkpoint; R2 and sign at the training speeds (240 trials, sign over the 160 "
          "rotation trials); 90 and 30 deg/s: rot_left and rot_right only. Samples to set accuracy: training trials "
          "until the validation turn-sign accuracy first reached 0.90 (`>` = censored at the budget). g, stability, "
          "drift and imitation R2 are stage-C metrics.", "",
          "| arm | seed | lr | R2 | sign | loss | 90 sign | 90 R2 | 30 sign | 30 R2 | samples to 0.90 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, rows in d["runs"].items():
        for s, r in rows.items():
            lr = "-" if r["lr"] is None else f"{r['lr']:g}"
            L.append(f"| {a} | {s[1:]} | {lr} | {_f(r['r2'])} | {_f(r['sign_accuracy'], 3)} | {_f(r['loss'])} | "
                     f"{_f(r['g90_sign_accuracy'], 3)} | {_f(r['g90_r2'])} | {_f(r['g30_sign_accuracy'], 3)} | "
                     f"{_f(r['g30_r2'])} | {'>' if r['samples_censored'] else ''}{r['samples_to_set_accuracy']:,} |")
    L += ["", "### Mean ± sd over seeds (sample sd, ddof = 1)", "",
          "| arm | seeds | R2 | sign | loss | 90 sign | 90 R2 | 30 sign | 30 R2 | samples to 0.90 (censored) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, s in d["summary"].items():
        L.append(f"| {a} | {s['r2']['n']} | {_ms(s['r2'])} | {_ms(s['sign_accuracy'], 3)} | {_ms(s['loss'])} | "
                 f"{_ms(s['g90_sign_accuracy'], 3)} | {_ms(s['g90_r2'])} | {_ms(s['g30_sign_accuracy'], 3)} | "
                 f"{_ms(s['g30_r2'])} | {_ms(s['samples_to_set_accuracy'], 0)} ({s['n_samples_censored']}) |")
    L += ["", "## 5. Learning (prereg/14 section 8.4, descriptive)", "", f"Rule: {d['learning']['rule']}.", "",
          "| arm | seeds | seeds meeting the rule | learned |", "|---|---:|---:|---|"]
    for a, x in d["learning"]["arms"].items():
        L.append(f"| {a} | {x['n_seeds']} | {x['n_meeting']} | "
                 f"{'n/a' if x['learned'] is None else 'yes' if x['learned'] else 'no'} |")
    L += ["", "`untrained` is the reference point (prereg/14 section 8.4).", "",
          "## 6. Exploratory follow-up (prereg/20 section 2)", "", d["exploratory"]["statement"], ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=1) + "\n")
    OUT_MD.write_text(render(d))
    print(summary(d), flush=True)


if __name__ == "__main__":
    main()
