"""F5 stage A: validation report and summary figures (validation split only; no test seed, no held-out speed).

    env -u PYTHONPATH python scripts/f5_stageA_validation.py              # docs/results/f5_stageA_validation.{json,md}
    env -u PYTHONPATH python scripts/f5_stageA_validation.py --figures    # + summary PNGs (not committed)

Inputs (not committed): the finished stage-A runs in runs/f5/A/<arm>/<run>/ (config.json, log.jsonl, DONE and, for the
prereg/19 replicate pair, step_00002000.pt), runs/f5/A/<arm>/lr_selected.json and runs/f5/A/guard.log. Every metric
here is a validation evaluation written during training (data seeds 160-199, training speeds only). Nothing here opens
a record, a test seed (200-239) or a held-out speed (30, 90 deg/s); the test evaluation (final-eval) has not run.
Refuses to overwrite without --overwrite.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train import f5_stageA as A  # noqa: E402

GUARD_LOG = A.OUT / "guard.log"
OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_validation.json"
OUT_MD = ROOT / "docs" / "results" / "f5_stageA_validation.md"
FIG_DIR = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/summary")
TRAINED = ("true", "shuffled", "derange", "mlp", "linear", "bridge_off")
DESCRIPTIVE = "descriptive, validation split, not the pre-registered test"
METRICS = ("loss", "sign_accuracy", "r2")


def _log(run):
    return A.JsonLog(run / "log.jsonl").read()


def evals_of(run):
    ev = [r for r in _log(run) if r.get("event") == "eval"]
    if any(e["set"] != "validation" for e in ev):
        raise SystemExit(f"{run}: an evaluation is not on the validation split")
    return ev


def selected_lrs():
    return {a: json.loads((A.OUT / a / "lr_selected.json").read_text()) for a in TRAINED}


def run_dirs(sel):
    """(arm, seed, run directory) of every registered stage-A run at the selected rate (untrained: its one run)."""
    out = [(a, t, A.run_dir_of(a, t, sel[a]["lr"], A.OUT)) for a in TRAINED for t in A.TRAIN_SEEDS[a]]
    return out + [("untrained", 0, A.run_dir_of("untrained", 0, None, A.OUT))]


def run_row(run):
    cfg, done = json.loads((run / "config.json").read_text()), json.loads((run / "DONE").read_text())
    if cfg["smoke"] or done["smoke"] or done["eval_set"] != "validation":
        raise SystemExit(f"{run}: not a validation run")
    ev = {e["step"]: e for e in evals_of(run)}
    fin, sel = ev[done["final_step"]], ev[done["selected_step"]]
    s2a = done["samples_to_set_accuracy"]
    return dict(final_step=done["final_step"], selected_step=done["selected_step"],
                final_loss=fin["loss"], final_sign_accuracy=fin["sign_accuracy"], final_r2=fin["r2"],
                best_loss=sel["loss"], best_sign_accuracy=sel["sign_accuracy"], best_r2=sel["r2"],
                trials_to_090=s2a["samples"], trials_to_090_censored=s2a["censored"],
                n_val_trials=fin["n_trials"], resumes=sum(r.get("event") == "resume" for r in _log(run)))


COLS = ("final_loss", "best_loss", "final_sign_accuracy", "best_sign_accuracy", "final_r2", "best_r2", "trials_to_090")


def mean_sd(rows):
    out = {}
    for c in COLS:
        x = np.array([r[c] for r in rows], float)
        out[c] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None)
    out["n_seeds"] = len(rows)
    out["n_censored_090"] = int(sum(r["trials_to_090_censored"] for r in rows))
    return out


def replicate_pair(sel):
    """prereg/19 section 1: the true arm at seed 0 and the selected rate, trained twice on the GPU."""
    import torch
    lr = sel["true"]["lr"]
    first, second = A.run_dir_of("true", 0, lr, A.OUT), A.run_dir_of("true", 0, lr, A.OUT, replicate=True)
    if not json.loads((second / "DONE").read_text())["replicate"]:
        raise SystemExit(f"{second} is not marked replicate")
    e1, e2 = {e["step"]: e for e in evals_of(first)}, {e["step"]: e for e in evals_of(second)}
    if e1.keys() != e2.keys():
        raise SystemExit("the replicate pair has different evaluation steps")
    d1, d2 = (json.loads((r / "DONE").read_text()) for r in (first, second))
    diff = {m: [abs(e1[s][m] - e2[s][m]) for s in sorted(e1)] for m in METRICS}
    p1, p2 = (torch.load(r / f"step_{A.STEPS:08d}.pt", weights_only=False, map_location="cpu")["model"]
              for r in (first, second))
    pmax = max(float((p1[k].double() - p2[k].double()).abs().max()) for k in p1 if torch.is_floating_point(p1[k]))
    return dict(lr=lr, runs=[first.name, second.name], n_evaluations=len(e1),
                max_abs_diff_over_evaluations={m: float(max(v)) for m, v in diff.items()},
                selected_steps=[d1["selected_step"], d2["selected_step"]],
                selected_checkpoint={m: [e1[d1["selected_step"]][m], e2[d2["selected_step"]][m]] for m in METRICS},
                max_abs_param_diff_final=pmax)


def guard_summary():
    text = GUARD_LOG.read_text().splitlines()
    pauses = [ln for ln in text if re.search(r" PAUSE #\d+: ", ln)]
    reasons = {}
    for ln in pauses:
        r = re.search(r"PAUSE #\d+: (.+?) [\d.]+ ?[A-Za-z]* >=", ln) or re.search(r"PAUSE #\d+: ([^;]+)", ln)
        reasons[r.group(1)] = reasons.get(r.group(1), 0) + 1
    stops = [ln[20:] for ln in text if "queue stopped" in ln]
    rc = [int(m.group(1)) for ln in text for m in [re.search(r" end rc=(-?\d+) ", ln)] if m]
    return dict(guard_starts=sum(" guard start: " in ln for ln in text), job_ends=len(rc),
                job_ends_rc0=sum(r == 0 for r in rc), job_errors=sum(r != 0 for r in rc), pauses=len(pauses),
                pause_reasons=reasons, queue_stops=stops, queue_done=text[-1].endswith("queue done"),
                first=text[0][:19], last=text[-1][:19])


def build():
    sel = selected_lrs()
    grid = {a: [dict(lr=g["lr"], final_val_loss=g["final_val_loss"]) for g in sel[a]["grid"]] for a in TRAINED}
    runs = {}
    for a, t, run in run_dirs(sel):
        runs.setdefault(a, {})[f"t{t}"] = run_row(run)
    grid_runs = {a: {f"{lr:g}": run_row(A.run_dir_of(a, 0, lr, A.OUT)) for lr in A.LR_GRID} for a in TRAINED}
    summ = {a: mean_sd(list(runs[a].values())) for a in runs}
    seeds = sorted(set(runs["true"]) & set(runs["shuffled"]), key=lambda s: int(s[1:]))
    tms = {s: {c: runs["true"][s][c] - runs["shuffled"][s][c] for c in COLS[:-1]} for s in seeds}
    tms_summary = {c: dict(mean=float(np.mean([tms[s][c] for s in seeds])),
                           sd=float(np.std([tms[s][c] for s in seeds], ddof=1)),
                           n_true_lower_or_higher=None) for c in COLS[:-1]}
    for c in COLS[:-1]:
        v = np.array([tms[s][c] for s in seeds])
        tms_summary[c]["n_true_lower_or_higher"] = dict(negative=int((v < 0).sum()), positive=int((v > 0).sum()))
    n_runs = sum(len(v) for v in runs.values()) + sum(len(A.LR_GRID) - 1 for _ in TRAINED) + 1
    return dict(stage=A.STAGE, split="validation", n_val_trials=240, test_evaluated=False,
                note=("validation evaluations written during training; no test seed (200-239), no held-out speed "
                      "(30, 90 deg/s), no final-eval"),
                n_finished_runs=n_runs, selection_rule=sel["true"]["rule"],
                selected_lr={a: sel[a]["lr"] for a in TRAINED}, lr_grid=grid, lr_grid_runs=grid_runs,
                runs=runs, summary=summ, sd="sample sd over training seeds, ddof = 1",
                trials_to_090_rule=("training trials presented until the validation turn-sign accuracy first reaches "
                                    "0.90; a censored run counts as the 32,000-trial budget"),
                true_minus_shuffled=dict(label=DESCRIPTIVE, per_seed=tms, summary=tms_summary),
                replicate=replicate_pair(sel), guard=guard_summary())


def _f(x, k=4):
    return "n/a" if x is None else f"{x:.{k}f}"


def _ms(d, k=4):
    return f"{d['mean']:.{k}f} ± {_f(d['sd'], k)}" if d["sd"] is not None else f"{d['mean']:.{k}f}"


def summary(d):
    s = d["summary"]
    return ("Stage A, validation split (descriptive; the test evaluation has not run): selected-checkpoint validation "
            "loss, mean ± sd over seeds: " + "; ".join(f"{a} {_ms(s[a]['best_loss'])}" for a in s) + ".")


def render(d):
    L = ["# F5 stage A: validation report", "",
         "Generated by `scripts/f5_stageA_validation.py`; checked by `scripts/verify_report.py`. **Validation split "
         "only** (data seeds 160-199, training speeds, 240 trials): every number is a validation evaluation written "
         "during training. No test seed (200-239) or held-out speed (30, 90 deg/s) was read; `final-eval` has not "
         "run, so nothing here is a pre-registered result.", "", summary(d), "",
         "## 1. Learning-rate selection (training seed 0)", "",
         f"Rule: {d['selection_rule']}; a tie goes to the smaller rate.", "",
         "| arm | " + " | ".join(f"lr {lr:g}" for lr in A.LR_GRID) + " | selected |",
         "|---|" + "---:|" * (len(A.LR_GRID) + 1)]
    for a in TRAINED:
        L.append(f"| {a} | " + " | ".join(_f(g["final_val_loss"]) for g in d["lr_grid"][a])
                 + f" | {d['selected_lr'][a]:g} |")
    L += ["", "Final validation loss (step 2,000) of each seed-0 run.", "",
          "## 2. Runs at the selected rate", "",
          "Final = step 2,000; best = the selected checkpoint (lowest validation loss, steps 50-2,000). Trials to "
          "0.90: training trials presented until the validation turn-sign accuracy first reached 0.90 (`>` = "
          "censored at the budget). `untrained` is one evaluation at initialisation.", ""]
    hdr = ("| arm | seed | final loss | best loss (step) | final sign | best sign | final R2 | best R2 | trials to 0.90 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    L += list(hdr)
    for a, rows in d["runs"].items():
        for s, r in rows.items():
            t90 = f"{'>' if r['trials_to_090_censored'] else ''}{r['trials_to_090']:,}"
            L.append(f"| {a} | {s[1:]} | {_f(r['final_loss'])} | {_f(r['best_loss'])} ({r['selected_step']}) | "
                     f"{_f(r['final_sign_accuracy'], 3)} | {_f(r['best_sign_accuracy'], 3)} | {_f(r['final_r2'])} | "
                     f"{_f(r['best_r2'])} | {t90} |")
    L += ["", f"### Mean ± sd over seeds ({d['sd']})", "",
          "| arm | seeds | final loss | best loss | final sign | best sign | final R2 | best R2 | trials to 0.90 "
          "(censored) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, s in d["summary"].items():
        L.append(f"| {a} | {s['n_seeds']} | {_ms(s['final_loss'])} | {_ms(s['best_loss'])} | "
                 f"{_ms(s['final_sign_accuracy'], 3)} | {_ms(s['best_sign_accuracy'], 3)} | {_ms(s['final_r2'])} | "
                 f"{_ms(s['best_r2'])} | {_ms(s['trials_to_090'], 0)} ({s['n_censored_090']}) |")
    L += ["", f"Trials to 0.90: {d['trials_to_090_rule']}.", "",
          f"## 3. true minus shuffled per seed ({DESCRIPTIVE})", "",
          "| seed | final loss | best loss | final sign | best sign | final R2 | best R2 |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    t = d["true_minus_shuffled"]
    for s, r in t["per_seed"].items():
        L.append(f"| {s[1:]} | " + " | ".join(_f(r[c]) for c in COLS[:-1]) + " |")
    L.append("| mean ± sd | " + " | ".join(_ms(t["summary"][c]) for c in COLS[:-1]) + " |")
    L.append("| seeds < 0 / > 0 | " + " | ".join(
        f"{t['summary'][c]['n_true_lower_or_higher']['negative']} / {t['summary'][c]['n_true_lower_or_higher']['positive']}"
        for c in COLS[:-1]) + " |")
    rp, g = d["replicate"], d["guard"]
    L += ["", "Negative loss differences and positive sign / R2 differences favour `true`. This is "
          f"{DESCRIPTIVE}; the registered comparison (prereg/14 section 8) uses the test split.", "",
          "## 4. GPU run-to-run difference (prereg/19 section 1)", "",
          "Training ran on the GPU, where the cuSPARSE CSR products are not bit-reproducible on the full VNC; runs, "
          "and resumed runs, agree only to float32 rounding (prereg/19). The `true` arm at training seed 0 and the "
          f"selected rate ({rp['lr']:g}) was trained twice (`{rp['runs'][0]}`, `{rp['runs'][1]}`); the second run is "
          "not used for any selection or result. Descriptive, no criterion.", "",
          f"- largest absolute difference over the {rp['n_evaluations']} validation evaluations: loss "
          f"{rp['max_abs_diff_over_evaluations']['loss']:.3e}, turn-sign accuracy "
          f"{rp['max_abs_diff_over_evaluations']['sign_accuracy']:.3e}, R2 {rp['max_abs_diff_over_evaluations']['r2']:.3e};",
          f"- selected steps: {rp['selected_steps'][0]} and {rp['selected_steps'][1]}; selected-checkpoint loss "
          f"{rp['selected_checkpoint']['loss'][0]:.6f} and {rp['selected_checkpoint']['loss'][1]:.6f};",
          f"- largest absolute parameter difference at step {A.STEPS}: {rp['max_abs_param_diff_final']:.3e}.", "",
          "## 5. Guard: pauses, stops and resumes", "",
          f"- {g['guard_starts']} guard starts between {g['first']} and {g['last']}; {g['job_ends']} job ends, "
          f"{g['job_ends_rc0']} with rc 0 and {g['job_errors']} with an error; the log ends with "
          f"\"queue done\": {'yes' if g['queue_done'] else 'no'}.",
          f"- {g['pauses']} pauses: " + ", ".join(f"{k} ({v})" for k, v in g["pause_reasons"].items()) + ".",
          "- queue stops: " + ("; ".join(f"`{s}`" for s in g["queue_stops"]) or "none")
          + (" (the learning-curve preview copy raised EINVAL; fixed in PR #28, the queue resumed from the step-1250 "
             "checkpoint)" if g["queue_stops"] else "") + ".",
          "- runs with a resume from a checkpoint: " + (", ".join(
              f"{a}/{s} ({r['resumes']})" for a, rows in d["runs"].items() for s, r in rows.items() if r["resumes"])
              or "none") + "; seed-0 grid runs: " + (", ".join(
              f"{a} lr {lr} ({r['resumes']})" for a, rows in d["lr_grid_runs"].items() for lr, r in rows.items()
              if r["resumes"]) or "none") + ".", ""]
    return "\n".join(L)


# ── figures (validation only; not committed) ───────────────────────────────────────────────────────────────────────
def figures(d, out=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    sel = d["selected_lr"]
    names = []

    def curve(a, t, lr, replicate=False):
        ev = [e for e in evals_of(A.run_dir_of(a, t, lr, A.OUT, replicate)) if e["step"] > 0 or a == "untrained"]
        return np.array([e["trials_seen"] for e in ev]), {m: np.array([e[m] for e in ev]) for m in METRICS}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        names.append(name)

    # a. learning-rate grid
    fig, ax = plt.subplots(len(TRAINED), 2, figsize=(11, 2.6 * len(TRAINED)))
    for i, a in enumerate(TRAINED):
        for lr in A.LR_GRID:
            x, m = curve(a, 0, lr)
            lab = f"lr {lr:g}" + (" (selected)" if lr == sel[a] else "")
            ax[i, 0].plot(x, m["loss"], "o-", ms=2, lw=1.4 if lr == sel[a] else 0.8, label=lab)
            ax[i, 1].plot(x, m["r2"], "o-", ms=2, lw=1.4 if lr == sel[a] else 0.8, label=lab)
        ax[i, 0].set_yscale("log")
        ax[i, 0].set_ylabel(f"{a}\nvalidation loss")
        ax[i, 1].set_ylabel("validation R2")
        ax[i, 1].set_ylim(-0.2, 1.02)
        ax[i, 0].legend(fontsize=7)
    for x_ in ax[-1]:
        x_.set_xlabel("training trials presented")
    fig.suptitle("F5 stage A: learning-rate grid, training seed 0 (validation split)")
    save(fig, "a_lr_grid.png")

    # b. selected rate, all seeds
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for k, a in enumerate(TRAINED):
        p = ax.flat[k]
        ys = []
        for t in A.TRAIN_SEEDS[a]:
            x, m = curve(a, t, sel[a])
            p.plot(x, m["r2"], lw=0.6, color="0.55")
            ys.append(m["r2"])
        p.plot(x, np.mean(ys, 0), lw=2.2, color="C3", label=f"mean of {len(ys)} seeds")
        p.set_title(f"{a} (lr {sel[a]:g})")
        p.set_ylim(-0.2, 1.02)
        p.set_xlabel("training trials presented")
        p.legend(fontsize=7)
    for p in ax[:, 0]:
        p.set_ylabel("validation R2")
    fig.suptitle("F5 stage A: selected learning rate, every training seed (thin), mean (bold); validation split")
    save(fig, "b_selected_lr_all_seeds.png")

    # c. arm comparison
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, a in enumerate(TRAINED):
        cs = [curve(a, t, sel[a]) for t in A.TRAIN_SEEDS[a]]
        x = cs[0][0]
        for j, mname in enumerate(METRICS):
            y = np.array([c[1][mname] for c in cs])
            mu, sd = y.mean(0), y.std(0, ddof=1)
            ax[j].plot(x, mu, color=f"C{k}", lw=1.4, label=f"{a} (lr {sel[a]:g}, n={len(cs)})")
            ax[j].fill_between(x, mu - sd, mu + sd, color=f"C{k}", alpha=0.18, lw=0)
    u = d["runs"]["untrained"]["t0"]
    for j, key in enumerate(("final_loss", "final_sign_accuracy", "final_r2")):
        ax[j].axhline(u[key], color="k", ls=":", lw=1, label="untrained (initialisation)")
    ax[0].set_yscale("log")
    ax[1].axhline(A.SET_ACCURACY, color="r", lw=0.6, ls="--")
    for p, lab in zip(ax, ("validation loss", "turn-sign accuracy", "R2 (window mean)")):
        p.set_ylabel(lab)
        p.set_xlabel("training trials presented")
    ax[2].set_ylim(-0.3, 1.02)
    ax[0].legend(fontsize=7)
    fig.suptitle("F5 stage A: arms at the selected learning rate, mean ± sd over training seeds (validation split)")
    save(fig, "c_arm_comparison.png")

    # d. true vs shuffled, paired
    fig, p = plt.subplots(figsize=(5.2, 4.6))
    tr, sh = d["runs"]["true"], d["runs"]["shuffled"]
    seeds = list(d["true_minus_shuffled"]["per_seed"])
    for s in seeds:
        p.plot([0, 1], [tr[s]["final_loss"], sh[s]["final_loss"]], "-", color="0.6", lw=0.8)
    p.plot([0] * len(seeds), [tr[s]["final_loss"] for s in seeds], "o", color="C0", label="true")
    p.plot([1] * len(seeds), [sh[s]["final_loss"] for s in seeds], "o", color="C1", label="shuffled")
    p.set_xticks([0, 1], ["true", "shuffled"])
    p.set_xlim(-0.4, 1.4)
    p.set_ylabel("final validation loss (step 2,000)")
    p.set_title(f"true vs shuffled, per training seed\n({DESCRIPTIVE})", fontsize=9)
    save(fig, "d_true_vs_shuffled_final_val_loss.png")
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--figures", action="store_true", help=f"also write the summary PNGs to {FIG_DIR}")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=1) + "\n")
    OUT_MD.write_text(render(d))
    print(summary(d), flush=True)
    if a.figures:
        for n in figures(d):
            print(FIG_DIR / n, flush=True)


if __name__ == "__main__":
    main()
