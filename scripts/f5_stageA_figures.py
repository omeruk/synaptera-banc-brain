"""F5 stage A: summary figures of the pre-registered test report, for viewing only (PNGs are never committed).

    env -u PYTHONPATH python scripts/f5_stageA_figures.py [--out DIR]     # default: the Windows preview test_summary

Reads only docs/results/f5_stageA_report.json and computes nothing new; every plotted value is a stored per-seed
value or a stored mean, and the titles quote the stored verdict, p, Holm p and effect interval. Writes:
- a_true_vs_shuffled_paired_r2.png: `true` against `shuffled` test R2 per seed (paired), verdict in the title;
- b_test_r2_per_arm.png: test R2 per arm, every seed a dot and the mean a bar;
- c_generalisation_true_vs_shuffled.png: `true` and `shuffled` sign accuracy and R2 at 30, 60/120 (training) and
  90 deg/s.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "results" / "f5_stageA_report.json"
OUT = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/test_summary")     # machine 2: viewing in Windows
FILES = ("a_true_vs_shuffled_paired_r2.png", "b_test_r2_per_arm.png", "c_generalisation_true_vs_shuffled.png")

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUE, SHUF, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8b2"
STYLE = {"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
         "grid.linewidth": 0.8, "axes.axisbelow": True}
CONDS = (("30 deg/s\n(held out)", "g30_"), ("60/120 deg/s\n(training)", ""), ("90 deg/s\n(held out)", "g90_"))


def seeds(runs, arm, key):
    return np.array([runs[arm][s][key] for s in sorted(runs[arm], key=lambda s: int(s[1:]))])


def paired(d, out):
    runs, prim = d["runs"], d["primary"]
    t, sh = seeds(runs, "true", "r2"), seeds(runs, "shuffled", "r2")
    fig, ax = plt.subplots(figsize=(6, 5))
    for a, b in zip(sh, t):
        ax.plot([0, 1], [a, b], color=NEUTRAL, lw=1.2, zorder=1)
    ax.scatter(np.zeros(len(sh)), sh, s=46, color=SHUF, edgecolor=SURF, lw=1.5, zorder=2, label="shuffled")
    ax.scatter(np.ones(len(t)), t, s=46, color=TRUE, edgecolor=SURF, lw=1.5, zorder=2, label="true")
    ax.set_xticks([0, 1], ["shuffled", "true"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("test R2 at the training speeds")
    h, e, tst = prim["holm"]["adjusted"]["A"], prim["effect"], prim["test"]
    ax.set_title(f"{prim['verdict']} ({prim['status']})\n"
                 f"one-sided p = {tst['p_greater']:.4f}, Holm p = {h:.4f}; "
                 f"mean d = {e['mean']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / FILES[0], dpi=160)
    plt.close(fig)


def per_arm(d, out):
    runs, summ = d["runs"], d["summary"]
    rng = np.random.default_rng(0)                       # horizontal jitter only, for legibility
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, arm in enumerate(runs):
        v, m = seeds(runs, arm, "r2"), summ[arm]["r2"]["mean"]
        col = TRUE if arm == "true" else SHUF if arm == "shuffled" else INK2
        ax.bar(i, m, width=0.62, color=col, alpha=0.22, zorder=1)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=22, color=col, edgecolor=SURF, lw=0.8, zorder=2)
        ax.text(i, max(m, 0) + 0.03, f"{round(m, 3) + 0.0:.3f}",            # + 0.0: no "-0.000" label
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(runs)), [f"{a}\n(n={len(runs[a])})" for a in runs])
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.12)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("test R2 at the training speeds")
    ax.set_title("Stage A test R2 per arm (dots: seeds; bar: mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / FILES[1], dpi=160)
    plt.close(fig)


def generalisation(d, out):
    runs, summ = d["runs"], d["summary"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, lab in zip(axes, ("sign_accuracy", "r2"), ("test turn-sign accuracy", "test R2")):
        for arm, col, off in (("shuffled", SHUF, -0.08), ("true", TRUE, 0.08)):
            for j, (_, pre) in enumerate(CONDS):
                v = seeds(runs, arm, pre + metric)
                ax.scatter(j + off + rng.uniform(-0.03, 0.03, len(v)), v, s=16, color=col, alpha=0.55,
                           edgecolor="none", zorder=2)
            ax.plot(np.arange(len(CONDS)) + off, [summ[arm][pre + metric]["mean"] for _, pre in CONDS], color=col,
                    lw=2, marker="o", ms=8, mec=SURF, mew=1.5, zorder=3, label=arm)
        ax.set_xticks(range(len(CONDS)), [c for c, _ in CONDS])
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10)
        if metric == "sign_accuracy":
            ax.set_ylim(0.9, 1.01)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Generalisation across drum speeds: true vs shuffled (dots: seeds; line: mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / FILES[2], dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    d = json.loads(Path(a.report).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        paired(d, out)
        per_arm(d, out)
        generalisation(d, out)
    for f in FILES:
        print(out / f, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
