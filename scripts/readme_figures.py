"""Figures 0, 1, 5 and 6 of the README: the pipeline schematic, the pre-registered stage-A result on the test split (no
post-hoc data; that is Figure 4), the training on the validation split and the FlyWire -> BANC alignment of the figures.

    env -u PYTHONPATH python scripts/readme_figures.py [--out DIR] [--overwrite]

Writes fig0_pipeline.png, fig1_stageA_result.png, fig5_training.png and fig6_alignment.png (each 180 mm wide, exactly
2000 px, at most 1.5 MB) to --out (default docs/figures/). Without --overwrite nothing is written if any of them exists (rule 13); --out
inside the local preview folder is refused.

Figure 0 (no data): six boxes in two rows, stimulus -> FlyGym eyes and FlyVis -> FlyWire brain model -> DN bridge ->
BANC VNC subgraph -> fixed readout, coloured by dataset (FlyWire, fly 1; BANC, fly 2); 9 pt text, no numbers.

Figure 1 reads only docs/results/f5_stageA_report.json and computes nothing new: every point is a stored per-run value
(runs[arm][t]["r2" | "g90_r2" | "g30_r2"]) and every bar or mean marker is the mean of those values.
- a: test R2 at the training speeds, true against shuffled per training seed (paired), means as short bars;
- b: test R2 at the training speeds of the five trained comparison arms, one dot per seed, means as short bars
  (bridge_off and untrained lie far below this axis and are shown in c);
- c: test R2 per condition (training speeds, 90 deg/s held out, 30 deg/s held out) for every arm in the JSON.

Figure 5 reads only docs/results/f5_stageA_curves.json and the lr_grid of docs/results/f5_stageA_validation.json
(validation split; every value a stored evaluation, every line or band a mean / SD of stored values):
- a: the learning-rate grid at training seed 0, final validation loss per arm and rate, the selected rate marked;
- b, c: validation loss and R2 over training trials presented at the selected rate, mean +- 1 SD over training seeds;
- d: training trials until the validation turn-sign accuracy first reached the set accuracy, per training seed.

Figure 6 reads only docs/figures/landmarks.csv, transforms.json, warp_check.csv and warp_check.json (the alignment used
by the figures; it does not enter the model):
- a: the landmarks in BANC coordinates on the two axes with the largest spread of the BANC landmarks: BANC landmark
  points and the affine-transformed FlyWire points, a line per landmark for its residual;
- b: residual norms of the affine and the similarity, control and check points;
- c: our affine against the BANC team's warp, an ECDF over the used DNs with the median marked;
- d: that difference against the distance from the landmark centroid, per used DN.
"""
import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "results" / "f5_stageA_report.json"
CURVES = ROOT / "docs" / "results" / "f5_stageA_curves.json"
VALIDATION = ROOT / "docs" / "results" / "f5_stageA_validation.json"
FIGDATA = ROOT / "docs" / "figures"                    # landmarks.csv, transforms.json, warp_check.csv, warp_check.json
OUT = ROOT / "docs" / "figures"
NAME = "fig1_stageA_result.png"
NAME0 = "fig0_pipeline.png"
NAME5 = "fig5_training.png"
NAME6 = "fig6_alignment.png"
FIG6_HEIGHT_MM = 112
FIG5_HEIGHT_MM = 118
FIG5_ARMS = ("true", "shuffled", "derange", "mlp", "linear", "bridge_off")      # the trained arms, legend order
FIG5_D_MAX = 4000                                     # panel d axis end (training trials); later or never: right edge
FIG5_R2_LIM = (-0.3, 1.02)                            # panel c axis (the linear mean leaves it at some evaluations)
FIG5_LOSS_LIM = (0.02, 2.0)                           # panel b axis (log); a band below it is cut at the axis
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")      # local preview folder: never written
WIDTH_MM, HEIGHT_MM, WIDTH_PX, MAX_BYTES = 180, 60, 2000, 1_500_000
MM = 1 / 25.4

# Copied from scripts/f5_stageA_posthoc.py (FONT_PT, ARM_COLOR, INK / INK2 / FAINT and the rcParams of font_setup()),
# not imported: that module imports torch and the training code. bridge_off: a dark grey no other arm uses.
FONT_PT = 7
ARM_COLOR = {"true": "#2a78d6", "shuffled": "#eb6834", "derange": "#1baf7a", "mlp": "#8a5cd6", "linear": "#b8860b",
             "untrained": "#8c8a84"}
INK, INK2, FAINT = "#000000", "#555555", "#bdbdbd"
HELD_OUT_BAND = "#efefef"                              # light grey behind the held-out speeds (Figure 1c)
RC = {"font.size": FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out",
      "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
      "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
      "axes.labelsize": FONT_PT, "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1,
      "legend.fontsize": FONT_PT - 1, "axes.titlesize": FONT_PT}

BRIDGE_OFF = "#333333"
# every arm in its ARM_COLOR colour (as in the post-hoc figure, Figure 4); marker shapes tell the arms apart as well
STYLE = {"true": (ARM_COLOR["true"], "o"), "shuffled": (ARM_COLOR["shuffled"], "o"), "mlp": (ARM_COLOR["mlp"], "o"),
         "derange": (ARM_COLOR["derange"], "s"), "linear": (ARM_COLOR["linear"], "D"),
         "untrained": (ARM_COLOR["untrained"], "^"), "bridge_off": (BRIDGE_OFF, "v")}
TRAINED = ("true", "shuffled", "derange", "mlp", "linear")
# Figure 0. FlyWire-DN and BANC-DN colours copied from STYLE["fw_dn"] / STYLE["banc_dn"] of scripts/viz_anatomy.py
# (Okabe-Ito sky blue and blue), not imported; the boxes use light tints of them (mixed with white).
FW_DN, BANC_DN, NEUTRAL = "#56B4E9", "#0072B2", "#ECECEC"
FIG0_PT, FIG0_HEIGHT_MM = 9, 74


def tint(hex_colour, a):
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)
    return "#" + "".join(f"{int(round(255 - a * (255 - c))):02x}" for c in rgb)


FW_TINT, BANC_TINT = tint(FW_DN, 0.30), tint(BANC_DN, 0.40)
BOXES = (  # (title, detail lines, fill); reading order: top row left to right, then the bottom row left to right
    ("Rotating drum", ("stimulus",), NEUTRAL),
    ("FlyGym eyes, FlyVis", ("visual input, fixed",), NEUTRAL),
    ("FlyWire brain model", ("spiking, not trained", "recorded once"), FW_TINT),
    ("DN bridge", ("by cell type, fixed",), (FW_TINT, BANC_TINT)),
    ("BANC VNC subgraph", ("rate model, trained",), BANC_TINT),
    ("Fixed readout", ("steering MNs \u2192 turn", "command dA, compared", "with the teacher dA*"), BANC_TINT),
)
GROUPS = (("r2", "training speeds\n(60, 120 deg/s)"), ("g90_r2", "90 deg/s\n(held out)"),
          ("g30_r2", "30 deg/s\n(held out)"))


def font_setup() -> str:
    """Arial, else Liberation Sans, else the Windows copy of Arial, else DejaVu Sans with a warning (as
    scripts/viz_anatomy.py does); never exits for a missing font."""
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    fam = next((w for w in ("Arial", "Liberation Sans") if any(f.name == w for f in fm.fontManager.ttflist)), None)
    if fam is None:
        win = Path("/mnt/c/Windows/Fonts")
        for f in ("arial.ttf", "arialbd.ttf"):
            if (win / f).exists():
                fm.fontManager.addfont(str(win / f))
        fam = "Arial" if any(f.name == "Arial" for f in fm.fontManager.ttflist) else None
    if fam is None:
        fam = "DejaVu Sans"
        warnings.warn("neither Arial nor Liberation Sans is available; using DejaVu Sans")
        print("warning: neither Arial nor Liberation Sans is available; using DejaVu Sans", file=sys.stderr)
    mpl.rcParams.update({**RC, "font.family": fam})
    return fam


def values(runs, arm, key):
    return np.array([runs[arm][t][key] for t in sorted(runs[arm], key=lambda s: int(s[1:]))])


def mean_bar(ax, x, m, color, half=0.18):
    ax.plot([x - half, x + half], [m, m], color=color, lw=1.4, solid_capstyle="butt", zorder=4)


def draw(d):
    runs = d["runs"]
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1.6, 1.6), left=0.07, right=0.995, bottom=0.17, top=0.93,
                          wspace=0.36)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # a: true vs shuffled per training seed
    sh, tr = values(runs, "shuffled", "r2"), values(runs, "true", "r2")
    for u, w in zip(sh, tr):
        axa.plot([0, 1], [u, w], color=FAINT, lw=0.5, zorder=1)
    for x, arm, v in ((0, "shuffled", sh), (1, "true", tr)):
        axa.scatter(np.full(len(v), x), v, s=9, color=STYLE[arm][0], lw=0, zorder=2)
        mean_bar(axa, x, v.mean(), STYLE[arm][0])
    axa.set_xticks([0, 1], ["shuffled", "true"])
    axa.set_xlim(-0.5, 1.5)
    axa.set_ylabel("test R2")

    # b: the five trained comparison arms at the training speeds
    rng = np.random.default_rng(0)                      # horizontal jitter only
    for i, arm in enumerate(TRAINED):
        v = values(runs, arm, "r2")
        col, mk = STYLE[arm]
        axb.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=col, marker=mk, lw=0, zorder=2)
        mean_bar(axb, i, v.mean(), col, half=0.26)
    axb.set_xticks(range(len(TRAINED)), TRAINED)
    axb.set_xlim(-0.6, len(TRAINED) - 0.4)
    axb.set_ylabel("test R2")

    # c: every arm per condition
    arms = [a for a in STYLE if a in runs] + [a for a in runs if a not in STYLE]
    off = np.linspace(-0.3, 0.3, len(arms))
    for g in (1, 2):
        axc.axvspan(g - 0.45, g + 0.45, color=HELD_OUT_BAND, lw=0, zorder=0)
    for k, arm in enumerate(arms):
        col, mk = STYLE.get(arm, (INK2, "o"))
        for g, (key, _) in enumerate(GROUPS):
            v = values(runs, arm, key)
            x = g + off[k]
            axc.scatter(np.full(len(v), x), v, s=3, color=col, marker=mk, lw=0, alpha=0.55, zorder=2)
            axc.scatter([x], [v.mean()], s=16, color=col, marker=mk, edgecolor="white", lw=0.4, zorder=3,
                        label=arm if g == 0 else None)
    axc.set_xticks(range(len(GROUPS)), [lab for _, lab in GROUPS])
    axc.set_xlim(-0.5, len(GROUPS) - 0.5)
    allv = np.concatenate([values(runs, a, k) for a in arms for k, _ in GROUPS])
    axc.set_ylim(min(-0.06, allv.min() - 0.03), max(1.06, allv.max() + 0.03))
    axc.set_ylabel("test R2")
    axc.legend(loc="center left", bbox_to_anchor=(0.0, 0.47), frameon=False, ncol=2, handletextpad=0.2,
               columnspacing=0.8, borderaxespad=0.3)

    for ax, lab in zip((axa, axb, axc), "abc"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.05, pos.y1 + 0.02, lab, ha="left", va="bottom", fontsize=FONT_PT + 1, fontweight="bold")
    return fig


def draw_fig0():
    from matplotlib.patches import FancyArrowPatch, Rectangle
    W, H = WIDTH_MM, FIG0_HEIGHT_MM
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    bw, bh, xs, ys = 52.0, 22.0, (31.0, 90.0, 149.0), (H - 15.0, H - 49.0)
    lw = 0.8
    centres = [(x, ys[0]) for x in xs] + [(x, ys[1]) for x in xs]
    for (title, lines, fill), (cx, cy) in zip(BOXES, centres):
        x0, y0 = cx - bw / 2, cy - bh / 2
        if isinstance(fill, tuple):                                   # the bridge: half FlyWire, half BANC
            ax.add_patch(Rectangle((x0, y0), bw / 2, bh, facecolor=fill[0], edgecolor="none"))
            ax.add_patch(Rectangle((cx, y0), bw / 2, bh, facecolor=fill[1], edgecolor="none"))
            ax.add_patch(Rectangle((x0, y0), bw, bh, facecolor="none", edgecolor=INK2, lw=lw))
        else:
            ax.add_patch(Rectangle((x0, y0), bw, bh, facecolor=fill, edgecolor=INK2, lw=lw))
        n = 1 + len(lines)
        step = 4.4
        top = cy + (n - 1) * step / 2
        ax.text(cx, top, title, ha="center", va="center", fontsize=FIG0_PT, fontweight="bold", color=INK)
        for k, ln in enumerate(lines):
            ax.text(cx, top - (k + 1) * step, ln, ha="center", va="center", fontsize=FIG0_PT, color=INK)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK, shrinkA=0,
                                     shrinkB=0))
    for row in (0, 1):
        for i in range(2):
            arrow((xs[i] + bw / 2 + 0.6, ys[row]), (xs[i + 1] - bw / 2 - 0.6, ys[row]))
    mid = (ys[0] - bh / 2 + ys[1] + bh / 2) / 2                        # elbow from the brain box to the bridge box
    ax.plot([xs[2], xs[2], xs[0]], [ys[0] - bh / 2 - 0.6, mid, mid], color=INK, lw=1.0, solid_capstyle="butt")
    arrow((xs[0], mid), (xs[0], ys[1] + bh / 2 + 0.6))
    ly = 7.0
    for k, (fill, lab) in enumerate(((FW_TINT, "FlyWire (fly 1)"), (BANC_TINT, "BANC (fly 2)"))):
        x = 58.0 + k * 40.0
        ax.add_patch(Rectangle((x, ly - 2.4), 6.0, 4.8, facecolor=fill, edgecolor=INK2, lw=lw))
        ax.text(x + 8.0, ly, lab, ha="left", va="center", fontsize=FIG0_PT, color=INK)
    return fig


def draw_fig5(cur, val):
    regs = [r for r in cur["runs"] if r["selected_rate"] and not r["replicate"]]
    fig = plt.figure(figsize=(WIDTH_MM * MM, FIG5_HEIGHT_MM * MM))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.075, top=0.955, wspace=0.26, hspace=0.42)
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))

    # a: the learning-rate grid at training seed 0
    for arm in FIG5_ARMS:
        col, mk = STYLE[arm]
        g = val["lr_grid"][arm]
        lrs, loss = [x["lr"] for x in g], [x["final_val_loss"] for x in g]
        axa.plot(lrs, loss, color=col, lw=0.8, marker=mk, ms=3, zorder=2, label=arm)
        k = lrs.index(cur["selected_lr"][arm])
        axa.scatter([lrs[k]], [loss[k]], s=48, facecolor="none", edgecolor=col, lw=0.9, zorder=3)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xticks(cur["lr_grid"], [f"{x:g}" for x in cur["lr_grid"]])
    axa.minorticks_off()
    axa.set_xlabel("learning rate")
    axa.set_ylabel("final validation loss")

    # b, c: mean +- 1 SD over training seeds at the selected rate
    for ax, metric in ((axb, "loss"), (axc, "r2")):
        for arm in FIG5_ARMS:
            col, _ = STYLE[arm]
            rs = [r for r in regs if r["arm"] == arm]
            x = np.array(rs[0]["evaluations"]["trials"])
            if any(r["evaluations"]["trials"] != list(x) for r in rs):
                raise SystemExit(f"{arm}: the runs have different evaluation steps")
            y = np.array([r["evaluations"][metric] for r in rs])
            m, sd = y.mean(0), y.std(0, ddof=1)
            lo = np.maximum(m - sd, FIG5_LOSS_LIM[0]) if metric == "loss" else m - sd    # log axis: cut at the axis
            ax.fill_between(x, lo, m + sd, color=col, alpha=0.18, lw=0, zorder=1)
            ax.plot(x, m, color=col, lw=0.9, zorder=2)
        ax.set_xlim(0, cur["steps"] * cur["batch"])
        ax.set_xlabel("training trials presented")
    un = next(r for r in cur["runs"] if r["arm"] == "untrained")
    axc.axhline(un["evaluations"]["r2"][0], color=STYLE["untrained"][0], lw=0.9, ls=(0, (1, 1.5)), zorder=1,
                label="untrained")
    axb.set_yscale("log")
    axb.set_ylim(*FIG5_LOSS_LIM)
    axb.set_ylabel("validation loss")
    axc.set_ylim(*FIG5_R2_LIM)
    axc.set_ylabel("validation R2")

    # d: training trials until the turn-sign accuracy first reached the set accuracy
    rng = np.random.default_rng(0)                    # vertical jitter only
    step_trials = cur["eval_every"] * cur["batch"]
    for i, arm in enumerate(FIG5_ARMS):
        col, mk = STYLE[arm]
        for r in [r for r in regs if r["arm"] == arm]:
            e = r["evaluations"]
            hit = next((t for s, t, acc in zip(e["step"], e["trials"], e["sign_accuracy"])
                        if s > 0 and acc >= cur["set_accuracy"]), None)
            y = i + rng.uniform(-0.22, 0.22)
            if hit is not None and hit <= FIG5_D_MAX:
                axd.scatter([hit], [y], s=12, color=col, marker=mk, lw=0, zorder=2)
            else:                                     # later than the axis, or never within the budget
                axd.scatter([FIG5_D_MAX], [y], s=14, facecolor="none", edgecolor=col, marker=mk, lw=0.7, zorder=2,
                            clip_on=False)
    axd.set_xlim(0, FIG5_D_MAX)
    axd.set_xticks(np.arange(0, FIG5_D_MAX + 1, step_trials))
    axd.set_yticks(range(len(FIG5_ARMS)), FIG5_ARMS)
    axd.set_ylim(len(FIG5_ARMS) - 0.5, -0.5)
    axd.tick_params(axis="y", length=0)
    axd.set_xlabel("training trials until turn-sign accuracy first reached the set level")

    handles, labels = axa.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    axa.legend(handles + h2, labels + l2, frameon=False, ncol=2, loc="upper left", handlelength=1.6,
               columnspacing=0.8, borderaxespad=0.2)
    for ax, lab in zip((axa, axb, axc, axd), "abcd"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.06, pos.y1 + 0.012, lab, ha="left", va="bottom", fontsize=FONT_PT + 1,
                 fontweight="bold")
    return fig


def _csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def draw_fig6(lm, tr, wr):
    """lm, wr: rows of landmarks.csv and warp_check.csv; tr: transforms.json."""
    X = np.array([[float(r[k]) for k in ("fw_x", "fw_y", "fw_z")] for r in lm])
    Y = np.array([[float(r[k]) for k in ("banc_x", "banc_y", "banc_z")] for r in lm]) / 1e3
    sets = np.array([r["set"] for r in lm])
    aff = tr["transforms"]["affine"]
    out = {(o["cell_type"], o["side"]) for o in aff["outliers"]}
    rejected = np.array([(r["cell_type"], r["side"]) in out for r in lm])
    group = np.where(rejected, "outlier", sets)                        # control (fitted), check, outlier
    fig = plt.figure(figsize=(WIDTH_MM * MM, FIG6_HEIGHT_MM * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), left=0.075, right=0.985, bottom=0.085, top=0.885,
                          wspace=0.3, hspace=0.45)
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
    marker = {"control": "o", "check": "s", "outlier": "X"}

    # a: landmarks on the two BANC axes with the largest spread
    i, j = sorted(np.argsort(Y.std(0))[::-1][:2])
    F = (X @ np.array(aff["matrix"]).T + np.array(aff["translation"])) / 1e3
    for k in range(len(Y)):
        axa.plot([F[k, i], Y[k, i]], [F[k, j], Y[k, j]], color=FAINT, lw=0.4, zorder=1)
    for g in ("control", "check", "outlier"):
        m = group == g
        axa.scatter(Y[m, i], Y[m, j], s=7 if g != "outlier" else 16, marker=marker[g], color=BANC_DN, lw=0, zorder=3)
        axa.scatter(F[m, i], F[m, j], s=7 if g != "outlier" else 16, marker=marker[g], color=FW_DN, lw=0, zorder=2)
    axa.set_aspect("equal", adjustable="datalim")
    axa.set_xlabel(f"BANC {'xyz'[i]} (um)")
    axa.set_ylabel(f"BANC {'xyz'[j]} (um)")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls="", marker="o", color=BANC_DN, ms=3.5, label="BANC landmark"),
          Line2D([], [], ls="", marker="o", color=FW_DN, ms=3.5, label="FlyWire landmark, affine"),
          Line2D([], [], color=FAINT, lw=0.8, label="residual"),
          Line2D([], [], ls="", marker="o", color=INK2, ms=3.5, label="control"),
          Line2D([], [], ls="", marker="s", color=INK2, ms=3.5, label="check"),
          Line2D([], [], ls="", marker="X", color=INK2, ms=4.5, label="rejected outlier")]
    axa.legend(handles=hd, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.03), handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0.0)

    # b: residual norms, affine and similarity, control and check
    rng = np.random.default_rng(0)                                     # horizontal jitter only
    cats = []
    for name in ("affine", "similarity"):
        T = tr["transforms"][name]
        Fn = (X @ np.array(T["matrix"]).T + np.array(T["translation"])) / 1e3
        res = np.linalg.norm(Fn - Y, axis=1)
        tout = {(o["cell_type"], o["side"]) for o in T["outliers"]}
        rej = np.array([(r["cell_type"], r["side"]) in tout for r in lm])
        for st in ("control", "check"):
            x0 = len(cats)
            cats.append(f"{name}\n{st}")
            m = sets == st
            for g, mm in (("control" if st == "control" else "check", m & ~rej), ("outlier", m & rej)):
                if mm.any():
                    axb.scatter(x0 + rng.uniform(-0.18, 0.18, mm.sum()), res[mm], s=5 if g != "outlier" else 16,
                                marker=marker[g], color=INK2, lw=0, alpha=0.8 if g != "outlier" else 1, zorder=2)
            med = float(np.median(res[m]))
            axb.plot([x0 - 0.3, x0 + 0.3], [med, med], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
    axb.set_xticks(range(len(cats)), cats)
    axb.set_xlim(-0.6, len(cats) - 0.4)
    axb.set_ylim(0, None)
    axb.set_ylabel("residual norm (um)")

    # c, d: our affine against the BANC team's warp, the used DNs
    used = np.array([r["used"] == "True" for r in wr])
    err = np.array([float(r["affine_vs_warp_um"]) for r, u in zip(wr, used) if u])
    rad = np.array([float(r["dist_from_landmark_centroid_um"]) for r, u in zip(wr, used) if u])
    e = np.sort(err)
    axc.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=BANC_DN, lw=1.0)
    axc.axvline(float(np.median(err)), color=INK, lw=0.7, ls=(0, (3, 2)))
    axc.set_xlim(0, None)
    axc.set_ylim(0, 1.0)
    axc.set_xlabel("our affine vs the BANC team's warp (um)")
    axc.set_ylabel("fraction of DNs")
    axd.scatter(rad, err, s=5, color=BANC_DN, lw=0, alpha=0.75)
    axd.set_xlabel("distance from the landmark centroid (um)")
    axd.set_ylabel("our affine vs the warp (um)")
    axd.set_ylim(0, None)
    for ax, lab in zip((axa, axb, axc, axd), "abcd"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.06, pos.y1 + 0.012, lab, ha="left", va="bottom", fontsize=FONT_PT + 1, fontweight="bold")
    return fig


def render_png(fig) -> bytes:
    """PNG bytes exactly WIDTH_PX wide (no bbox_inches="tight"); the dpi is raised slightly if rounding loses a pixel."""
    from PIL import Image
    dpi = WIDTH_PX / (WIDTH_MM / 25.4)
    for _ in range(10):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        w = Image.open(io.BytesIO(buf.getvalue())).size[0]
        if w == WIDTH_PX:
            return buf.getvalue()
        dpi += 0.01 if w < WIDTH_PX else -0.01
    raise SystemExit(f"the PNG is {w} px wide, not {WIDTH_PX}")


def inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT), help="output folder (default docs/figures/)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if inside(out, PREVIEW):
        raise SystemExit(f"{out}: inside the local preview folder, which is never written")
    paths = (out / NAME0, out / NAME, out / NAME5, out / NAME6)
    exist = [str(p) for p in paths if p.exists()]
    if exist and not a.overwrite:
        raise SystemExit(f"{', '.join(exist)} exists; pass --overwrite to replace it (nothing written)")
    fam = font_setup()
    pngs = []
    for draw_fn in (lambda: draw(json.loads(REPORT.read_text())), draw_fig0,      # Figure 1 first, as before
                    lambda: draw_fig5(json.loads(CURVES.read_text()), json.loads(VALIDATION.read_text())),
                    lambda: draw_fig6(_csv(FIGDATA / "landmarks.csv"), json.loads((FIGDATA / "transforms.json").read_text()),
                                      _csv(FIGDATA / "warp_check.csv"))):
        fig = draw_fn()
        pngs.append(render_png(fig))
        plt.close(fig)
    png1, png0, png5, png6 = pngs
    for path, png in zip(paths, (png0, png1, png5, png6)):
        if len(png) > MAX_BYTES:
            raise SystemExit(f"{path.name}: {len(png):,} bytes, over {MAX_BYTES:,} (nothing written)")
    out.mkdir(parents=True, exist_ok=True)
    for path, png in zip(paths, (png0, png1, png5, png6)):
        path.write_bytes(png)
        print(f"{path}: {WIDTH_PX} px wide, {len(png):,} bytes, font {fam}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
