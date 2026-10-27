"""Figure 1 of the README: the pre-registered stage-A result on the test split (no post-hoc data; that is Figure 4).

    env -u PYTHONPATH python scripts/readme_figures.py [--out DIR] [--overwrite]

Reads only docs/results/f5_stageA_report.json and computes nothing new: every point is a stored per-run value
(runs[arm][t]["r2" | "g90_r2" | "g30_r2"]) and every bar or mean marker is the mean of those values. Writes
fig1_stageA_result.png (180 mm wide, exactly 2000 px, at most 1.5 MB) to --out (default docs/figures/); an existing
file is never overwritten without --overwrite (rule 13), and --out inside the local preview folder is refused.
- a: test R2 at the training speeds, true against shuffled per training seed (paired), means as short bars;
- b: test R2 at the training speeds of the five trained comparison arms, one dot per seed, means as short bars
  (bridge_off and untrained lie far below this axis and are shown in c);
- c: test R2 per condition (training speeds, 90 deg/s held out, 30 deg/s held out) for every arm in the JSON.
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
OUT = ROOT / "docs" / "figures"
NAME = "fig1_stageA_result.png"
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")      # local preview folder: never written
WIDTH_MM, HEIGHT_MM, WIDTH_PX, MAX_BYTES = 180, 60, 2000, 1_500_000
MM = 1 / 25.4

# Copied from scripts/f5_stageA_posthoc.py (FONT_PT, ARM_COLOR, INK / INK2 / FAINT and the rcParams of font_setup()),
# not imported: that module imports torch and the training code. bridge_off: a dark grey no other arm uses.
FONT_PT = 7
ARM_COLOR = {"true": "#2a78d6", "shuffled": "#eb6834", "derange": "#1baf7a", "mlp": "#8a5cd6", "linear": "#b8860b",
             "untrained": "#8c8a84"}
INK, INK2, FAINT = "#000000", "#555555", "#bdbdbd"
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
        axc.axvspan(g - 0.45, g + 0.45, color="#efefef", lw=0, zorder=0)
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
    path = out / NAME
    if path.exists() and not a.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    fam = font_setup()
    fig = draw(json.loads(REPORT.read_text()))
    png = render_png(fig)
    plt.close(fig)
    if len(png) > MAX_BYTES:
        raise SystemExit(f"the PNG is {len(png):,} bytes, over {MAX_BYTES:,}")
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    print(f"{path}: {WIDTH_PX} px wide, {len(png):,} bytes, font {fam}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
