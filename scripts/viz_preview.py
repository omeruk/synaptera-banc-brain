"""Visualisation scaffold: three orthogonal projections of the FlyWire brain (transformed into BANC space) and the BANC
VNC subgraph, coloured by dataset, with the DN landmarks. Reads only runs/viz/ (scripts/viz_positions.py).

    env -u PYTHONPATH python scripts/viz_preview.py [--transform affine|similarity|both] [--by-source] [--out DIR]

BANC axes as the views assume them: y the body axis (brain at small y, VNC at large y), x left-right, z depth.
One PNG per transform: positions_<transform>.png (front x-z, side y-z, top x-y); with --by-source the brain is coloured
by position source instead (positions_<transform>_by_source.png). PNGs are never committed.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "viz"
OUT = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/viz")     # machine 2: viewing in Windows

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BRAIN, VNC = "#2a78d6", "#eb6834"
SOURCES = (("soma", "#2a78d6", 0.25), ("centroid_synapses", "#1aa39a", 0.5), ("centroid_fallback", "#c0398f", 0.8))
VIEWS = (("front (x-z)", 0, 2), ("side (y-z)", 1, 2), ("top (x-y)", 0, 1))


def draw(transform: str, runs: Path, out: Path, by_source: bool = False) -> Path:
    tr = json.loads((runs / "transforms.json").read_text())["transforms"][transform]
    M, t = np.array(tr["matrix"]), np.array(tr["translation"])
    brain = np.load(runs / "positions_brain.npz")
    vnc = np.load(runs / "positions_vnc.npz")
    lm = pd.read_csv(runs / "landmarks.csv")
    b = (brain["xyz_nm"] @ M.T + t) / 1e3
    v = vnc["xyz_nm"][vnc["kept"]] / 1e3
    lf = (lm[["fw_x", "fw_y", "fw_z"]].to_numpy(float) @ M.T + t) / 1e3
    lb = lm[["banc_x", "banc_y", "banc_z"]].to_numpy(float) / 1e3
    src = brain["source"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), facecolor=SURF)
    for ax, (name, i, j) in zip(axes, VIEWS):
        ax.set_facecolor(SURF)
        if by_source:
            ax.scatter(v[:, i], v[:, j], s=0.4, color=GRID, alpha=0.6, lw=0, rasterized=True)
            for sname, col, al in SOURCES:                   # rarest last, so it stays visible
                m = src == sname
                ax.scatter(b[m, i], b[m, j], s=0.6 if sname == "centroid_fallback" else 0.15, color=col, alpha=al,
                           lw=0, rasterized=True)
        else:
            ax.scatter(b[:, i], b[:, j], s=0.15, color=BRAIN, alpha=0.25, lw=0, rasterized=True)
            ax.scatter(v[:, i], v[:, j], s=0.4, color=VNC, alpha=0.35, lw=0, rasterized=True)
        ax.scatter(lb[:, i], lb[:, j], s=14, marker="x", color=INK, lw=0.8, zorder=3)
        ax.scatter(lf[:, i], lf[:, j], s=16, facecolor="none", edgecolor=INK2, lw=0.8, zorder=3)
        ax.set_aspect("equal")
        ax.invert_yaxis() if j == 1 else None
        ax.set_title(name, fontsize=11, color=INK)
        ax.set_xlabel(f"BANC {'xyz'[i]} (um)", color=INK2)
        ax.set_ylabel(f"BANC {'xyz'[j]} (um)", color=INK2)
        ax.tick_params(colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)
    if by_source:
        handles = [plt.Line2D([], [], ls="", marker="o", color=col, label=f"brain, {name}, {int((src == name).sum()):,}")
                   for name, col, _ in SOURCES]
        handles.append(plt.Line2D([], [], ls="", marker="o", color=GRID, label=f"BANC VNC subgraph, {len(v):,}"))
    else:
        handles = [plt.Line2D([], [], ls="", marker="o", color=BRAIN, label=f"FlyWire brain, {len(b):,} (transformed)"),
                   plt.Line2D([], [], ls="", marker="o", color=VNC, label=f"BANC VNC subgraph, {len(v):,}")]
    handles += [plt.Line2D([], [], ls="", marker="x", color=INK, label="landmark, BANC mean soma"),
               plt.Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="landmark, FlyWire mean soma (transformed)")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if by_source else 4, frameon=False, fontsize=10)
    fig.suptitle(f"FlyWire -> BANC, {transform} ({tr['n_params']} parameters): RMSE control "
                 f"{tr['rmse_control_um']:.1f} um, check {tr['rmse_check_um']:.1f} um; determinant "
                 f"{tr['determinant']:.3f}. Visualisation scaffold only.", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08 if by_source else 0.05, 1, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"positions_{transform}{'_by_source' if by_source else ''}.png"
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="both", choices=("affine", "similarity", "both"))
    ap.add_argument("--by-source", action="store_true", help="colour the brain by position source")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    for tf in (("affine", "similarity") if a.transform == "both" else (a.transform,)):
        print(draw(tf, Path(a.runs), Path(a.out), a.by_source), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
