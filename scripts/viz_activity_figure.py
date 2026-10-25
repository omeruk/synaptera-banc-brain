"""Visualisation (demonstration, no claim): stage-A activity on the anatomy of scripts/viz_anatomy.py. Origin class E
(engineering scaffold, viewing only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_activity_figure.py [data|figure|all] [--overwrite]

Trials: the validation split only (configs/f5_splits.json, data seeds 160-199), rot_left / rot_right at the training
speeds 60 and 120 deg/s: 160 trials, read through train/f5_stageA.load_trials (manifest SHA-256, split and DN-column
checks; it refuses test seeds and the held-out speeds). No new brain run. CPU only.
- data: FlyWire DNs: per DN column the mean rate (Hz) over the stage-A analysis window (intervals WINDOW) from the
  stored dn_counts. VNC: the `true` training-seed-0 best.pt (selected rate) with the `true` bridge input as in training
  (BANC DN clamp = unit rate / R_IN); per trial the window-mean state of every VNC neuron, the 16 steering-MN states and
  dA. Preference of a neuron = mean over the 80 (seed, speed) pairs of rot_right minus rot_left. Checks: dA from the
  states against the model's forward pass, the validation sign accuracy against the training log at the best step.
  -> runs/viz/activity_figure/val_activity.npz, data.json
- figure: a dorsal and b lateral views as in the anatomy figure (same skeleton reduction, same scale; FlyWire DNs in
  the brain, BANC DNs from the neck split on, interneurons and MNs whole), each neuron one colour of a diverging map
  (Okabe-Ito blue - light grey - vermillion, interpolated in OKLab), weakest first; symmetric ranges at the 95th
  percentile of |preference| per population (brain: FlyWire DNs, Hz; cord: BANC neurons, model a.u.). c: window-mean
  left-minus-right state per steering-MN type and direction (mean +- s.d. over the trials of a direction, both
  speeds), with the readout sign. d: dA over time per direction and speed (mean +- s.d. over its trials), teacher
  dA* dashed.
  The analysis window is shaded in d; in c, whose values are window means, the whole plot area carries that shade.
  -> stageA_circuit_activity.pdf (vector) / .png (600 dpi), caption_activity.txt, methods_activity.txt in the
  preview folder; figure.json in runs/viz/activity_figure/.
Outputs are never overwritten without --overwrite (rule 13).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import viz_anatomy as V  # noqa: E402

OUT = ROOT / "runs" / "viz" / "activity_figure"
ANAT = ROOT / "runs" / "viz" / "anatomy"
FIG_DIR = V.FIG_DIR
FIG_NAME = "stageA_circuit_activity"
COND = (("rot_left", 60), ("rot_right", 60), ("rot_left", 120), ("rot_right", 120))
ARM, TRAINING_SEED = "true", 0
CHUNK = 16
FILES = {"data": ("val_activity.npz", "data.json"), "figure": ("figure.json",)}
FIG_FILES = {"figure": (f"{FIG_NAME}.pdf", f"{FIG_NAME}.png", "caption_activity.txt", "methods_activity.txt")}

NEG, MID, POS = "#0072B2", "#D0D0D0", "#D55E00"      # Okabe-Ito blue, neutral light grey, Okabe-Ito vermillion
N_LUT = 257                                          # odd: the middle entry is the centre
DIR_COL = {"rot_right": POS, "rot_left": NEG}         # direction colour = the pole its preference points to
SPEED_LS = {120: "-", 60: (0, (1, 1.2))}
RANGE_PCT = 95
LW = {"fw_dn": 0.15, "banc_dn": 0.15, "interneuron": 0.15, "mn": 0.22}
LINE_ALPHA = 0.8
WINDOW_SHADE = "#F2F2F2"
TEACHER_COL, TEACHER_LW = "#4D4D4D", 0.5
MN_TYPES = ("b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4")
GAP_MM, PLOT_GAP_MM, TOP_MM = 5.0, 9.0, 3.5
CBAR_W_MM, CBAR_H_MM = 14.0, 1.3


# ── step: data ──────────────────────────────────────────────────────────────────────────────────────────────────────
def pair_diff(x, stim, speed, seed):
    """Mean over (seed, speed) pairs of x[rot_right] - x[rot_left]; x has the trials on axis 0."""
    idx = {(s, v, st): b for b, (st, v, s) in enumerate(zip(stim, speed, seed))}
    keys = sorted({(s, v) for _, v, s in zip(stim, speed, seed)})
    d = [x[idx[(s, v, "rot_right")]] - x[idx[(s, v, "rot_left")]] for s, v in keys]
    return np.mean(d, axis=0), len(keys)


def step_data(a):
    import torch
    import f5_train as T
    from engine import rate_torch as RT
    from train import f5_stageA as A

    torch.set_num_threads(A.TEST_THREADS)
    t0 = time.time()
    setup = T.load_setup()
    bridge = A.make_bridge(setup, ARM, TRAINING_SEED)
    sp = json.loads(A.SPLIT_FILE.read_text())
    if sorted(v for _, v in COND) != sorted(2 * list(sp["train_speeds_deg_s"])):
        raise SystemExit("the conditions are not the training speeds")
    tr = A.load_trials("val", setup, bridge, conditions=COND)
    man = A.read_manifest(A.RECORDS / "manifest.sha256")
    lo, hi = A.WINDOW
    dn_rate = np.zeros((len(tr), len(setup.fw_index)), np.float64)
    for b, f in enumerate(tr.files):
        p = A.RECORDS / f
        if A.sha256_file(p) != man[f]:
            raise SystemExit(f"{f}: SHA-256 differs from the manifest")
        z = A.open_record(p)
        if not np.array_equal(z["dn_index"], setup.fw_index):
            raise SystemExit(f"{f}: DN columns differ from the setup")
        dn_rate[b] = z["dn_counts"][lo:hi].mean(0) / (A.INTERVAL_MS / 1e3)

    model, _ = A.make_model(ARM, TRAINING_SEED, setup, bridge, T.GRAPHS)
    run_dir = A.run_dir_of(ARM, TRAINING_SEED, T.selected_lr(ARM), A.OUT)
    ck = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    B, n = len(tr), setup.n
    sigma = np.asarray(setup.sigma, np.float64)
    wmean = np.zeros((B, n), np.float32)
    mn = np.zeros((A.T_TRIAL, B, len(setup.readout)), np.float32)
    dA = np.zeros((A.T_TRIAL, B), np.float64)
    with torch.no_grad():
        for c in range(0, B, CHUNK):
            st, _ = model.net.run(model.clamp(torch.from_numpy(tr.units[:, c:c + CHUNK])), record=None)
            st = st.double().numpy()                                     # (T, chunk, n)
            wmean[c:c + CHUNK] = st[lo:hi].mean(0)
            mn[:, c:c + CHUNK] = st[:, :, setup.readout]
            dA[:, c:c + CHUNK] = A.SCALE * (st[:, :, setup.readout] * sigma).sum(2)
    dA_fwd = A.predict(model, tr.units, "cpu")
    met = A.metrics(dA_fwd, tr)
    logged = [json.loads(ln) for ln in (run_dir / "log.jsonl").read_text().splitlines()]
    ev = next(e for e in logged if e.get("event") == "eval" and e.get("step") == int(ck["step"]))

    stim, speed, seed = list(tr.stimulus), tr.speed.astype(int), tr.seed.astype(int)
    pref_fw, n_pairs = pair_diff(dn_rate, stim, speed, seed)
    pref_vnc, _ = pair_diff(wmean.astype(np.float64), stim, speed, seed)
    check = dict(
        max_abs_dA_states_vs_forward=float(np.abs(dA - dA_fwd).max()),
        sign_accuracy=met["sign_accuracy"], logged_sign_accuracy=ev["sign_accuracy"],
        logged_n_rotation=ev["n_rotation"], best_step=int(ck["step"]), logged_best_step=ev["best_step"])
    check["passed"] = bool(check["max_abs_dA_states_vs_forward"] < 1e-4
                           and check["sign_accuracy"] == check["logged_sign_accuracy"]
                           and check["logged_n_rotation"] == B and check["best_step"] == check["logged_best_step"])
    if not check["passed"]:
        raise SystemExit(f"model check failed: {check}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "val_activity.npz", stimulus=np.array(stim), speed=speed, seed=seed, files=np.array(tr.files),
        sha256=np.array(tr.sha256), dn_rate=dn_rate.astype(np.float32), fw_index=setup.fw_index,
        vnc_window_state=wmean, mn_state=mn, readout=np.asarray(setup.readout),
        readout_names=np.array(setup.meta["readout_names"]), sigma=sigma, dA=dA.astype(np.float32),
        dA_forward=dA_fwd.astype(np.float32), target=tr.target, pref_fw_hz=pref_fw, pref_vnc=pref_vnc,
        window=np.array(A.WINDOW), interval_ms=A.INTERVAL_MS, n_grey=A.N_GREY, scale=A.SCALE,
        note="visualisation only (origin class E); validation split, training speeds")
    res = dict(note="visualisation (origin class E); validation trials at the training speeds",
               trials=B, pairs=n_pairs, seeds=[int(seed.min()), int(seed.max())], n_seeds=len(set(seed.tolist())),
               speeds=sorted(set(speed.tolist())), per_direction=int(sum(st == "rot_right" for st in stim)),
               per_condition=int(sum((st == "rot_right") and (v == COND[1][1]) for st, v in zip(stim, speed))),
               bridge=dict(window_intervals=A.LATENCIES["bridge_window_intervals"],
                           latency_intervals=A.LATENCIES["bridge_latency_intervals"]),
               constants=dict(G1=A.G1, R_IN=RT.R_IN, SCALE=A.SCALE, A_MAX=A.A_MAX, V_MAX=A.V_MAX, K_P=A.K_P),
               window_intervals=list(A.WINDOW), interval_ms=A.INTERVAL_MS, n_grey=A.N_GREY,
               run_dir=str(run_dir.relative_to(ROOT)), arm=ARM, training_seed=TRAINING_SEED, lr=T.selected_lr(ARM),
               threads=A.TEST_THREADS, check=check, metrics=met, seconds=round(time.time() - t0, 1))
    (OUT / "data.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ── colour map ──────────────────────────────────────────────────────────────────────────────────────────────────────
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _lin(rgb):
    rgb = np.asarray(rgb, float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def _oklab(lin):
    return np.cbrt(np.clip(lin, 0, None) @ _M1.T) @ _M2.T


def _from_oklab(lab):
    return (lab @ np.linalg.inv(_M2).T) ** 3 @ np.linalg.inv(_M1).T


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def diverging_lut():
    """(N_LUT, 3) sRGB: NEG -> MID -> POS, each arm linear in OKLab."""
    lab = {k: _oklab(_lin(_hex(h))) for k, h in (("neg", NEG), ("mid", MID), ("pos", POS))}
    t = np.linspace(-1, 1, N_LUT)[:, None]
    arm = np.where(t < 0, lab["neg"], lab["pos"])
    return _srgb(_from_oklab(lab["mid"] + np.abs(t) * (arm - lab["mid"])))


def diverging_check(lut) -> dict:
    """Mirror separation (OKLab x 100 between +t and -t, |t| = 0.25 .. 1) and centre-to-pole distance for normal
    vision and simulated CVD (viz_anatomy.CVD), and the OKLab lightness along each arm."""
    mid = N_LUT // 2
    ts = (0.25, 0.5, 0.75, 1.0)
    out = {}
    for name, M in [("normal", np.eye(3))] + [(k, np.array(v)) for k, v in V.CVD.items()]:
        lab = _oklab(_lin(lut) @ np.asarray(M).T)
        k = [int(round(t * mid)) for t in ts]
        mirror = [float(100 * np.linalg.norm(lab[mid + i] - lab[mid - i])) for i in k]
        out[name] = dict(mirror=dict(zip(map(str, ts), np.round(mirror, 1).tolist())),
                         mirror_min=round(min(mirror), 1),
                         centre_to_neg=round(float(100 * np.linalg.norm(lab[0] - lab[mid])), 1),
                         centre_to_pos=round(float(100 * np.linalg.norm(lab[-1] - lab[mid])), 1))
    L = _oklab(_lin(lut))[:, 0]
    out["lightness"] = dict(neg=round(float(L[0]), 3), centre=round(float(L[mid]), 3), pos=round(float(L[-1]), 3),
                            monotone=bool(np.all(np.diff(L[:mid + 1]) >= 0) and np.all(np.diff(L[mid:]) <= 0)))
    return out


# ── step: figure ────────────────────────────────────────────────────────────────────────────────────────────────────
def circuit_lines(y_cut):
    """Per drawn neuron (role, id, node, parts): parts = list of (vertices um, start mask) of the drawn pieces. FlyWire
    DNs, interneurons and MNs whole; BANC DNs from the neck split (y >= y_cut) on. Reduction as viz_anatomy."""
    circ = pd.read_csv(ANAT / "circuit.csv")
    man = pd.read_csv(ANAT / "fetch_manifest.csv").fillna({"object": ""})
    man = man[man["kind"].isin(["fafb_banc", "banc"]) & (man["object"] != "")]
    obj = dict(zip(zip(man["role"], man["id"]), man["object"]))

    def one(r):
        v, st = V.polylines(*V.read_swc(obj[(r.role, r.id)]), V.TWIG_UM * 1e3, V.STEP_UM * 1e3)
        v = v / 1e3
        keep = np.r_[np.inf, np.linalg.norm(np.diff(v, axis=0), axis=1)] <= V.MAX_EDGE_UM
        if r.role == "banc_dn":
            keep &= v[:, 1] >= y_cut
        return r.role, int(r.id), int(r.node), V.runs(v, st, keep)

    rows = [r for r in circ.itertuples() if (r.role, r.id) in obj]
    with ThreadPoolExecutor(V.THREADS) as ex:
        return list(ex.map(one, rows)), len(circ) - len(rows)


def _style_axes(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, labelsize=V.FONT_PT, pad=1.5)


def colour_bar(fig, x, y, lut, r, unit, label, W, H):
    """Horizontal colour bar at (x, y) mm (lower left), drawn as vector rectangles; ticks -r, 0, +r."""
    from matplotlib.collections import PolyCollection
    ax = fig.add_axes([x / W, y / H, CBAR_W_MM / W, CBAR_H_MM / H])
    edges = np.linspace(-1, 1, len(lut) + 1)
    polys = [[(edges[i], 0), (edges[i + 1], 0), (edges[i + 1], 1), (edges[i], 1)] for i in range(len(lut))]
    ax.add_collection(PolyCollection(polys, facecolors=lut, edgecolors=lut, linewidths=0.1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([f"\u2212{r:.2g}", "0", f"{r:.2g}"])
    ax.tick_params(width=0.4, length=1.5, labelsize=V.KEY_PT, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{label} ({unit})", fontsize=V.KEY_PT, pad=1.5, loc="left")


def render(lines, meshes, d, ranges, lut, paths) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    fam = matplotlib.rcParams["font.family"][0]                          # superscripts (s^-1) in the text font
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
                                "mathtext.bf": f"{fam}:bold"})
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    s = json.loads((ANAT / "figure.json").read_text())["scale_mm_per_100um"] / 100     # mm per um, as the anatomy
    views = ("dorsal", "lateral")
    sil, uv, bounds = {}, {}, {}
    for view in views:
        R, U = (np.array(v, float) for v in V.VIEWS[view])
        sil[view] = [c for Vx, F in meshes.values() for c in V.silhouette(Vx, F, view)]
        uv[view] = [[(np.c_[p @ R, p @ U], st) for p, st in [parts] if len(p)] for _, _, _, parts in lines]
        pts = np.concatenate(sil[view] + [q for nr in uv[view] for q, _ in nr])
        bounds[view] = np.r_[pts.min(0) - V.PAD_UM, pts.max(0) + V.PAD_UM]
    lo_v = min(bounds[v][1] for v in views)
    hi_v = max(bounds[v][3] for v in views)
    for v in views:
        bounds[v][[1, 3]] = lo_v, hi_v
    w = {v: (bounds[v][2] - bounds[v][0]) * s for v in views}
    h_mm = (hi_v - lo_v) * s
    W = V.FIG_W_MM
    H = TOP_MM + h_mm + V.MARGIN_MM
    x0 = {"dorsal": V.MARGIN_MM, "lateral": V.MARGIN_MM + w["dorsal"] + GAP_MM}
    fig = plt.figure(figsize=(W * V.MM, H * V.MM))
    fig.patch.set_facecolor("white")
    fill = to_rgba(V.OUTLINE_FILL, V.OUTLINE_ALPHA)
    pref = np.array([ranges["value"][k] for k in range(len(lines))])
    rng = np.array([ranges[("brain" if role == "fw_dn" else "cord")] for role, _, _, _ in lines])
    t = np.clip(pref / rng, -1, 1)
    order = np.argsort(np.abs(t), kind="stable")                          # weakest first, strongest on top
    col = lut[np.round((t + 1) / 2 * (N_LUT - 1)).astype(int)]
    for letter, view in zip("ab", views):
        ax = fig.add_axes([x0[view] / W, V.MARGIN_MM / H, w[view] / W, h_mm / H])
        ax.set_xlim(bounds[view][0], bounds[view][2])
        ax.set_ylim(lo_v, hi_v)
        ax.set_axis_off()
        for c in sil[view]:
            ax.add_artist(PathPatch(MPath(np.r_[c, c[:1]], [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 1)
                                          + [MPath.CLOSEPOLY]), facecolor=fill, edgecolor=V.OUTLINE_EDGE,
                                    lw=V.OUTLINE_LW))
        for k in order:
            role = lines[k][0]
            for p, st in uv[view][k]:
                if len(p) < 2:
                    continue
                codes = np.where(st, MPath.MOVETO, MPath.LINETO).astype(np.uint8)
                ax.add_artist(PathPatch(MPath(p, codes), fill=False, edgecolor=col[k], lw=LW[role],
                                        alpha=LINE_ALPHA, capstyle="round", joinstyle="round"))
        fig.text((x0[view] - 0.2) / W, (H - TOP_MM + 1.2) / H, letter, fontsize=V.LETTER_PT, fontweight="bold",
                 ha="left", va="bottom")
    # orientation keys: lower right of a, lower left of b; scale bar and colour bars: lower left of a
    V.orientation_key(fig, x0["dorsal"] + w["dorsal"] - 4.2, V.MARGIN_MM + 4.2, "dorsal")
    V.orientation_key(fig, x0["lateral"] + 4.2, V.MARGIN_MM + 4.2, "lateral")
    xb, yb = x0["dorsal"] + 3.5, V.MARGIN_MM + 3.0
    colour_bar(fig, xb, yb, lut, ranges["cord"], "a.u.", "BANC neurons", W, H)
    colour_bar(fig, xb, yb + 8.5, lut, ranges["brain"], "Hz", "FlyWire DNs", W, H)
    sb_y = yb + 17.5
    fig.add_artist(Line2D([xb / W, (xb + V.SCALE_UM * s) / W], [sb_y / H] * 2, transform=fig.transFigure,
                          color="black", lw=1.0, solid_capstyle="butt"))
    fig.text((xb + V.SCALE_UM * s / 2) / W, (sb_y + 0.8) / H, f"{V.SCALE_UM} \u00b5m", ha="center", va="bottom",
             fontsize=V.FONT_PT)

    # c and d on the right
    xr = x0["lateral"] + w["lateral"] + PLOT_GAP_MM + 4.0
    wr = W - V.MARGIN_MM - 3.0 - xr                                      # room for the last tick label
    hc = (H - TOP_MM - V.MARGIN_MM - 7.0 - TOP_MM - 6.0) / 2
    lo, hi = d["window"]
    tms = (np.arange(d["dA"].shape[0]) - d["n_grey"]) * d["interval_ms"]
    stim = d["stimulus"]
    names = list(d["readout_names"])
    axc = fig.add_axes([xr / W, (V.MARGIN_MM + 7.0 + hc + TOP_MM + 6.0) / H, wr / W, hc / H])
    axc.axvspan(-0.6, len(MN_TYPES) - 0.4, color=WINDOW_SHADE, lw=0, zorder=0)
    axc.axhline(0, color="#9A9A9A", lw=0.4, zorder=1)
    signs = []
    for k, mt in enumerate(MN_TYPES):
        iL, iR = names.index(f"{mt}|left"), names.index(f"{mt}|right")
        lr = d["mn_state"][lo:hi, :, iL].mean(0) - d["mn_state"][lo:hi, :, iR].mean(0)
        signs.append("+" if d["sigma"][iL] > 0 else "\u2212")
        for dx, stn in ((-0.17, "rot_right"), (0.17, "rot_left")):
            x = lr[stim == stn]
            axc.errorbar(k + dx, x.mean(), yerr=x.std(ddof=1), fmt="o", ms=2.2, mew=0, color=DIR_COL[stn],
                         elinewidth=0.6, capsize=0, zorder=3)
    axc.set_xlim(-0.6, len(MN_TYPES) - 0.4)
    axc.set_xticks(range(len(MN_TYPES)))
    axc.set_xticklabels([f"{mt}\n{sg}" for mt, sg in zip(MN_TYPES, signs)])
    axc.set_ylabel("L \u2212 R state (a.u.)", fontsize=V.FONT_PT, labelpad=2)
    _style_axes(axc)
    axc.tick_params(axis="x", length=0)
    axd = fig.add_axes([xr / W, (V.MARGIN_MM + 7.0) / H, wr / W, hc / H])
    axd.axvspan(tms[lo], tms[hi - 1] + d["interval_ms"], color=WINDOW_SHADE, lw=0, zorder=0)
    axd.axhline(0, color="#9A9A9A", lw=0.4, zorder=1)
    for stn in ("rot_right", "rot_left"):
        for v in (120, 60):
            m = (stim == stn) & (d["speed"] == v)
            x = d["dA"][:, m]
            mu, sd = x.mean(1), x.std(1, ddof=1)
            axd.fill_between(tms, mu - sd, mu + sd, color=DIR_COL[stn], alpha=0.18, lw=0, zorder=2)
            axd.plot(tms, mu, color=DIR_COL[stn], lw=0.8, ls=SPEED_LS[v], zorder=3)
            axd.plot(tms, d["target"][:, np.flatnonzero(m)[0]], color=TEACHER_COL, lw=TEACHER_LW, ls=(0, (3, 2)),
                     zorder=4)
    axd.set_xlim(tms[0], tms[-1] + d["interval_ms"])
    axd.set_xlabel("Time from stimulus onset (ms)", fontsize=V.FONT_PT, labelpad=1.5)
    axd.set_ylabel("dA (deg)", fontsize=V.FONT_PT, labelpad=2)
    _style_axes(axd)
    handles = [Line2D([], [], color=POS, lw=0.8), Line2D([], [], color=NEG, lw=0.8),
               Line2D([], [], color="black", lw=0.8, ls=SPEED_LS[120]),
               Line2D([], [], color="black", lw=0.8, ls=SPEED_LS[60]),
               Line2D([], [], color=TEACHER_COL, lw=TEACHER_LW, ls=(0, (3, 2)))]
    axd.legend(handles, ["rot. right", "rot. left", "120\u00b0 s$^{-1}$", "60\u00b0 s$^{-1}$", "dA*"],
               loc="upper left", fontsize=V.KEY_PT, frameon=False, handlelength=2.2, borderaxespad=0.2,
               labelspacing=0.25, ncol=1)
    for ax, letter in ((axc, "c"), (axd, "d")):
        bb = ax.get_position()
        fig.text((xr - 9.0) / W, bb.y1 + TOP_MM * 0.35 / H, letter, fontsize=V.LETTER_PT, fontweight="bold",
                 ha="left", va="bottom")
    t0 = time.time()
    for ext, path in paths.items():
        fig.savefig(path, dpi=V.DPI, facecolor="white")
    plt.close(fig)
    return dict(size_mm=[W, round(H, 1)], scale_mm_per_100um=round(100 * s, 2), save_s=round(time.time() - t0, 1),
                bytes={ext: Path(p).stat().st_size for ext, p in paths.items()},
                readout_signs=dict(zip(MN_TYPES, signs)))


def step_figure(a):
    fam = V.font_setup()
    d = dict(np.load(OUT / "val_activity.npz"))
    data = json.loads((OUT / "data.json").read_text())
    fetch = json.loads((ANAT / "fetch.json").read_text())
    meshes, y_cut = V.outline_meshes(fetch)
    lines, n_missing = circuit_lines(y_cut)
    col = {int(b): k for k, b in enumerate(d["fw_index"])}
    value = [float(d["pref_fw_hz"][col[node]]) if role == "fw_dn" else float(d["pref_vnc"][node])
             for role, _, node, _ in lines]
    roles = np.array([r for r, _, _, _ in lines])
    val = np.array(value)
    ranges = dict(value=value, brain=float(np.percentile(np.abs(val[roles == "fw_dn"]), RANGE_PCT)),
                  cord=float(np.percentile(np.abs(val[roles != "fw_dn"]), RANGE_PCT)))
    lut = diverging_lut()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    info = render(lines, meshes, d, ranges, lut,
                  {"pdf": FIG_DIR / f"{FIG_NAME}.pdf", "png": FIG_DIR / f"{FIG_NAME}.png"})
    pops = {"brain": roles == "fw_dn", "cord": roles != "fw_dn"}
    info.update(
        font=fam, neck_cut_y_um=round(y_cut, 1), neurons_missing=n_missing,
        neurons=dict(pd.Series(roles).value_counts()), range_pct=RANGE_PCT,
        ranges={k: ranges[k] for k in ("brain", "cord")},
        saturated={k: int((np.abs(val[m]) > ranges[k]).sum()) for k, m in pops.items()},
        population_n={k: int(m.sum()) for k, m in pops.items()},
        preference_extremes={r: [float(val[roles == r].min()), float(val[roles == r].max())] for r in V.ROLES},
        diverging_check=diverging_check(lut),
        pdf_images=(FIG_DIR / f"{FIG_NAME}.pdf").read_bytes().count(b"/Subtype /Image"))
    info["neurons"] = {k: int(v) for k, v in info["neurons"].items()}
    info["caption_words"] = len(caption_body(info, data).split())
    (FIG_DIR / "caption_activity.txt").write_text(V._um(caption_body(info, data) + "\n\n" + V.CREDITS) + "\n",
                                                   encoding="utf-8")
    (FIG_DIR / "methods_activity.txt").write_text(methods_text(info, data, d), encoding="utf-8")
    (OUT / "figure.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def caption_body(info, data) -> str:
    n = info["neurons"]
    sp = " and ".join(str(v) for v in data["speeds"])
    return (
        "Direction preference of the stage-A circuit (validation trials).\n\n"
        "a, b, Dorsal (a) and lateral (b) views as in the anatomy figure. Each neuron is coloured by its mean "
        f"window activity for rightward minus leftward rotation ({data['n_seeds']} seeds, {sp} deg/s): "
        f"{n['fw_dn']} FlyWire DNs "
        "in Hz (spiking brain model); BANC DNs, VNC interneurons and steering motor neurons in model units "
        "(trained rate model, true wiring); strongest on top. c, Left-minus-right state per steering motor-neuron "
        "type and direction (mean +- s.d.); +/\u2212, readout sign. d, Turn command dA per direction and speed "
        "(mean +- "
        "s.d.); dashed, teacher dA*. Shading, analysis window. Demonstration only."
    ).replace(" deg/s", "\u00b0 s\u207b\u00b9").replace("+-", "\u00b1")


def methods_text(info, data, d) -> str:
    lo, hi = data["window_intervals"]
    ms = data["interval_ms"]
    dc = info["diverging_check"]
    ck = data["check"]
    r = info["ranges"]
    sig = info["readout_signs"]
    cst = data["constants"]
    paras = [
        ("Status", "Demonstration figure (origin class E, viewing only): not an experiment, no claim. Code: "
                   "scripts/viz_activity_figure.py (steps data, figure); anatomy, skeletons and outlines from "
                   "scripts/viz_anatomy.py (see methods_figure.txt)."),
        ("Trials",
         f"Validation split only (configs/f5_splits.json, data seeds {data['seeds'][0]}-{data['seeds'][1]}): "
         f"rot_left and rot_right at the training speeds {' and '.join(map(str, data['speeds']))} deg/s, "
         f"{data['trials']} trials ({data['pairs']} seed-speed pairs). No test seed and no held-out speed; every "
         "record is read through train/f5_stageA.load_trials (manifest SHA-256, split and DN-column checks). The "
         "stored stage-A records (runs/f5/stageA/brain/) are used as they are; no new brain run."),
        ("Window",
         f"The stage-A analysis window: intervals {lo}-{hi - 1} of {ms:g} ms ({(lo - data['n_grey']) * ms:g}-"
         f"{(hi - data['n_grey']) * ms:g} ms after stimulus onset; the grating starts at interval {data['n_grey']})."),
        ("FlyWire DNs",
         "Per DN column of the record, the mean spike count over the window divided by the interval length (Hz); "
         "the spiking FlyWire v783 brain model of the stage-A records."),
        ("VNC model",
         f"The `{data['arm']}` arm, training seed {data['training_seed']}, selected rate {data['lr']:g} "
         f"({data['run_dir']}/best.pt, step {ck['best_step']}), run on the CPU ({data['threads']} threads). Input "
         "as in training: the recorded DN counts through the `true` DN bridge (window "
         f"{data['bridge']['window_intervals']} intervals, latency {data['bridge']['latency_intervals']}) into "
         "unit rates, BANC DN clamp = unit rate / R_IN. State = the rate-model state of each neuron (model a.u.; "
         "for the BANC DNs the clamp). Checks: dA from the states equals the forward pass (max difference "
         f"{ck['max_abs_dA_states_vs_forward']:.1e} deg); validation sign accuracy {ck['sign_accuracy']:.2f} over "
         f"{data['trials']} rotation trials, as logged at the best step ({ck['logged_sign_accuracy']:.2f})."),
        ("Only the true model",
         "The shuffled control is a degree-preserving shuffle of the edges, so its wiring does not correspond to "
         "these skeletons; showing its activity on them would suggest an anatomy it does not have."),
        ("Preference",
         "Per neuron, the window mean for rot_right minus that for rot_left of the same seed and speed, averaged "
         f"over the {data['pairs']} pairs. Normalisation: symmetric ranges at the {info['range_pct']}th percentile "
         f"of |preference| of each population: brain (FlyWire DNs, n = {info['population_n']['brain']}) "
         f"{r['brain']:.3g} Hz; cord (BANC DNs, interneurons and MNs, n = {info['population_n']['cord']}) "
         f"{r['cord']:.3g} a.u.; values beyond saturate ({info['saturated']['brain']} and "
         f"{info['saturated']['cord']} neurons)."),
        ("Colour map",
         f"Diverging, {NEG} (Okabe-Ito blue, rot_left preferred) - {MID} (neutral light grey) - {POS} (Okabe-Ito "
         f"vermillion, rot_right preferred), each arm linear in OKLab, {N_LUT} steps. Check (OKLab x 100, simulated "
         "colour-vision deficiency as in methods_figure.txt): smallest separation of +t and -t over |t| = 0.25-1: "
         + ", ".join(f"{k} {dc[k]['mirror_min']:g}" for k in ("normal", "protan", "deutan", "tritan"))
         + "; centre to the poles (blue, vermillion): "
         + ", ".join(f"{k} {dc[k]['centre_to_neg']:g} / {dc[k]['centre_to_pos']:g}"
                     for k in ("normal", "protan", "deutan", "tritan"))
         + f"; OKLab lightness blue {dc['lightness']['neg']:g}, centre {dc['lightness']['centre']:g}, vermillion "
           f"{dc['lightness']['pos']:g}, monotone along each arm: {'yes' if dc['lightness']['monotone'] else 'no'}. "
           "The direction colours of c and d are the two poles."),
        ("Drawing (a, b)",
         "Views, scale, outlines, skeleton reduction and the neck split as in the anatomy figure (methods_figure.txt"
         f", {info['scale_mm_per_100um']:g} mm per 100 um). FlyWire DNs are drawn in the brain, BANC DNs from the "
         f"posterior end of the brain outline on (y >= {info['neck_cut_y_um']:g} um), interneurons and MNs whole. "
         "Each neuron has one colour; neurons are drawn in order of |preference| / range, weakest first, so the "
         f"strongest lie on top. Line widths {', '.join(f'{V.KEY_LABELS[k]} {LW[k]:g} pt' for k in V.ROLES)}; "
         f"alpha {LINE_ALPHA:g}."),
        ("Steering motor neurons (c)",
         "Per trial and MN type, the window-mean state of the left minus the right MN; mean +- s.d. over the "
         f"{data['per_direction']} trials of each direction ({data['n_seeds']} seeds x {len(data['speeds'])} "
         "speeds). Readout signs sigma "
         "from the model's fixed readout "
         "(setup.sigma, Lindsay lateral set): " + ", ".join(f"{k} {v}" for k, v in sig.items())
         + f". dA = G1 x R_IN x sum over types of sigma (s_L - s_R), G1 = {cst['G1']:g} deg/Hz, R_IN = "
         f"{cst['R_IN']:g} Hz (scale "
         f"{float(d['scale']):g} deg per a.u.), so a type contributes to a right turn when its L - R state has the "
         "sign of sigma. Types are not averaged together because their signs differ. The values are window means, "
         "so the whole plot area carries the window shading."),
        ("Turn command (d)",
         f"dA per {ms:g} ms interval, mean +- s.d. over the {data['per_condition']} trials of each direction and "
         "speed; time from stimulus onset (interval start). Teacher dA* (thin dashed): K_P v with K_P = A_MAX / "
         f"V_MAX = {cst['A_MAX']:g} deg / {cst['V_MAX']:g} deg/s = {cst['K_P']:.3g} s, v = +speed for rot_right "
         "and -speed for rot_left after onset, 0 in the grey lead-in. "
         "Shading: the analysis window."),
        ("Type and output",
         f"{info['font']}, {V.FONT_PT} pt (keys {V.KEY_PT:g} pt), panel letters {V.LETTER_PT} pt bold, "
         f"{info['size_mm'][0]:g} x {info['size_mm'][1]:g} mm. PDF fully vector ({info['pdf_images']} raster "
         f"images; {info['bytes']['pdf'] / 1e6:.0f} MB; colour bars drawn as {N_LUT} rectangles); PNG {V.DPI} dpi."),
        ("Data", V.CREDITS.removeprefix("Data: "))]
    text = "Figure methods: stage-A activity on the circuit anatomy\n\n" + "\n\n".join(f"{h}. {t}" for h, t in paras)
    return V._um(text.replace(" deg/s", "\u00b0 s\u207b\u00b9").replace("+-", "\u00b1")) + "\n"


STEPS = {"data": step_data, "figure": step_figure}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    steps = list(STEPS) if a.step == "all" else [a.step]
    existing = [str(OUT / f) for st in steps for f in FILES.get(st, ()) if (OUT / f).exists()]
    existing += [str(FIG_DIR / f) for st in steps for f in FIG_FILES.get(st, ()) if (FIG_DIR / f).exists()]
    if existing and not a.overwrite:
        raise SystemExit(f"{existing} exist; pass --overwrite")
    for st in steps:
        STEPS[st](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
