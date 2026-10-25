"""Visualisation (demonstration, no claim): a short video of the stage-A circuit. Origin class E (viewing only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_video.py [data|stills|test|video] [--overwrite]

Trials: the validation split only (configs/f5_splits.json, data seeds 160-199), rot_right and rot_left at 120 deg/s
(one speed, so dA and the teacher match): 80 trials through train/f5_stageA.load_trials (manifest SHA-256, split and
DN-column checks; it refuses test seeds and held-out speeds). CPU only.
- data: FlyWire DNs: per DN column and 5 ms interval the rate (Hz) from the stored dn_counts (untrained reference
  brain, arm R required in every record). Cord: the `true` training-seed-0 best.pt with the `true` bridge input as in
  training; the state of each drawn BANC neuron per interval, and dA. Per direction the mean over the 40 seeds; per
  neuron minus its own pre-stimulus baseline (mean over BASELINE_MS) and smoothed causally (exponential, TAU_MS).
  -> runs/viz/video/traces.npz, data.json
- stills: +0, +300 and +800 ms of rot_right on a light and a dark background, and the title and end cards (dark).
- test: a 3 s test clip (dark), rot_right from +0 to +300 ms of the trial.
- video: title card, rot_right, rot_left (each -200 to +1000 ms, slowed SLOW x), end card.
Neurons, anatomy and neck split as scripts/viz_activity_figure.py (circuit_lines). Colour: the activity figure's
diverging map (light: light-grey centre; dark: dark-grey centre), one symmetric range per population at the 95th
percentile of |value| over its neurons and every frame of both clips. Camera: perspective turntable about the fly's
dorsal-ventral axis (BANC z), elevation ELEV_DEG, from the fly's left side (anterior-posterior axis horizontal)
ORBIT_DEG per clip with cosine easing; dorsal stays up. Frames are drawn at 2x (3840 x 2160, matplotlib Agg) and
downsampled (area); 1 pt = 1 px at 1080p. Encoding: H.264 (libx264, yuv420p, CRF 18, +faststart), 30 fps, via the
ffmpeg of imageio-ffmpeg. Outputs into the preview video folder; never overwritten without --overwrite (rule 13).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import viz_activity_figure as AF  # noqa: E402
import viz_anatomy as V  # noqa: E402

OUT = ROOT / "runs" / "viz" / "video"
ANAT = ROOT / "runs" / "viz" / "anatomy"
VID_DIR = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/video")
SPEED = 120
STIMULI = ("rot_right", "rot_left")
COND = tuple((s, SPEED) for s in ("rot_left", "rot_right"))
ARM, TRAINING_SEED, BRAIN_ARM = "true", 0, "R"
CHUNK = 16
BASELINE_MS = (-200, 0)
TAU_MS = 25.0
CLIP_MS = (-200, 1000)
STILL_MS = (0, 300, 800)
TEST_MS = (0, 300)
FPS, SLOW = 30, 10
TITLE_S, END_S, FADE_S = 2.0, 3.0, 0.3
ELEV_DEG, AZ0_DEG, ORBIT_DEG = 30.0, 0.0, 90.0
DIST_UM = 4500.0                                     # camera distance: mild perspective
RANGE_PCT = 95
W, H, SS = 1920, 1080, 2                             # output size, supersampling
VIEW_BOX = (110, 130, 1810, 880)                     # x0, y0, x1, y1 (px) for the circuit
LW_PX = {"fw_dn": 0.7, "banc_dn": 0.7, "interneuron": 0.8, "mn": 1.2}
LINE_ALPHA = 0.85
THREADS = 12
TITLE = "Visual turning signal from brain to wing motor neurons"
SOURCES = "Data: FlyWire v783 connectome (brain) and BANC v888 connectome (brain and nerve cord)"
END_LINE = "Brain: untrained spiking reference model; only the nerve cord was trained."
END_CREDITS = (
    "FlyWire v783: Dorkenwald et al. 2024; Schlegel et al. 2024; EM: Zheng et al. 2018. CC BY-NC 4.0.",
    "BANC v888: Bates, Phelps, Kim, Yang et al. 2026, Nature 656:957-970. CC BY 4.0. doi:10.7910/DVN/7WTH1N.",
    "Skeletons, warped FlyWire skeletons and outlines: gs://lee-lab_brain-and-nerve-cord-fly-connectome.",
    "This video: CC BY-NC 4.0 (it combines both sources).")
THEMES = {
    "light": dict(bg="#FFFFFF", fill=V.OUTLINE_FILL, fill_alpha=V.OUTLINE_ALPHA, edge=V.OUTLINE_EDGE, mid=AF.MID,
                  ink="#1A1A1A", ink2="#6B6B6B", teacher=AF.TEACHER_COL, zero="#BDBDBD"),
    "dark": dict(bg="#0E0F11", fill="#24262A", fill_alpha=0.75, edge="#3E4146", mid="#4D4D4D",
                 ink="#E8E8E8", ink2="#9A9A9A", teacher="#A8A8A8", zero="#4A4A4A")}
FILES = {"data": ("traces.npz", "data.json")}


def n_clip() -> int:
    """Frames per clip, both ends included: one frame every 1000 / (SLOW x FPS) ms of trial time."""
    return int(round((CLIP_MS[1] - CLIP_MS[0]) / 1000 * SLOW * FPS)) + 1


# ── step: data ──────────────────────────────────────────────────────────────────────────────────────────────────────
def smooth(x, dt_ms):
    """Causal exponential filter along axis 0, y_k = a y_{k-1} + (1 - a) x_k, y_{-1} = 0, a = exp(-dt / TAU_MS)."""
    a = math.exp(-dt_ms / TAU_MS)
    y, out = np.zeros(x.shape[1:]), np.zeros(x.shape)
    for k in range(len(x)):
        y = a * y + (1 - a) * x[k]
        out[k] = y
    return out


def step_data(a):
    import pandas as pd
    import torch
    import f5_train as T
    from train import f5_stageA as A

    torch.set_num_threads(A.TEST_THREADS)
    t0 = time.time()
    setup = T.load_setup()
    bridge = A.make_bridge(setup, ARM, TRAINING_SEED)
    if SPEED not in json.loads(A.SPLIT_FILE.read_text())["train_speeds_deg_s"]:
        raise SystemExit(f"{SPEED} deg/s is not a training speed")
    tr = A.load_trials("val", setup, bridge, conditions=COND)
    man = A.read_manifest(A.RECORDS / "manifest.sha256")
    rate = np.zeros((A.T_TRIAL, len(tr), len(setup.fw_index)))
    arms, model_sha = set(), set()
    for b, f in enumerate(tr.files):
        p = A.RECORDS / f
        if A.sha256_file(p) != man[f]:
            raise SystemExit(f"{f}: SHA-256 differs from the manifest")
        z = A.open_record(p)
        if not np.array_equal(z["dn_index"], setup.fw_index):
            raise SystemExit(f"{f}: DN columns differ from the setup")
        rate[:, b] = z["dn_counts"] / (A.INTERVAL_MS / 1e3)
        arms.add(str(z["model_arm"]))
        model_sha.add(str(z["model_sha256"]))
    if arms != {BRAIN_ARM}:
        raise SystemExit(f"brain records come from arms {sorted(arms)}, not only the reference arm {BRAIN_ARM}")

    circ = pd.read_csv(ANAT / "circuit.csv")
    nodes = circ.loc[circ["role"] != "fw_dn", "node"].to_numpy(np.int64)        # drawn BANC neurons
    model, _ = A.make_model(ARM, TRAINING_SEED, setup, bridge, T.GRAPHS)
    run_dir = A.run_dir_of(ARM, TRAINING_SEED, T.selected_lr(ARM), A.OUT)
    ck = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    B = len(tr)
    cord = np.zeros((A.T_TRIAL, B, len(nodes)))
    dA = np.zeros((A.T_TRIAL, B))
    sigma = np.asarray(setup.sigma, np.float64)
    with torch.no_grad():
        for c in range(0, B, CHUNK):
            st, _ = model.net.run(model.clamp(torch.from_numpy(tr.units[:, c:c + CHUNK])), record=None)
            st = st.double().numpy()
            cord[:, c:c + CHUNK] = st[:, :, nodes]
            dA[:, c:c + CHUNK] = A.SCALE * (st[:, :, setup.readout] * sigma).sum(2)
    dA_fwd = A.predict(model, tr.units, "cpu")
    diff = float(np.abs(dA - dA_fwd).max())
    if diff > 1e-4:
        raise SystemExit(f"dA from the states differs from the forward pass ({diff})")

    ms = A.INTERVAL_MS
    t_end = (np.arange(A.T_TRIAL) - A.N_GREY + 1) * ms                     # end of each interval, ms from onset
    base = (t_end > BASELINE_MS[0]) & (t_end <= BASELINE_MS[1])
    stim = np.array(tr.stimulus)
    out = dict(t_ms=t_end, cord_nodes=nodes, fw_index=setup.fw_index, interval_ms=ms, n_grey=A.N_GREY,
               baseline_ms=np.array(BASELINE_MS), tau_ms=TAU_MS, speed=SPEED)
    for s in STIMULI:
        m = stim == s
        for name, x in (("fw", rate[:, m].mean(1)), ("cord", cord[:, m].mean(1))):
            out[f"{name}_raw_{s}"] = x.astype(np.float32)
            out[f"{name}_{s}"] = smooth(x - x[base].mean(0), ms).astype(np.float32)
        out[f"dA_{s}"] = dA_fwd[:, m].mean(1).astype(np.float32)
        out[f"dA_sd_{s}"] = dA_fwd[:, m].std(1, ddof=1).astype(np.float32)
        out[f"target_{s}"] = tr.target[:, np.flatnonzero(m)[0]]
        out[f"n_{s}"] = int(m.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "traces.npz", **out)
    res = dict(note="visualisation (origin class E); validation trials at 120 deg/s", trials=B,
               per_direction={s: out[f"n_{s}"] for s in STIMULI}, seeds=[int(tr.seed.min()), int(tr.seed.max())],
               speed=SPEED, baseline_ms=list(BASELINE_MS), baseline_intervals=int(base.sum()), tau_ms=TAU_MS,
               run_dir=str(run_dir.relative_to(ROOT)), best_step=int(ck["step"]), brain_model_arm=sorted(arms),
               brain_model_sha256=sorted(model_sha), max_abs_dA_states_vs_forward=diff, cord_neurons=len(nodes),
               fw_columns=len(setup.fw_index), seconds=round(time.time() - t0, 1))
    (OUT / "data.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ── scene ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def schedule():
    """Every frame of the full video: kind (title, clip, end), stimulus, trial time (ms), camera azimuth (deg)."""
    fr = [dict(kind="title", i=i, n=int(TITLE_S * FPS)) for i in range(int(TITLE_S * FPS))]
    n = n_clip()
    for c, s in enumerate(STIMULI):
        for i in range(n):
            u = i / (n - 1)
            fr.append(dict(kind="clip", stim=s, t=CLIP_MS[0] + u * (CLIP_MS[1] - CLIP_MS[0]),
                           az=AZ0_DEG + ORBIT_DEG * (c + ease(u))))
    fr += [dict(kind="end", i=i, n=int(END_S * FPS)) for i in range(int(END_S * FPS))]
    return fr


def clip_frame(stim, t):
    """The clip frame of a stimulus at trial time t (nearest frame)."""
    n = n_clip()
    i = int(round((t - CLIP_MS[0]) / (CLIP_MS[1] - CLIP_MS[0]) * (n - 1)))
    c = STIMULI.index(stim)
    u = i / (n - 1)
    return dict(kind="clip", stim=stim, t=CLIP_MS[0] + u * (CLIP_MS[1] - CLIP_MS[0]),
                az=AZ0_DEG + ORBIT_DEG * (c + ease(u)))


class Camera:
    """Perspective camera on a circle about the vertical through the circuit centre; focal and screen offset fixed."""

    def __init__(self, centre, focal=1.0, cx=0.0, cy=0.0):
        self.c, self.f, self.cx, self.cy = np.asarray(centre, float), focal, cx, cy

    def basis(self, az):
        a, e = math.radians(az), math.radians(ELEV_DEG)
        pos = self.c + DIST_UM * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        fwd = (self.c - pos) / np.linalg.norm(self.c - pos)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return pos, fwd, right, np.cross(right, fwd)

    def project(self, P, az):
        pos, fwd, right, up = self.basis(az)
        d = P - pos
        z = d @ fwd
        return np.c_[self.cx + self.f * (d @ right) / z, self.cy - self.f * (d @ up) / z]


def fit_camera(points, centre) -> Camera:
    """Focal length and screen offset so that the circuit fits VIEW_BOX at every azimuth of the orbit."""
    cam = Camera(centre)
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for az in np.linspace(AZ0_DEG, AZ0_DEG + 2 * ORBIT_DEG, 37):
        q = cam.project(points, az)
        lo, hi = np.minimum(lo, q.min(0)), np.maximum(hi, q.max(0))
    bx0, by0, bx1, by1 = VIEW_BOX
    f = min((bx1 - bx0) / (hi[0] - lo[0]), (by1 - by0) / (hi[1] - lo[1]))
    return Camera(centre, f, (bx0 + bx1) / 2 - f * (lo[0] + hi[0]) / 2, (by0 + by1) / 2 - f * (lo[1] + hi[1]) / 2)


def lut_for(theme):
    """The activity figure's diverging map (OKLab arms) with the theme's neutral centre."""
    lab = {k: AF._oklab(AF._lin(AF._hex(h))) for k, h in (("neg", AF.NEG), ("mid", THEMES[theme]["mid"]),
                                                          ("pos", AF.POS))}
    t = np.linspace(-1, 1, AF.N_LUT)[:, None]
    arm = np.where(t < 0, lab["neg"], lab["pos"])
    return AF._srgb(AF._from_oklab(lab["mid"] + np.abs(t) * (arm - lab["mid"])))


def palette_check(theme) -> dict:
    """The activity figure's check (mirror and centre-to-pole separation under simulated CVD), lightness monotone
    away from the centre in either direction, and the distance of centre and poles from the background."""
    lut = lut_for(theme)
    c = AF.diverging_check(lut)
    L = AF._oklab(AF._lin(lut))[:, 0]
    mid = AF.N_LUT // 2
    up = np.all(np.diff(L[mid:]) >= 0) and np.all(np.diff(L[:mid + 1]) <= 0)
    down = np.all(np.diff(L[mid:]) <= 0) and np.all(np.diff(L[:mid + 1]) >= 0)
    c["lightness"]["monotone"] = bool(up or down)
    c["lightness"]["direction_from_centre"] = "up" if up else "down" if down else "mixed"
    bg = AF._oklab(AF._lin(AF._hex(THEMES[theme]["bg"])))
    lab = AF._oklab(AF._lin(lut))

    def lum(rgb):
        lin = AF._lin(rgb)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])

    lb = lum(AF._hex(THEMES[theme]["bg"]))
    c["background"] = {k: dict(delta_e=round(float(100 * np.linalg.norm(lab[i] - bg)), 1),
                               contrast=round((max(lum(lut[i]), lb) + 0.05) / (min(lum(lut[i]), lb) + 0.05), 2))
                       for k, i in (("neg", 0), ("centre", mid), ("pos", -1))}
    return c


class Scene:
    """Geometry (projected per frame), activity traces, ranges and camera; shared by the frame workers."""

    def __init__(self):
        import pandas as pd
        fetch = json.loads((ANAT / "fetch.json").read_text())
        meshes, y_cut = V.outline_meshes(fetch)
        self.meshes = [(Vx / 1e3, F) for Vx, F in meshes.values()]
        lines, _ = AF.circuit_lines(y_cut)
        self.role = np.array([r for r, _, _, _ in lines])
        self.node = np.array([nd for _, _, nd, _ in lines])
        verts = [p for _, _, _, (p, _) in lines]
        self.codes = [np.where(st, 1, 2).astype(np.uint8) for _, _, _, (_, st) in lines]
        self.off = np.r_[0, np.cumsum([len(p) for p in verts])]
        self.P = np.concatenate(verts)
        tr = dict(np.load(OUT / "traces.npz"))
        self.t_ms = tr["t_ms"]
        col = {int(b): k for k, b in enumerate(tr["fw_index"])}
        cpos = {int(n): k for k, n in enumerate(tr["cord_nodes"])}
        circ = pd.read_csv(ANAT / "circuit.csv")
        if not np.array_equal(np.sort(tr["cord_nodes"]), np.sort(circ.loc[circ["role"] != "fw_dn", "node"])):
            raise SystemExit("traces.npz and the circuit differ")
        self.fw = self.role == "fw_dn"
        self.trace = {}
        for s in STIMULI:
            x = np.zeros((len(self.t_ms), len(lines)))
            x[:, self.fw] = tr[f"fw_{s}"][:, [col[int(n)] for n in self.node[self.fw]]]
            x[:, ~self.fw] = tr[f"cord_{s}"][:, [cpos[int(n)] for n in self.node[~self.fw]]]
            self.trace[s] = x
            self.trace[f"dA_{s}"], self.trace[f"target_{s}"] = tr[f"dA_{s}"], tr[f"target_{s}"]
        frames = [f for f in schedule() if f["kind"] == "clip"]
        allv = np.stack([self.value(f["stim"], f["t"]) for f in frames])     # (frames, neurons)
        self.range = {"brain": float(np.percentile(np.abs(allv[:, self.fw]), RANGE_PCT)),
                      "cord": float(np.percentile(np.abs(allv[:, ~self.fw]), RANGE_PCT))}
        self.rng = np.where(self.fw, self.range["brain"], self.range["cord"])
        sample = np.concatenate([self.P[::40]] + [Vx for Vx, _ in self.meshes])
        self.camera = fit_camera(sample, sample.mean(0))

    def value(self, stim, t):
        """Per neuron, the trace linearly interpolated at trial time t (ms)."""
        x = self.trace[stim]
        j = float(np.interp(t, self.t_ms, np.arange(len(self.t_ms))))
        i0 = int(math.floor(j))
        i1 = min(i0 + 1, len(self.t_ms) - 1)
        return x[i0] * (1 - (j - i0)) + x[i1] * (j - i0)


# ── frame drawing ───────────────────────────────────────────────────────────────────────────────────────────────────
_FIG = {}


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "fig" not in _FIG:
        V.font_setup()
        dpi = 72 * SS                                                     # 1 pt = 1 px at 1080p
        fig = plt.figure(figsize=(W * SS / dpi, H * SS / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        _FIG.update(fig=fig, ax=ax)
    return _FIG["fig"], _FIG["ax"]


def silhouettes(scene, az):
    """Outer contours (1080p px) of the projected brain and VNC outlines, rasterised at the 2x size."""
    import cv2
    out = []
    for Vx, F in scene.meshes:
        q = np.round(scene.camera.project(Vx, az) * SS).astype(np.int32)
        img = np.zeros((H * SS, W * SS), np.uint8)
        cv2.fillPoly(img, list(q[F]), 255)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out += [cv2.approxPolyDP(c, 1.0, True)[:, 0, :] / SS for c in cs if cv2.contourArea(c) > 50]
    return out


def draw_drum(ax, th, stim, x, y, r=34):
    """Drum seen from above, the fly in the middle (anterior up); the arrow shows the pattern motion in front of the
    fly: to the fly's right (clockwise from above) for rot_right."""
    from matplotlib.patches import Circle, Polygon
    ax.add_artist(Circle((x, y), r, fill=False, ec=th["ink"], lw=1.4))
    for k in range(12):
        a = 2 * math.pi * k / 12
        ax.plot([x + 0.78 * r * math.cos(a), x + r * math.cos(a)], [y + 0.78 * r * math.sin(a), y + r * math.sin(a)],
                color=th["ink"], lw=1.4, solid_capstyle="butt")
    ax.add_artist(Polygon([(x, y - 9), (x - 6, y + 7), (x + 6, y + 7)], closed=True, fc=th["ink"], ec="none"))
    col = AF.DIR_COL[stim]
    ra = r + 9
    a0, a1 = (math.radians(130), math.radians(50)) if stim == "rot_right" else (math.radians(50), math.radians(130))
    a = np.linspace(a0, a1, 40)
    ax.plot(x + ra * np.cos(a), y - ra * np.sin(a), color=col, lw=2.2, solid_capstyle="round")
    tip = np.array([x + ra * math.cos(a1), y - ra * math.sin(a1)])
    tang = np.array([-math.sin(a1), -math.cos(a1)]) * (-1 if a1 < a0 else 1)     # motion direction at the tip
    tang /= np.linalg.norm(tang)
    nrm = np.array([-tang[1], tang[0]])
    ax.add_artist(Polygon([tip + 9 * tang, tip - 3 * tang + 6 * nrm, tip - 3 * tang - 6 * nrm], closed=True, fc=col,
                          ec="none"))


def draw_colour_bars(ax, th, lut, ranges, x=130, y=905):
    from matplotlib.collections import PolyCollection
    w, h = 190, 9
    for k, (label, unit, r) in enumerate((("FlyWire DNs", "Hz", ranges["brain"]),
                                          ("BANC neurons", "a.u.", ranges["cord"]))):
        yy = y + 62 * k
        e = np.linspace(x, x + w, len(lut) + 1)
        ax.add_collection(PolyCollection([[(e[i], yy), (e[i + 1], yy), (e[i + 1], yy + h), (e[i], yy + h)]
                                          for i in range(len(lut))], facecolors=lut, edgecolors=lut, linewidths=0.3))
        ax.text(x, yy - 5, f"{label} ({unit})", color=th["ink"], fontsize=17, ha="left", va="bottom")
        for v, s in ((x, f"\u2212{r:.2g}"), (x + w / 2, "0"), (x + w, f"{r:.2g}")):
            ax.plot([v, v], [yy + h, yy + h + 4], color=th["ink"], lw=1.0)
            ax.text(v, yy + h + 6, s, color=th["ink"], fontsize=15, ha="center", va="top")


def draw_dA(ax, th, scene, stim, t, box=(560, 915, 1360, 1010)):
    x0, y0, x1, y1 = box
    ym = (y0 + y1) / 2
    amp = 14.0                                                            # deg at the box edge (A_MAX 12.5 inside)
    tx = lambda tt: x0 + (tt - CLIP_MS[0]) / (CLIP_MS[1] - CLIP_MS[0]) * (x1 - x0)   # noqa: E731
    ty = lambda v: ym - v / amp * (y1 - y0) / 2                                        # noqa: E731
    m = (scene.t_ms >= CLIP_MS[0]) & (scene.t_ms <= CLIP_MS[1])
    tt = scene.t_ms[m]
    ax.plot([x0, x1], [ym, ym], color=th["zero"], lw=0.8)
    ax.plot([tx(0), tx(0)], [y0, y1], color=th["zero"], lw=0.8)
    ax.plot(tx(tt), ty(scene.trace[f"target_{stim}"][m]), color=th["teacher"], lw=1.0, ls=(0, (4, 3)))
    k = tt <= t
    if k.any():
        cur = np.interp(t, scene.t_ms, scene.trace[f"dA_{stim}"])
        xs = np.r_[tx(tt[k]), tx(t)]
        ys = np.r_[ty(scene.trace[f"dA_{stim}"][m][k]), ty(cur)]
        ax.plot(xs, ys, color=AF.DIR_COL[stim], lw=1.8, solid_capstyle="round")
        ax.plot([xs[-1]], [ys[-1]], "o", color=AF.DIR_COL[stim], ms=4.5, mew=0)


def draw_card(ax, th, f):
    i, n = f["i"], f["n"]
    tsec = i / FPS
    alpha = min(1.0, tsec / FADE_S, (n / FPS - tsec) / FADE_S) if f["kind"] == "title" else min(1.0, tsec / FADE_S)
    if f["kind"] == "title":
        ax.text(W / 2, H / 2 - 20, TITLE, color=th["ink"], fontsize=46, ha="center", va="bottom", alpha=alpha)
        ax.text(W / 2, H / 2 + 30, SOURCES, color=th["ink2"], fontsize=22, ha="center", va="top", alpha=alpha)
    else:
        ax.text(W / 2, H / 2 - 120, END_LINE, color=th["ink"], fontsize=28, ha="center", va="bottom", alpha=alpha)
        for k, ln in enumerate(END_CREDITS):
            ax.text(W / 2, H / 2 - 40 + 40 * k, ln, color=th["ink2"], fontsize=21, ha="center", va="top", alpha=alpha)


def render_frame(scene, f, theme) -> np.ndarray:
    import cv2
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    th = THEMES[theme]
    fig, ax = _figure()
    ax.cla()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    fig.patch.set_facecolor(th["bg"])
    if f["kind"] != "clip":
        draw_card(ax, th, f)
    else:
        lut = lut_for(theme)
        for c in silhouettes(scene, f["az"]):
            ax.add_artist(PathPatch(MPath(np.r_[c, c[:1]], [1] + [2] * (len(c) - 1) + [79]),
                                    fc=to_rgba(th["fill"], th["fill_alpha"]), ec=th["edge"], lw=1.0))
        q = scene.camera.project(scene.P, f["az"])
        v = scene.value(f["stim"], f["t"])
        tn = np.clip(v / scene.rng, -1, 1)
        col = lut[np.round((tn + 1) / 2 * (AF.N_LUT - 1)).astype(int)]
        for k in np.argsort(np.abs(tn), kind="stable"):                       # weakest first
            a, b = scene.off[k], scene.off[k + 1]
            if b - a < 2:
                continue
            ax.add_artist(PathPatch(MPath(q[a:b], scene.codes[k]), fill=False, ec=col[k], lw=LW_PX[scene.role[k]],
                                    alpha=LINE_ALPHA, capstyle="round", joinstyle="round"))
        t = f["t"]
        sign = "+" if round(t) >= 0 else "\u2212"
        ax.text(W - 110, 70, f"{sign}{abs(round(t)):d} ms", color=th["ink"], fontsize=30, ha="right", va="top")
        draw_drum(ax, th, f["stim"], 160, 115)
        draw_colour_bars(ax, th, lut, scene.range)
        draw_dA(ax, th, scene, f["stim"], t)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


# ── outputs ─────────────────────────────────────────────────────────────────────────────────────────────────────────
_SCENE = {}


def _worker(job):
    import cv2
    f, theme, path = job
    t0 = time.time()
    img = render_frame(_SCENE["scene"], f, theme)
    cv2.imwrite(str(path), img[..., ::-1])
    return time.time() - t0


def render_frames(scene, jobs, threads=THREADS):
    import multiprocessing as mp
    _SCENE["scene"] = scene
    t0 = time.time()
    with mp.get_context("fork").Pool(threads) as pool:
        per = pool.map(_worker, jobs, chunksize=1)
    return dict(frames=len(jobs), wall_s=round(time.time() - t0, 1), per_frame_s=round(float(np.mean(per)), 2),
                wall_per_frame_s=round((time.time() - t0) / len(jobs), 3), workers=threads)


def encode(frame_dir, out_path):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-framerate", str(FPS),
           "-i", str(frame_dir / "%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out_path)]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return round(time.time() - t0, 1)


def guard(paths, overwrite):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise SystemExit(f"{existing} exist; pass --overwrite")


def step_stills(a):
    scene = Scene()
    names = [VID_DIR / f"still_{th}_{t:+05d}ms.png" for th in THEMES for t in STILL_MS]
    names += [VID_DIR / "card_title_dark.png", VID_DIR / "card_end_dark.png"]
    guard(names + [OUT / "stills.json"], a.overwrite)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(clip_frame("rot_right", t), th, VID_DIR / f"still_{th}_{t:+05d}ms.png") for th in THEMES for t in STILL_MS]
    sch = schedule()
    jobs += [(next(f for f in sch if f["kind"] == "title" and f["i"] == int(TITLE_S * FPS) // 2), "dark", names[-2]),
             (next(f for f in sch if f["kind"] == "end" and f["i"] == int(END_S * FPS) - 1), "dark", names[-1])]
    t0 = time.time()
    single = _single_timing(scene, jobs[0])
    info = dict(single_process_frame_s=single, **render_frames(scene, jobs), prep_s=round(t0 - _T0, 1),
                ranges=scene.range, camera=dict(focal_px=scene.camera.f, cx=scene.camera.cx, cy=scene.camera.cy,
                                                dist_um=DIST_UM, elev_deg=ELEV_DEG),
                palette={th: palette_check(th) for th in THEMES},
                still_frames={str(p.name): dict(t_ms=round(f.get("t", 0), 1), az_deg=round(f.get("az", 0), 1))
                        for f, _, p in jobs})
    (OUT / "stills.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _single_timing(scene, job):
    import cv2
    t0 = time.time()
    img = render_frame(scene, job[0], job[1])
    dt = time.time() - t0
    cv2.imwrite(str(OUT / "_timing.png"), img[..., ::-1])
    (OUT / "_timing.png").unlink()
    return round(dt, 2)


def _clip_jobs(frames, theme, frame_dir):
    return [(f, theme, frame_dir / f"{i:05d}.png") for i, f in enumerate(frames)]


def step_test(a):
    out = VID_DIR / "test_dark_3s.mp4"
    guard([out, OUT / "test.json"], a.overwrite)
    scene = Scene()
    clip = [f for f in schedule() if f["kind"] == "clip" and f["stim"] == "rot_right"]
    i0 = min(range(len(clip)), key=lambda i: abs(clip[i]["t"] - TEST_MS[0]))
    frames = clip[i0:i0 + int(3 * FPS)]
    fd = OUT / "frames_test"
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir(parents=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    r = render_frames(scene, _clip_jobs(frames, "dark", fd))
    enc = encode(fd, out)
    shutil.rmtree(fd)
    total = len(schedule())
    size = out.stat().st_size
    info = dict(**r, encode_s=enc, bytes=size, seconds=len(frames) / FPS, t_ms=[frames[0]["t"], frames[-1]["t"]],
                az_deg=[frames[0]["az"], frames[-1]["az"]], full_frames=total, full_seconds=total / FPS,
                projected_render_min=round(total * r["wall_per_frame_s"] / 60, 1),
                projected_bytes_upper=int(size / (len(frames) / FPS) * total / FPS))
    (OUT / "test.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def step_video(a):
    raise SystemExit("the full video is step 2 (not rendered yet)")


STEPS = {"data": step_data, "stills": step_stills, "test": step_test, "video": step_video}
_T0 = time.time()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "data":
        guard([OUT / f for f in FILES["data"]], a.overwrite)
    STEPS[a.step](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
