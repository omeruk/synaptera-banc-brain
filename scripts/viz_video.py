"""Visualisation (demonstration, no claim): a short video of the stage-A circuit. Origin class E (viewing only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_video.py [data|stills|video] [--overwrite]

Trials: the validation split only (configs/f5_splits.json, data seeds 160-199), rot_right and rot_left at 120 deg/s
(one speed, so dA and the teacher match): 80 trials through train/f5_stageA.load_trials (manifest SHA-256, split and
DN-column checks; it refuses test seeds and held-out speeds). CPU only.
- data: FlyWire DNs: per DN column and 5 ms interval the rate (Hz) from the stored dn_counts (untrained reference
  brain, arm R required in every record). Cord: the `true` training-seed-0 best.pt with the `true` bridge input as in
  training; the state of each drawn BANC neuron per interval, and dA. Per direction the mean over the 40 seeds; per
  neuron minus its own pre-stimulus baseline (mean over BASELINE_MS) and smoothed causally (exponential, TAU_MS).
  -> runs/viz/video/traces.npz, data.json
- stills: the check stills STILLS (act, ms). -> preview video folder, runs/viz/video/stills.json
- video: title card, act 1, a cross-fade, act 2, end card. -> VIDEO_NAME in the preview video folder, video.json
Act 1, "activity": rot_right, each neuron's change from its pre-stimulus baseline. Act 2, "direction": per neuron and
interval the trial-mean rot_right minus rot_left (the same causal smoothing); the dA trace shows both directions.
Each act runs CLIP_MS around stimulus onset, slowed SLOW x. Neurons, anatomy and neck split as
scripts/viz_activity_figure.py (circuit_lines); in addition the FlyWire DN segments outside the brain neuropil
outline are dropped (point-in-mesh by ray parity on a NECK_GRID_UM grid; a segment needs both ends inside). Colour:
the activity figure's diverging map with a dark-grey centre (dark theme only), per act and population one symmetric
range at the 95th percentile of |value| over its neurons and every frame of the act. Camera: fixed (open loop: the
fly does not turn), perspective from straight behind the fly (azimuth AZ_DEG, posterior), elevation ELEV_DEG, dorsal
up. Frames are drawn at 2x (3840 x 2160, matplotlib Agg) and downsampled (area); 1 pt = 1 px at 1080p. Encoding:
H.264 (libx264, yuv420p, CRF 18, +faststart), 30 fps, via the ffmpeg of imageio-ffmpeg. Outputs are never
overwritten without --overwrite (rule 13).
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
STILLS = ((1, 300), (2, 300), (2, 800))              # (act, ms from onset)
FPS, SLOW = 30, 10
TITLE_S, END_S, FADE_S, XFADE_S = 2.0, 3.0, 0.3, 0.5   # cards, card text fade, cross-fade between the acts
ELEV_DEG, AZ_DEG = 30.0, 90.0                        # azimuth 90: camera at +y, straight behind the fly
NECK_GRID_UM = 2.0
ACT_LABEL = {1: "change from baseline", 2: "right minus left"}
VIDEO_NAME = "stageA_circuit_video.mp4"
DIST_UM = 4500.0                                     # camera distance: mild perspective
RANGE_PCT = 95
W, H, SS = 1920, 1080, 2                             # output size, supersampling
VIEW_BOX = (110, 130, 1810, 880)                     # x0, y0, x1, y1 (px) for the circuit
LW_PX = {"fw_dn": 0.7, "banc_dn": 0.7, "interneuron": 0.8, "mn": 1.2}
LINE_ALPHA = 0.85
THREADS = 12
TITLE = "Visual turning signal from brain to wing motor neurons"
SOURCES = "Data: FlyWire v783 connectome (brain) and BANC v888 connectome (brain and nerve cord)"
END_LINES = ("Brain: untrained spiking reference model; only the nerve cord was trained.",
             "Open-loop simulation: the fly is held fixed; colours show model activity.")
END_CREDITS = (
    "FlyWire v783: Dorkenwald et al. 2024; Schlegel et al. 2024; EM: Zheng et al. 2018. CC BY-NC 4.0.",
    "BANC v888: Bates, Phelps, Kim, Yang et al. 2026, Nature 656:957-970. CC BY 4.0. doi:10.7910/DVN/7WTH1N.",
    "Skeletons, warped FlyWire skeletons and outlines: gs://lee-lab_brain-and-nerve-cord-fly-connectome.",
    "This video: CC BY-NC 4.0 (it combines both sources).")
TH = dict(bg="#0E0F11", fill="#24262A", fill_alpha=0.75, edge="#3E4146", mid="#4D4D4D", ink="#E8E8E8",
          ink2="#9A9A9A", teacher="#A8A8A8", zero="#4A4A4A")       # dark theme
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
def act_frame(act, i):
    n = n_clip()
    return dict(kind="clip", act=act, i=i, t=CLIP_MS[0] + i / (n - 1) * (CLIP_MS[1] - CLIP_MS[0]))


def schedule():
    """Every frame: title card, act 1, cross-fade (last act-1 frame into the first act-2 frame), act 2, end card."""
    fr = [dict(kind="title", i=i, n=int(TITLE_S * FPS)) for i in range(int(TITLE_S * FPS))]
    n, nf = n_clip(), int(XFADE_S * FPS)
    fr += [act_frame(1, i) for i in range(n)]
    fr += [dict(kind="fade", i=i, w=(i + 1) / (nf + 1)) for i in range(nf)]
    fr += [act_frame(2, i) for i in range(n)]
    fr += [dict(kind="end", i=i, n=int(END_S * FPS)) for i in range(int(END_S * FPS))]
    return fr


def still_frame(act, t):
    n = n_clip()
    return act_frame(act, int(round((t - CLIP_MS[0]) / (CLIP_MS[1] - CLIP_MS[0]) * (n - 1))))


class Camera:
    """Fixed perspective camera: azimuth AZ_DEG about the vertical through the circuit centre, elevation ELEV_DEG."""

    def __init__(self, centre, focal=1.0, cx=0.0, cy=0.0):
        self.c, self.f, self.cx, self.cy = np.asarray(centre, float), focal, cx, cy
        a, e = math.radians(AZ_DEG), math.radians(ELEV_DEG)
        self.pos = self.c + DIST_UM * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        self.fwd = (self.c - self.pos) / np.linalg.norm(self.c - self.pos)
        r = np.cross(self.fwd, [0.0, 0.0, 1.0])
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)

    def project(self, P):
        d = P - self.pos
        z = d @ self.fwd
        return np.c_[self.cx + self.f * (d @ self.right) / z, self.cy - self.f * (d @ self.up) / z]


def fit_camera(points, centre) -> Camera:
    """Focal length and screen offset so that the circuit fills VIEW_BOX."""
    q = Camera(centre).project(points)
    lo, hi = q.min(0), q.max(0)
    bx0, by0, bx1, by1 = VIEW_BOX
    f = min((bx1 - bx0) / (hi[0] - lo[0]), (by1 - by0) / (hi[1] - lo[1]))
    return Camera(centre, f, (bx0 + bx1) / 2 - f * (lo[0] + hi[0]) / 2, (by0 + by1) / 2 - f * (lo[1] + hi[1]) / 2)


def lut_dark():
    """The activity figure's diverging map (OKLab arms) with the dark theme's neutral centre."""
    lab = {k: AF._oklab(AF._lin(AF._hex(h))) for k, h in (("neg", AF.NEG), ("mid", TH["mid"]), ("pos", AF.POS))}
    t = np.linspace(-1, 1, AF.N_LUT)[:, None]
    arm = np.where(t < 0, lab["neg"], lab["pos"])
    return AF._srgb(AF._from_oklab(lab["mid"] + np.abs(t) * (arm - lab["mid"])))


def palette_check() -> dict:
    """The activity figure's check (mirror and centre-to-pole separation under simulated CVD), lightness monotone
    away from the centre, and the distance of centre and poles from the background."""
    lut = lut_dark()
    c = AF.diverging_check(lut)
    L = AF._oklab(AF._lin(lut))[:, 0]
    mid = AF.N_LUT // 2
    c["lightness"]["monotone"] = bool(np.all(np.diff(L[mid:]) >= 0) and np.all(np.diff(L[:mid + 1]) <= 0))
    c["lightness"]["direction_from_centre"] = "up"
    bg = AF._oklab(AF._lin(AF._hex(TH["bg"])))
    lab = AF._oklab(AF._lin(lut))

    def lum(rgb):
        lin = AF._lin(rgb)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])

    lb = lum(AF._hex(TH["bg"]))
    c["background"] = {k: dict(delta_e=round(float(100 * np.linalg.norm(lab[i] - bg)), 1),
                               contrast=round((max(lum(lut[i]), lb) + 0.05) / (min(lum(lut[i]), lb) + 0.05), 2))
                       for k, i in (("neg", 0), ("centre", mid), ("pos", -1))}
    return c


def inside_mesh(Vx, F, res=NECK_GRID_UM):
    """Test for points (um) inside a closed triangle mesh (um): parity of the crossings, left of the point, of the ray
    along x through the centre of the point's (y, z) grid cell (spacing res; centres offset so none falls on a grid
    line). Returns the test as a function of an (n, 3) array."""
    off = 0.3137 * res
    lo = Vx[:, 1:].min(0) - res
    n = np.ceil((Vx[:, 1:].max(0) + res - lo) / res).astype(int) + 1
    cells, xs = [], []
    for tri in Vx[F]:
        yz = tri[:, 1:]
        i0 = np.floor((yz.min(0) - lo - off) / res).astype(int)
        i1 = np.ceil((yz.max(0) - lo - off) / res).astype(int)
        iy, iz = np.meshgrid(np.arange(i0[0], i1[0] + 1), np.arange(i0[1], i1[1] + 1), indexing="ij")
        c = np.c_[lo[0] + off + iy.ravel() * res, lo[1] + off + iz.ravel() * res]
        v0, v1, v2 = yz[1] - yz[0], yz[2] - yz[0], c - yz[0]
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-12:
            continue
        b1 = (d11 * (v2 @ v0) - d01 * (v2 @ v1)) / den
        b2 = (d00 * (v2 @ v1) - d01 * (v2 @ v0)) / den
        m = (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
        cells.append((iy.ravel() * n[1] + iz.ravel())[m])
        xs.append((tri[0, 0] * (1 - b1 - b2) + tri[1, 0] * b1 + tri[2, 0] * b2)[m])
    cells, xs = np.concatenate(cells), np.concatenate(xs)
    x0, big = xs.min() - 1e4, xs.max() - xs.min() + 2e4                 # x - x0 in (0, big) for any point nearby
    key = np.sort(cells * big + (xs - x0))

    def test(pts):
        ip = np.round((pts[:, 1:] - lo - off) / res).astype(int)
        ok = (ip >= 0).all(1) & (ip < n).all(1)
        cp = ip[:, 0] * n[1] + ip[:, 1]
        xr = np.clip(pts[:, 0] - x0, 0, big - 1)
        cnt = np.searchsorted(key, cp * big + xr) - np.searchsorted(key, cp * big)
        return ok & (cnt % 2 == 1)

    return test


class Scene:
    """Geometry, the per-act traces and ranges, and the fixed camera; shared by the frame workers."""

    def __init__(self):
        import pandas as pd
        fetch = json.loads((ANAT / "fetch.json").read_text())
        meshes, y_cut = V.outline_meshes(fetch)
        self.meshes = [(Vx / 1e3, F) for Vx, F in meshes.values()]
        lines, _ = AF.circuit_lines(y_cut)
        brain_V, brain_F = self.meshes[0]
        inside = inside_mesh(brain_V, brain_F)
        before = after = touched = 0
        verts, self.codes = [], []
        for role, _, _, (p, st) in lines:
            if role == "fw_dn" and len(p):                          # neck cleanup: FlyWire DNs inside the brain outline
                ins = inside(p)
                q, s2 = V.runs(p, st, ins & np.r_[False, ins[:-1]])
                before, after, touched = before + len(p), after + len(q), touched + int(len(q) < len(p))
                p, st = q, s2
            verts.append(p)
            self.codes.append(np.where(st, 1, 2).astype(np.uint8))
        self.neck = dict(fw_dn_vertices_before=before, after=after, dropped=before - after,
                         dropped_fraction=round((before - after) / before, 4), neurons_with_drops=touched,
                         fw_dn_inside_fraction=round(after / before, 4),
                         cord_vertices_inside_brain_fraction=round(float(np.mean(inside(np.concatenate(
                             [p for r, _, _, (p, _) in lines if r in ("interneuron", "mn") and len(p)])))), 4))
        self.role = np.array([r for r, _, _, _ in lines])
        self.node = np.array([nd for _, _, nd, _ in lines])
        self.off = np.r_[0, np.cumsum([len(p) for p in verts])]
        self.P = np.concatenate(verts)
        tr = dict(np.load(OUT / "traces.npz"))
        self.t_ms = tr["t_ms"]
        circ = pd.read_csv(ANAT / "circuit.csv")
        if not np.array_equal(np.sort(tr["cord_nodes"]), np.sort(circ.loc[circ["role"] != "fw_dn", "node"])):
            raise SystemExit("traces.npz and the circuit differ")
        col = {int(b): k for k, b in enumerate(tr["fw_index"])}
        cpos = {int(nd): k for k, nd in enumerate(tr["cord_nodes"])}
        self.fw = self.role == "fw_dn"
        fi = [col[int(nd)] for nd in self.node[self.fw]]
        ci = [cpos[int(nd)] for nd in self.node[~self.fw]]

        def per_neuron(fw, cord):
            x = np.zeros((len(self.t_ms), len(lines)))
            x[:, self.fw], x[:, ~self.fw] = fw[:, fi], cord[:, ci]
            return x

        ms = float(tr["interval_ms"])
        self.trace = {1: per_neuron(tr["fw_rot_right"], tr["cord_rot_right"]),
                      2: per_neuron(smooth(tr["fw_raw_rot_right"].astype(float) - tr["fw_raw_rot_left"], ms),
                                    smooth(tr["cord_raw_rot_right"].astype(float) - tr["cord_raw_rot_left"], ms))}
        self.dA = {s: tr[f"dA_{s}"] for s in STIMULI}
        self.target = {s: tr[f"target_{s}"] for s in STIMULI}
        pre = self.t_ms <= 0
        self.act2_pre_max = {"brain": float(np.abs(self.trace[2][pre][:, self.fw]).max()),
                             "cord": float(np.abs(self.trace[2][pre][:, ~self.fw]).max())}
        self.range = {}
        for act in (1, 2):
            v = np.stack([self.value(act, f["t"]) for f in schedule() if f["kind"] == "clip" and f["act"] == act])
            self.range[act] = {"brain": float(np.percentile(np.abs(v[:, self.fw]), RANGE_PCT)),
                               "cord": float(np.percentile(np.abs(v[:, ~self.fw]), RANGE_PCT))}
        self.rng = {act: np.where(self.fw, r["brain"], r["cord"]) for act, r in self.range.items()}
        sample = np.concatenate([self.P[::40]] + [Vx for Vx, _ in self.meshes])
        self.camera = fit_camera(sample, sample.mean(0))
        self.silhouettes = silhouettes(self)

    def value(self, act, t):
        """Per neuron, the act's trace linearly interpolated at trial time t (ms)."""
        x = self.trace[act]
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


def silhouettes(scene):
    """Outer contours (1080p px) of the projected brain and VNC outlines, rasterised at the 2x size (fixed camera)."""
    import cv2
    out = []
    for Vx, F in scene.meshes:
        q = np.round(scene.camera.project(Vx) * SS).astype(np.int32)
        img = np.zeros((H * SS, W * SS), np.uint8)
        cv2.fillPoly(img, list(q[F]), 255)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out += [cv2.approxPolyDP(c, 1.0, True)[:, 0, :] / SS for c in cs if cv2.contourArea(c) > 50]
    return out


def _arrow(ax, x, y, ra, a0, a1, col):
    from matplotlib.patches import Polygon
    a = np.linspace(a0, a1, 40)
    ax.plot(x + ra * np.cos(a), y - ra * np.sin(a), color=col, lw=2.2, solid_capstyle="round")
    tip = np.array([x + ra * math.cos(a1), y - ra * math.sin(a1)])
    tang = np.array([-math.sin(a1), -math.cos(a1)]) * (-1 if a1 < a0 else 1)     # motion direction at the tip
    nrm = np.array([-tang[1], tang[0]])
    ax.add_artist(Polygon([tip + 9 * tang, tip - 3 * tang + 6 * nrm, tip - 3 * tang - 6 * nrm], closed=True, fc=col,
                          ec="none"))


def draw_drum(ax, act, x, y, r=34):
    """Drum seen from above, the fly in the middle (anterior up). Act 1: the rot_right arrow (pattern in front of the
    fly moving to its right, clockwise from above); act 2: both arrows from the front, right (vermillion) and left
    (blue)."""
    from matplotlib.patches import Circle, Polygon
    ax.add_artist(Circle((x, y), r, fill=False, ec=TH["ink"], lw=1.4))
    for k in range(12):
        a = 2 * math.pi * k / 12
        ax.plot([x + 0.78 * r * math.cos(a), x + r * math.cos(a)], [y + 0.78 * r * math.sin(a), y + r * math.sin(a)],
                color=TH["ink"], lw=1.4, solid_capstyle="butt")
    ax.add_artist(Polygon([(x, y - 9), (x - 6, y + 7), (x + 6, y + 7)], closed=True, fc=TH["ink"], ec="none"))
    ra = r + 9
    if act == 1:
        _arrow(ax, x, y, ra, math.radians(130), math.radians(50), AF.DIR_COL["rot_right"])
    else:
        _arrow(ax, x, y, ra, math.radians(95), math.radians(35), AF.DIR_COL["rot_right"])
        _arrow(ax, x, y, ra, math.radians(85), math.radians(145), AF.DIR_COL["rot_left"])


def draw_colour_bars(ax, lut, ranges, label, x=130, y=905):
    from matplotlib.collections import PolyCollection
    w, h = 190, 9
    ax.text(x, y - 34, label, color=TH["ink2"], fontsize=16, ha="left", va="bottom")
    for k, (name, unit, r) in enumerate((("FlyWire DNs", "Hz", ranges["brain"]),
                                         ("BANC neurons", "a.u.", ranges["cord"]))):
        yy = y + 62 * k
        e = np.linspace(x, x + w, len(lut) + 1)
        ax.add_collection(PolyCollection([[(e[i], yy), (e[i + 1], yy), (e[i + 1], yy + h), (e[i], yy + h)]
                                          for i in range(len(lut))], facecolors=lut, edgecolors=lut, linewidths=0.3))
        ax.text(x, yy - 5, f"{name} ({unit})", color=TH["ink"], fontsize=17, ha="left", va="bottom")
        for v, s in ((x, f"−{r:.2g}"), (x + w / 2, "0"), (x + w, f"{r:.2g}")):
            ax.plot([v, v], [yy + h, yy + h + 4], color=TH["ink"], lw=1.0)
            ax.text(v, yy + h + 6, s, color=TH["ink"], fontsize=15, ha="center", va="top")


def draw_dA(ax, scene, act, t, box=(560, 915, 1360, 1010)):
    """dA up to the current time (act 1: rot_right; act 2: both directions), the teacher dA* dashed."""
    x0, y0, x1, y1 = box
    ym = (y0 + y1) / 2
    amp = 14.0                                                            # deg at the box edge (A_MAX 12.5 inside)
    tx = lambda tt: x0 + (tt - CLIP_MS[0]) / (CLIP_MS[1] - CLIP_MS[0]) * (x1 - x0)   # noqa: E731
    ty = lambda v: ym - v / amp * (y1 - y0) / 2                                        # noqa: E731
    m = (scene.t_ms >= CLIP_MS[0]) & (scene.t_ms <= CLIP_MS[1])
    tt = scene.t_ms[m]
    ax.plot([x0, x1], [ym, ym], color=TH["zero"], lw=0.8)
    ax.plot([tx(0), tx(0)], [y0, y1], color=TH["zero"], lw=0.8)
    for s in (("rot_right",) if act == 1 else STIMULI):
        ax.plot(tx(tt), ty(scene.target[s][m]), color=TH["teacher"], lw=1.0, ls=(0, (4, 3)))
        k = tt <= t
        if k.any():
            cur = np.interp(t, scene.t_ms, scene.dA[s])
            xs, ys = np.r_[tx(tt[k]), tx(t)], np.r_[ty(scene.dA[s][m][k]), ty(cur)]
            ax.plot(xs, ys, color=AF.DIR_COL[s], lw=1.8, solid_capstyle="round")
            ax.plot([xs[-1]], [ys[-1]], "o", color=AF.DIR_COL[s], ms=4.5, mew=0)


def draw_card(ax, f):
    tsec, n = f["i"] / FPS, f["n"] / FPS
    if f["kind"] == "title":
        alpha = min(1.0, tsec / FADE_S, (n - tsec) / FADE_S)
        ax.text(W / 2, H / 2 - 20, TITLE, color=TH["ink"], fontsize=46, ha="center", va="bottom", alpha=alpha)
        ax.text(W / 2, H / 2 + 30, SOURCES, color=TH["ink2"], fontsize=22, ha="center", va="top", alpha=alpha)
    else:
        alpha = min(1.0, tsec / FADE_S)
        for k, ln in enumerate(END_LINES):
            ax.text(W / 2, H / 2 - 150 + 44 * k, ln, color=TH["ink"], fontsize=28, ha="center", va="bottom",
                    alpha=alpha)
        for k, ln in enumerate(END_CREDITS):
            ax.text(W / 2, H / 2 - 20 + 40 * k, ln, color=TH["ink2"], fontsize=21, ha="center", va="top", alpha=alpha)


def render_frame(scene, f) -> np.ndarray:
    import cv2
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    if f["kind"] == "fade":                                               # cross-fade between the two acts
        n = n_clip()
        a = render_frame(scene, act_frame(1, n - 1)).astype(float)
        b = render_frame(scene, act_frame(2, 0)).astype(float)
        return np.round((1 - f["w"]) * a + f["w"] * b).astype(np.uint8)
    fig, ax = _figure()
    ax.cla()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    fig.patch.set_facecolor(TH["bg"])
    if f["kind"] != "clip":
        draw_card(ax, f)
    else:
        act, t = f["act"], f["t"]
        lut = lut_dark()
        for c in scene.silhouettes:
            ax.add_artist(PathPatch(MPath(np.r_[c, c[:1]], [1] + [2] * (len(c) - 1) + [79]),
                                    fc=to_rgba(TH["fill"], TH["fill_alpha"]), ec=TH["edge"], lw=1.0))
        q = scene.camera.project(scene.P)
        tn = np.clip(scene.value(act, t) / scene.rng[act], -1, 1)
        col = lut[np.round((tn + 1) / 2 * (AF.N_LUT - 1)).astype(int)]
        for k in np.argsort(np.abs(tn), kind="stable"):                       # weakest first
            a, b = scene.off[k], scene.off[k + 1]
            if b - a < 2:
                continue
            ax.add_artist(PathPatch(MPath(q[a:b], scene.codes[k]), fill=False, ec=col[k], lw=LW_PX[scene.role[k]],
                                    alpha=LINE_ALPHA, capstyle="round", joinstyle="round"))
        sign = "+" if round(t) >= 0 else "−"
        ax.text(W - 110, 70, f"{sign}{abs(round(t)):d} ms", color=TH["ink"], fontsize=30, ha="right", va="top")
        draw_drum(ax, act, 160, 115)
        draw_colour_bars(ax, lut, scene.range[act], ACT_LABEL[act])
        draw_dA(ax, scene, act, t)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


# ── outputs ─────────────────────────────────────────────────────────────────────────────────────────────────────────
_SCENE = {}


def _worker(job):
    import cv2
    f, path = job
    t0 = time.time()
    cv2.imwrite(str(path), render_frame(_SCENE["scene"], f)[..., ::-1])
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


def scene_info(scene) -> dict:
    return dict(neck_cleanup=scene.neck, ranges=scene.range, act2_pre_stimulus_max_abs=scene.act2_pre_max,
                camera=dict(az_deg=AZ_DEG, elev_deg=ELEV_DEG, dist_um=DIST_UM, focal_px=scene.camera.f),
                palette=palette_check())


def step_stills(a):
    names = [VID_DIR / f"still_act{act}_{t:+05d}ms.png" for act, t in STILLS]
    guard(names + [OUT / "stills.json"], a.overwrite)
    t0 = time.time()
    scene = Scene()
    prep = time.time() - t0
    VID_DIR.mkdir(parents=True, exist_ok=True)
    r = render_frames(scene, [(still_frame(act, t), p) for (act, t), p in zip(STILLS, names)])
    info = dict(prep_s=round(prep, 1), **r, **scene_info(scene))
    (OUT / "stills.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def step_video(a):
    out = VID_DIR / VIDEO_NAME
    guard([out, OUT / "video.json"], a.overwrite)
    t0 = time.time()
    scene = Scene()
    prep = time.time() - t0
    frames = schedule()
    fd = OUT / "frames_video"
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir(parents=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    r = render_frames(scene, [(f, fd / f"{i:05d}.png") for i, f in enumerate(frames)])
    enc = encode(fd, out)
    shutil.rmtree(fd)
    kinds = {k: sum(f["kind"] == k for f in frames) for k in ("title", "clip", "fade", "end")}
    info = dict(file=str(out), bytes=out.stat().st_size, frames=len(frames), seconds=round(len(frames) / FPS, 2),
                frames_by_kind=kinds, prep_s=round(prep, 1), render=r, encode_s=enc,
                total_s=round(time.time() - t0, 1), **scene_info(scene))
    (OUT / "video.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


STEPS = {"data": step_data, "stills": step_stills, "video": step_video}


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
