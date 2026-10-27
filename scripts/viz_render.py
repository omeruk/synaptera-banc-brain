"""Visualisation scaffold: 3D video frames of one trial, brain spikes and VNC model activity (demonstration only; no
claim, no prereg). CPU only; numpy perspective projection, OpenCV blur for the glow, PIL for text and insets.

    env -u PYTHONPATH python scripts/viz_render.py stills [--times -100,100,500] [--out DIR]
    env -u PYTHONPATH python scripts/viz_render.py video [--out FILE]            # step 4b, not run yet

Reads only runs/viz/ (viz_positions.py, viz_activity.py) and the stage-A setup / bridge (highlight sets). Trial:
rot_right, 60 deg/s, validation seed 160; VNC: the `true` training-seed-0 best.pt.

BANC axes as read from the data (viz step 4a): +z dorsal (BANC DNg somata, in the ventral GNG, at z ~65 um; DNa / DNp at
~170-194 um; the affine sends FAFB +y, ventral, to BANC -z), +y posterior (DNa anterior of DNp), +x the fly's left
(BANC neurons labelled left at mean x ~703 um, right ~271 um).

Timing: one frame per 5 ms interval (FRAMES_PER_INTERVAL), FPS 30, so the 1.5 s trial plays in 10 s (6.7x slower).
Interval j covers [(j - N_GREY) * 5, (j - N_GREY + 1) * 5) ms from stimulus onset; its frame shows the clock at the
interval's end. Brain: spike counts with an exponential fade (TAU_MS). VNC: model state minus the mean over -200..0 ms,
one fixed range (the 99th percentile of |change| over the whole trial). Frames are PNGs; never committed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RUNS = ROOT / "runs" / "viz"
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/viz")     # machine 2: viewing in Windows
STIM, ARM, TRANSFORM = "rot_right", "true", "affine"

W, H = 1920, 1080
VIEW_W = 1380                                    # 3D view: x < VIEW_W; insets to the right
BG = np.array([5, 7, 13], np.float32) / 255      # #05070d
FPS, FRAMES_PER_INTERVAL = 30, 1
INTERVAL_MS, N_GREY, N_INT = 5.0, 100, 300
TAU_MS = 25.0
BASELINE_MS = (-200.0, 0.0)
RANGE_PCT = 99.0
ORBIT_DEG, AZ0_DEG, EL_DEG = 40.0, -70.0, 22.0   # azimuth from the fly's front, + towards the fly's left: front-right
                                                 # oblique (a pure front view hides the nerve cord behind the brain)
CAM_DIST_UM = 3200.0
FRONT, LEFT, UP = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

HAZE = np.array([0.55, 0.58, 0.66], np.float32)
SPIKE = np.array([0.55, 0.90, 1.00], np.float32)          # bright cool
INC = np.array([1.00, 0.55, 0.18], np.float32)            # increase: warm orange
DEC = np.array([0.22, 0.48, 1.00], np.float32)            # decrease: cool blue
MARK = np.array([0.85, 0.85, 0.90], np.float32)
W_HAZE_BRAIN, W_HAZE_VNC, W_SPIKE, W_VNC, W_MARK = 0.16, 0.20, 1.1, 1.8, 0.10
DA_COL, TEACH_COL, LEFT_COL, RIGHT_COL = (126, 224, 195), (150, 158, 176), (185, 166, 255), (232, 232, 236)
INK, INK2, PANEL, LINE = (236, 238, 243), (150, 156, 170), (13, 17, 28), (44, 50, 66)
FOOTER = ("Data: FlyWire v783, BANC. Approximate alignment (affine, about 40 um). "
          "Open-loop visualisation, not an experiment.")
FONT = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def t_ms(j):
    """Clock of interval j's frame: the end of the interval, ms from stimulus onset."""
    return (j - N_GREY + 1) * INTERVAL_MS


def interval_at(ms):
    return int(round(N_GREY + ms / INTERVAL_MS - 1))


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Scene:
    def __init__(self, runs=RUNS):
        from f5_train import load_setup
        from train import f5_stageA as A
        tr = json.loads((runs / "transforms.json").read_text())["transforms"][TRANSFORM]
        M, t = np.array(tr["matrix"]), np.array(tr["translation"])
        pb, pv = np.load(runs / "positions_brain.npz"), np.load(runs / "positions_vnc.npz")
        self.brain = ((pb["xyz_nm"] @ M.T + t) / 1e3).astype(np.float32)
        kept = pv["kept"]
        self.vnc_kept = np.flatnonzero(kept)
        self.vnc = (pv["xyz_nm"][kept] / 1e3).astype(np.float32)
        b = np.load(runs / f"activity_brain_{STIM}.npz")
        self.counts = b["counts"].astype(np.float32)                                # (300, n_brain)
        v = np.load(runs / "activity_vnc.npz")
        col = [str(s) for s in v["stimuli"]].index(STIM)
        state = v[f"rates_hz_{ARM}"][:, col].astype(np.float64) / float(v["r_in_hz"])     # model state (a.u.)
        lo, hi = interval_at(BASELINE_MS[0]) + 1, interval_at(BASELINE_MS[1]) + 1         # intervals -200..0 ms
        self.delta = (state - state[lo:hi].mean(0)).astype(np.float32)
        self.range = float(np.percentile(np.abs(self.delta), RANGE_PCT))
        self.dA, self.target = v[f"dA_{ARM}"][:, col], v["target"][:, col]
        self.readout = np.asarray(v["readout"])
        setup = load_setup()
        bridge = A.make_bridge(setup, "true", 0)
        fw = np.unique(np.concatenate([setup.fw_index[np.asarray(u["fw_pos"])] for u in bridge.units]))
        bc = np.unique(np.concatenate([np.atleast_1d(u["banc_index"]) for u in bridge.units]))
        if not np.isin(bc, setup.banc_index).all() or not kept[bc].all() or not kept[self.readout].all():
            raise SystemExit("a bridged DN or steering MN has no position")
        self.mn_names = json.loads((A.OUT / "setup.json").read_text())["readout_names"]
        pos = np.full(len(kept), -1)
        pos[self.vnc_kept] = np.arange(len(self.vnc_kept))
        self.hl_brain, self.hl_vnc, self.mn_pos = fw, pos[bc], pos[self.readout]
        self.n_units = len(bridge.units)
        self.mn_delta = self.delta[:, self.readout]
        self.mn_range = float(np.abs(self.mn_delta).max())
        self.dvnc = self.delta[:, self.vnc_kept]
        self.centre = np.concatenate([self.brain, self.vnc]).mean(0)
        self.brain_c, self.vnc_c = np.median(self.brain, 0), np.median(self.vnc, 0)
        self.phase_deg = float(b["phase_deg"])
        self.speed = float(b["speed_deg_s"])

    def glow(self, j, prev=None):
        """Faded spike count at interval j: g_j = g_{j-1} exp(-dt / tau) + c_j (from zero at the trial start)."""
        d = np.float32(np.exp(-INTERVAL_MS / TAU_MS))
        if prev is not None:
            return prev * d + self.counts[j]
        g = np.zeros(self.counts.shape[1], np.float32)
        for k in range(j + 1):
            g = g * d + self.counts[k]
        return g


# ── camera ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, az_deg, centre, focal, cx, cy):
        az, el = np.radians(az_deg), np.radians(EL_DEG)
        d = np.cos(el) * (np.cos(az) * FRONT + np.sin(az) * LEFT) + np.sin(el) * UP
        self.pos = centre + CAM_DIST_UM * d
        self.f = -d
        self.r = np.cross(self.f, UP)
        self.r /= np.linalg.norm(self.r)
        self.u = np.cross(self.r, self.f)
        self.focal, self.cx, self.cy = focal, cx, cy

    def project(self, X):
        p = X - self.pos
        z = p @ self.f
        return self.cx + self.focal * (p @ self.r) / z, self.cy - self.focal * (p @ self.u) / z


def az_of(frame, n_frames):
    return AZ0_DEG + ORBIT_DEG * frame / max(n_frames - 1, 1)


def fit_focal(scene, n_frames):
    """One focal length and image centre for the whole orbit: every point inside the view with a margin."""
    X = np.concatenate([scene.brain[::7], scene.vnc[::3]])
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for az in np.linspace(az_of(0, n_frames), az_of(n_frames - 1, n_frames), 9):
        x, y = Camera(az, scene.centre, 1.0, 0.0, 0.0).project(X)
        lo, hi = np.minimum(lo, [x.min(), y.min()]), np.maximum(hi, [x.max(), y.max()])
    focal = min((VIEW_W - 80) / (hi[0] - lo[0]), (H - 330) / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    return focal, VIEW_W / 2 - focal * mid[0], H / 2 + 5 - focal * mid[1]


# ── drawing ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def splat(buf, x, y, w):
    ix, iy = np.round(x).astype(np.int64), np.round(y).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    idx = iy[ok] * W + ix[ok]
    for c in range(3):
        buf[..., c] += np.bincount(idx, w[ok, c], minlength=W * H).reshape(H, W)


def colours_vnc(dv, rng):
    v = np.clip(dv / rng, -1.0, 1.0)[:, None]
    return W_VNC * (np.maximum(v, 0) * INC + np.maximum(-v, 0) * DEC)


def render_3d(scene, cam, g, j):
    haze, act, mark = (np.zeros((H, W, 3), np.float32) for _ in range(3))
    bx, by = cam.project(scene.brain)
    vx, vy = cam.project(scene.vnc)
    splat(haze, bx, by, np.broadcast_to(HAZE * W_HAZE_BRAIN, (len(bx), 3)))
    splat(haze, vx, vy, np.broadcast_to(HAZE * W_HAZE_VNC, (len(vx), 3)))
    gb = np.minimum(g, 4.0)[:, None] * SPIKE * W_SPIKE
    splat(act, bx, by, gb)
    cv = colours_vnc(scene.dvnc[j], scene.range)
    splat(act, vx, vy, cv)
    hb, hv = scene.hl_brain, np.concatenate([scene.hl_vnc, scene.mn_pos])
    splat(mark, bx[hb], by[hb], MARK * W_MARK + 2.5 * gb[hb])                     # larger markers: blurred wider
    splat(mark, vx[hv], vy[hv], MARK * W_MARK + 2.5 * cv[hv])
    acc = (cv2.GaussianBlur(haze, (0, 0), 0.8)
           + cv2.GaussianBlur(act, (0, 0), 0.9) + 0.5 * cv2.GaussianBlur(act, (0, 0), 4.0)
           + 0.25 * cv2.GaussianBlur(act, (0, 0), 12.0)
           + 6.0 * cv2.GaussianBlur(mark, (0, 0), 2.2) + 0.6 * cv2.GaussianBlur(mark, (0, 0), 8.0))
    img = BG + (1.0 - BG) * (1.0 - np.exp(-acc))
    img[:, VIEW_W:] = BG
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def text(d, xy, s, size=18, fill=INK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def panel(d, box, title):
    d.rounded_rectangle(box, 10, fill=PANEL, outline=LINE)
    text(d, (box[0] + 16, box[1] + 12), title, 17, INK, True)


def stimulus_panel(d, scene, j, box):
    panel(d, box, "Stimulus: grating drum, seen from above")
    cx, cy, R, w = (box[0] + box[2]) // 2 - 70, box[1] + 175, 105, 20
    ms = t_ms(j)
    on = ms > 0
    if on:
        rot = scene.phase_deg + scene.speed * ms / 1e3      # rot_right: clockwise from above (moves to the fly's right)
        for k in range(24):                                  # 12 cycles of a 30 deg wavelength
            a0 = -90 + rot + 15 * k
            d.arc((cx - R, cy - R, cx + R, cy + R), a0, a0 + 15, fill=(220, 224, 232) if k % 2 else (40, 44, 56),
                  width=w)
        a0, a1 = -125, -55                                   # arrow along the front of the drum, to the fly's right
        Ra = R + 22
        d.arc((cx - Ra, cy - Ra, cx + Ra, cy + Ra), a0, a1, fill=INK, width=3)
        ex, ey = cx + Ra * np.cos(np.radians(a1)), cy + Ra * np.sin(np.radians(a1))
        tang = np.radians(a1 + 90)
        for s in (-1, 1):
            hx = ex - 13 * np.cos(tang) + s * 7 * np.cos(tang + np.pi / 2)
            hy = ey - 13 * np.sin(tang) + s * 7 * np.sin(tang + np.pi / 2)
            d.line((ex, ey, hx, hy), fill=INK, width=3)
    else:
        d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(120, 124, 134), width=w)
    d.ellipse((cx - 9, cy - 46, cx + 9, cy - 28), fill=(200, 204, 214))          # head (anterior up)
    d.ellipse((cx - 14, cy - 30, cx + 14, cy + 40), fill=(150, 156, 170))        # body
    text(d, (cx - 34, cy), "L", 16, INK2, anchor="mm")
    text(d, (cx + 34, cy), "R", 16, INK2, anchor="mm")
    tx = box[0] + 300
    text(d, (tx, box[1] + 70), f"t = {ms:+.0f} ms", 27, INK, True)
    text(d, (tx, box[1] + 112), "from stimulus onset", 14, INK2)
    if on:
        text(d, (tx, box[1] + 160), "grating moves to", 15, INK)
        text(d, (tx, box[1] + 180), "the fly's right", 15, INK)
        text(d, (tx, box[1] + 204), f"{scene.speed:.0f} deg/s", 15, INK2)
    else:
        text(d, (tx, box[1] + 160), "grey lead-in", 15, INK)
        text(d, (tx, box[1] + 180), "(uniform drum)", 15, INK2)


def axis_box(box, pad=(58, 46, 16, 40)):
    return box[0] + pad[0], box[1] + pad[1], box[2] - pad[2], box[3] - pad[3]


def dA_panel(d, scene, j, box):
    panel(d, box, "Wing-amplitude difference dA (deg)")
    x0, y0, x1, y1 = axis_box(box)
    tmin, tmax, ymax = -500.0, 1000.0, 12.5
    X = lambda t: x0 + (t - tmin) / (tmax - tmin) * (x1 - x0)                  # noqa: E731
    Y = lambda v: y0 + (ymax - v) / (2 * ymax) * (y1 - y0)                     # noqa: E731
    for v in (-10, -5, 0, 5, 10):
        d.line((x0, Y(v), x1, Y(v)), fill=LINE if v else (80, 86, 102), width=1)
        text(d, (x0 - 8, Y(v)), f"{v:+d}" if v else "0", 13, INK2, anchor="rm")
    for t in (-500, 0, 500, 1000):
        text(d, (X(t), y1 + 8), f"{t}", 13, INK2, anchor="ma")
    text(d, ((x0 + x1) / 2, y1 + 24), "ms from stimulus onset", 13, INK2, anchor="ma")
    tt = t_ms(np.arange(j + 1))
    teach = [(X(a), Y(b)) for a, b in zip(tt, scene.target[:j + 1])]
    for k in range(0, len(teach) - 1, 2):                                       # dashed
        d.line(teach[k:k + 2], fill=TEACH_COL, width=2)
    d.line([(X(a), Y(b)) for a, b in zip(tt, scene.dA[:j + 1])], fill=DA_COL, width=3)
    d.line((X(tt[-1]), y0, X(tt[-1]), y1), fill=(90, 96, 112), width=1)
    lx = x0 + 10
    d.line((lx, y0 + 10, lx + 26, y0 + 10), fill=DA_COL, width=3)
    text(d, (lx + 34, y0 + 10), "true model", 13, INK, anchor="lm")
    d.line((lx + 130, y0 + 10, lx + 140, y0 + 10), fill=TEACH_COL, width=2)
    d.line((lx + 146, y0 + 10, lx + 156, y0 + 10), fill=TEACH_COL, width=2)
    text(d, (lx + 164, y0 + 10), "teacher dA*", 13, INK, anchor="lm")


def mn_panel(d, scene, j, box):
    panel(d, box, "16 steering MNs, change from baseline")
    x0, y0, x1, y1 = axis_box(box, (58, 70, 16, 36))
    types = [n.split("|")[0] for n in scene.mn_names[::2]]
    r = scene.mn_range
    Y = lambda v: y0 + (r - v) / (2 * r) * (y1 - y0)                          # noqa: E731
    d.line((x0, Y(0), x1, Y(0)), fill=(80, 86, 102), width=1)
    text(d, (x0 - 8, Y(r)), f"+{r:.2f}", 12, INK2, anchor="rm")
    text(d, (x0 - 8, Y(-r)), f"-{r:.2f}", 12, INK2, anchor="rm")
    text(d, (x0 - 8, Y(0)), "0", 12, INK2, anchor="rm")
    gw = (x1 - x0) / len(types)
    for k, tname in enumerate(types):
        for s, colr in ((0, LEFT_COL), (1, RIGHT_COL)):
            v = float(scene.mn_delta[j, 2 * k + s])
            bx0 = x0 + k * gw + 6 + s * (gw - 12) / 2
            bx1 = bx0 + (gw - 12) / 2 - 3
            d.rectangle((bx0, min(Y(v), Y(0)), bx1, max(Y(v), Y(0))), fill=colr)
        text(d, (x0 + (k + 0.5) * gw, y1 + 8), tname, 13, INK2, anchor="ma")
    lx, ly = box[0] + 18, box[1] + 44
    for lab, colr, off in (("left", LEFT_COL, 0), ("right", RIGHT_COL, 80)):
        d.rectangle((lx + off, ly - 6, lx + off + 14, ly + 6), fill=colr)
        text(d, (lx + off + 20, ly), lab, 13, INK, anchor="lm")
    text(d, (box[2] - 16, ly), "model activity (a.u.)", 13, INK2, anchor="rm")


def legend(d, scene, x, y):
    s = tuple(int(c * 255) for c in SPIKE)
    d.ellipse((x, y - 6, x + 12, y + 6), fill=s)
    text(d, (x + 20, y), f"brain spike (fades, tau {TAU_MS:.0f} ms)", 14, INK, anchor="lm")
    y += 30
    bw = 220
    for k in range(bw):
        v = 2 * k / (bw - 1) - 1
        c = np.maximum(v, 0) * INC + np.maximum(-v, 0) * DEC
        c = HAZE * 0.35 * (1 - abs(v)) + c * abs(v)
        d.line((x + k, y - 7, x + k, y + 7), fill=tuple(int(255 * min(1, q)) for q in c))
    text(d, (x, y + 12), f"-{scene.range:.3f}", 12, INK2)
    text(d, (x + bw / 2, y + 12), "0", 12, INK2, anchor="ma")
    text(d, (x + bw, y + 12), f"+{scene.range:.3f}", 12, INK2, anchor="ra")
    text(d, (x + bw + 14, y), "model activity (a.u.), nerve cord:", 14, INK, anchor="lm")
    text(d, (x + bw + 14, y + 18), "change from the -200..0 ms mean", 12, INK2, anchor="lm")
    y += 46
    d.ellipse((x, y - 7, x + 14, y + 7), outline=(220, 220, 230), width=2)
    text(d, (x + 22, y), f"larger markers: bridged DNs ({len(scene.hl_brain)} FlyWire, {len(scene.hl_vnc)} BANC; "
                         f"{scene.n_units} units) and the 16 steering MNs", 14, INK, anchor="lm")


def frame(scene, j, frame_idx, n_frames, view, g):
    cam = Camera(az_of(frame_idx, n_frames), scene.centre, *view)
    img = Image.fromarray(render_3d(scene, cam, g, j))
    d = ImageDraw.Draw(img)
    text(d, (40, 30), "Fly brain and nerve cord during a rotating drum (rot_right, 60 deg/s)", 26, INK, True)
    text(d, (40, 66), "One validation trial (seed 160). Brain: FlyWire spiking model. Nerve cord: BANC rate model "
                      "driven by the brain's descending neurons.", 15, INK2)
    for (X, c, lab, below) in ((scene.brain, scene.brain_c, "FlyWire brain, spiking model", True),
                               (scene.vnc, scene.vnc_c, "BANC nerve cord, trained rate model (connectome-constrained)",
                                False)):
        _, py = cam.project(X[::11])
        cx, _ = cam.project(c[None])
        y = np.percentile(py, 99.5) + 16 if below else np.percentile(py, 0.5) - 16
        text(d, (float(np.clip(cx[0], 330, VIEW_W - 330)), float(y)), lab, 17, INK, True, anchor="ma" if below else "md")
    for side, s in (("fly's left", 1), ("fly's right", -1)):
        px, py = cam.project((scene.brain_c + s * 480 * LEFT - 60 * UP)[None])
        text(d, (float(np.clip(px[0], 60, VIEW_W - 60)), float(py[0])), side, 15, INK2, anchor="mm")
    legend(d, scene, 40, H - 130)
    text(d, (40, H - 26), FOOTER, 13, INK2, anchor="ls")
    x0 = VIEW_W + 10
    stimulus_panel(d, scene, j, (x0, 30, W - 30, 330))
    dA_panel(d, scene, j, (x0, 350, W - 30, 680))
    mn_panel(d, scene, j, (x0, 700, W - 30, 1000))
    return img


def stills(a):
    t0 = time.time()
    scene = Scene()
    n_frames = N_INT * FRAMES_PER_INTERVAL
    view = fit_focal(scene, n_frames)
    print(f"scene loaded ({time.time() - t0:.1f} s): {len(scene.brain):,} brain, {len(scene.vnc):,} VNC; VNC range "
          f"{scene.range:.4f} a.u. (p{RANGE_PCT:.0f} |change|), MN range {scene.mn_range:.4f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ms in (float(x) for x in a.times.split(",")):
        j = interval_at(ms)
        t1 = time.time()
        g = scene.glow(j)
        t2 = time.time()
        img = frame(scene, j, j * FRAMES_PER_INTERVAL, n_frames, view, g)
        t3 = time.time()
        p = out / f"frame_{ms:+05.0f}ms.png"
        img.save(p)
        print(f"{p}: interval {j}, azimuth {az_of(j * FRAMES_PER_INTERVAL, n_frames):.1f} deg; glow {t2 - t1:.2f} s, "
              f"frame {t3 - t2:.2f} s, save {time.time() - t3:.2f} s", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("stills", "video"))
    ap.add_argument("--times", default="-100,100,500", help="stills: ms from stimulus onset")
    ap.add_argument("--out", default=str(PREVIEW / "frames"))
    a = ap.parse_args(argv)
    if a.part == "video":
        raise SystemExit("the full video is step 4b (not yet)")
    return stills(a)


if __name__ == "__main__":
    sys.exit(main())
