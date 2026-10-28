"""Visual input of stage A for the README: re-make six phase-0 renders, gate them, export the figure data.

The renders behind Figures S9-S11 (runs/f5/stageA/tmp/) were deleted batch by batch during the record generation; this
script re-makes the few the README needs, proves them identical to the originals and exports what the figure needs.

    env -u PYTHONPATH python scripts/readme_visual_input.py render                 # runs/f5/stageA/readme_renders/
    env -u PYTHONPATH python scripts/readme_visual_input.py gate                   # exit 1 unless identical
    env -u PYTHONPATH python scripts/readme_visual_input.py export [--overwrite]   # docs/figures/visual_input.json,
                                                                                    # docs/figures/input_checks.csv
    env -u PYTHONPATH python scripts/readme_visual_input.py figure [--out DIR] [--overwrite]
                                                                    # docs/figures/fig7_visual_input.png

render: the keys of KEYS (`render_key` names of scripts/f5_stageA_records.py, phase 0), by the code path of its
`worker_render` (f3b1_brain.build, render_setup, render_stimulus with key_params), in a child process (`render-worker`)
started with the records' WORKER_ENV and MUJOCO_GL=egl, so the thread settings hold before numpy / torch / numba are
imported. Fresh eyes must equal runs/f5/stageA/boundary_meta.npz exactly. Output <key>.npz (L, R as returned, float32),
written as <key>.tmp.npz and renamed; an existing file is skipped. Nothing else in runs/f5/ is written.

gate: (a) bit for bit against runs/f5/stageA_gate5/stim_rates.npz (60 deg/s phase-0 render of the four stimuli);
(b) the window means of runs/f5/A/gates/flyvis_wm/<key>.npy, recomputed as `flyvis_worker` of scripts/f5_train.py;
(c) A_in_T4aT5a of the input check (f3b1_brain.input_check) at 60 and 120 deg/s against runs/f5/stageA/input_checks.json
and input_checks_batch0 of docs/results/f5_stageA_records.json. Every key must be covered by (a) or (b); (c) must pass.

export (stops unless the gate passes; rule 13: refuses if either output exists, unless --overwrite):
- docs/figures/visual_input.json: provenance, the input-check window (f3b1_common), the T4/T5 subtypes and their
  neuron counts per eye, the mean rate per key, eye and subtype over that window (float64 mean of the float32 rates),
  the per-subtype asymmetry a_in(mean left eye, mean right eye) of those floats, the lead-in mean T4/T5 rate, one
  heatmap colour limit (99th percentile, T4/T5, both eyes, rot_left and rot_right at 60 deg/s), the SHA-256 of the
  raw eye images of `eye_images()` in scripts/f5_stageA_preview.py (recomputed, not stored), and the gate values.
- docs/figures/input_checks.csv: runs/f5/stageA/input_checks.json copied, one row per seed group and speed.

figure: Figure 7 of the README, in the style of scripts/readme_figures.py (whose style constants, font_setup and
render_png it imports; 180 mm wide, exactly 2000 px, at most 1.5 MB; --out inside the local preview folder refused;
rule 13). It stops unless every render's SHA-256 and that of boundary_meta.npz equal `provenance` of visual_input.json
and the recomputed eye images equal its `eye_images` entry. Nothing is computed about the test split.
- a: what the eyes see, rot_left at 60 deg/s, phase 0, one column per eye above its heat maps of b: the raw eye images
  of `eye_images()` at the grey lead-in and the first stimulus time, and a space-time plot of the middle image row
  (row H // 2 of every frame of the trial, against the azimuth of tasks/t0_stimuli.py, + = left), each interval over
  the same time bin as in b; the row equals the hashed frames at their intervals;
- b: the T4/T5 boundary rates of rot_left and rot_right at 60 deg/s, left and right eye, neurons grouped by subtype
  (within a subtype the stored order, as scripts/f5_stageA_preview.py), magma from 0 to `heatmap_vmax_hz`;
- c: the per-subtype eye asymmetry of `asymmetry`, the four rotation renders;
- d: A_in of T4a + T5a of the input check (input_checks.csv), training and validation seed groups, per drum speed.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import f3b1_common as C1  # noqa: E402
import f5_stageA_records as R  # noqa: E402

STAGE = ROOT / "runs" / "f5" / "stageA"
META = STAGE / "boundary_meta.npz"
CHECKS = STAGE / "input_checks.json"
RENDERS = STAGE / "readme_renders"
GATE5 = ROOT / "runs" / "f5" / "stageA_gate5" / "stim_rates.npz"
FLYVIS_WM = ROOT / "runs" / "f5" / "A" / "gates" / "flyvis_wm"
RECORDS_JSON = ROOT / "docs" / "results" / "f5_stageA_records.json"
PREVIEW_SCRIPT = ROOT / "scripts" / "f5_stageA_preview.py"
OUT_JSON = ROOT / "docs" / "figures" / "visual_input.json"
OUT_CSV = ROOT / "docs" / "figures" / "input_checks.csv"
ROT = ("rot_left", "rot_right")
GATE_SPEEDS = (60, 120)                                  # the speeds whose phase-0 renders are re-made
KEYS = tuple([R.render_key(s, v, 0) for v in GATE_SPEEDS for s in ROT] + [R.render_key(s, GATE_SPEEDS[0], 0)
                                                                          for s in ("static", "grey")])
ROT_KEYS = KEYS[:4]
EYES = ("L", "R")
FIG7 = "fig7_visual_input.png"
FIG7_HEIGHT_MM = 170
# Copied from scripts/viz_activity_figure.py (NEG, POS and DIR_COL, the direction colours of Figure 3), not imported:
# that module imports pandas and viz_anatomy. Okabe-Ito blue and vermillion.
NEG, POS = "#0072B2", "#D55E00"
DIR_COL = {"rot_right": POS, "rot_left": NEG}
A_IN_LABEL = {"c": r"eye asymmetry $A_\mathrm{in}$", "d": r"$A_\mathrm{in}$ of T4a + T5a"}    # y labels of 7c, 7d
SPEED_MARKER = dict(zip(GATE_SPEEDS, ("o", "s")))         # Figure 7c: marker shape by drum speed


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_meta():
    m = np.load(META)
    return {"L": m["tcode_L"], "R": m["tcode_R"]}, [str(t) for t in m["fw_types"]], m


# ── render ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def worker():
    import gc

    import torch
    torch.set_num_threads(1)
    try:
        import numba
        nb = numba.config.NUMBA_NUM_THREADS
    except ImportError:
        nb = "numba not installed"
    print(json.dumps(dict(torch_threads=torch.get_num_threads(), numba_threads=nb,
                          env={k: os.environ.get(k) for k in [*R.WORKER_ENV, "MUJOCO_GL"]})), flush=True)
    from f3b1_brain import build, render_setup, render_stimulus
    cx, groups_in, eyes = build(log=lambda *a, **k: None)
    del cx, groups_in                                  # as worker_render: the eyes keep what they need
    gc.collect()
    _, _, m = load_meta()
    fresh = {"tcode_L": eyes.tcode["L"], "tcode_R": eyes.tcode["R"], "fw_types": np.array(eyes.fw_types)}
    for k, v in fresh.items():
        if not (v.dtype == m[k].dtype and v.shape == m[k].shape and np.array_equal(v, m[k])):
            raise SystemExit(f"fresh eyes differ from {META} in {k}: stop")
    print("fresh eyes equal boundary_meta.npz (tcode_L, tcode_R, fw_types)", flush=True)
    rays, retina = render_setup()
    for key in KEYS:
        path = RENDERS / f"{key}.npz"
        if path.exists():
            print(f"{key}: exists, skipped", flush=True)
            continue
        t1 = time.time()
        cond, tf, ph = R.key_params(key)
        L, Rr = render_stimulus(eyes, cond, rays, retina, tf_hz=tf, phase_deg=ph)
        tmp = RENDERS / f"{key}.tmp.npz"
        np.savez(tmp, L=L, R=Rr)
        tmp.rename(path)
        print(f"render {key}: {time.time() - t1:.1f} s, L {L.shape} {L.dtype}, R {Rr.shape} {Rr.dtype}", flush=True)
    print(json.dumps(dict(peak_rss_gb=R.peak_rss_gb(), rss_gb=R.peak_rss_gb("VmRSS:"))), flush=True)


def render():
    RENDERS.mkdir(parents=True, exist_ok=True)
    todo = [k for k in KEYS if not (RENDERS / f"{k}.npz").exists()]
    if not todo:
        print("every render exists; nothing rendered")
    else:
        env = dict(os.environ, **R.WORKER_ENV, MUJOCO_GL="egl")
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "render-worker"], cwd=ROOT, env=env)
        print(f"wall time {time.time() - t0:.1f} s")
        if r.returncode:
            raise SystemExit(f"render-worker failed ({r.returncode})")
    for k in KEYS:
        z = np.load(RENDERS / f"{k}.npz")
        print(k, {e: (z[e].shape, str(z[e].dtype)) for e in EYES})
    size = sum(p.stat().st_size for p in RENDERS.iterdir())
    print(f"{RENDERS}: {len(list(RENDERS.iterdir()))} files, {size} bytes")
    return 0


# ── gate ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def identical(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))


def gate():
    """Every comparison as dict(test, what, equal); A_in values; ok."""
    from f3b1_brain import input_check
    missing = [k for k in KEYS if not (RENDERS / f"{k}.npz").exists()]
    if missing:
        raise SystemExit(f"renders missing: {missing} (run `render` first)")
    rd = {k: dict(np.load(RENDERS / f"{k}.npz")) for k in KEYS}
    comps, covered = [], {}
    if GATE5.exists():                                                   # (a)
        z = np.load(GATE5)
        for st in C1.STIMULI:
            key = R.render_key(st, GATE_SPEEDS[0], 0)
            eq = all(identical(z[f"{st}__{e}"], rd[key][e]) for e in EYES)
            comps.append(dict(test="a", what=f"{key} == {GATE5.relative_to(ROOT)} {st}__L/__R", equal=eq))
            covered.setdefault(key, []).append(("a", eq))
    from train import f5_stageA as A                                     # (b)
    lo, hi = A.WINDOW
    for key in KEYS:
        p = FLYVIS_WM / f"{key}.npy"
        if not p.exists():
            continue
        L, Rr = rd[key]["L"], rd[key]["R"]
        wm = np.concatenate([L[lo:hi].mean(0), Rr[lo:hi].mean(0)]).astype(np.float32)
        eq = identical(wm, np.load(p))
        comps.append(dict(test="b", what=f"window mean of {key} == {p.relative_to(ROOT)}", equal=eq))
        covered.setdefault(key, []).append(("b", eq))
    tcode, fw_types, _ = load_meta()                                     # (c)
    ic, b0 = json.loads(CHECKS.read_text()), json.loads(RECORDS_JSON.read_text())["input_checks_batch0"]
    a_vals = {}
    for v in GATE_SPEEDS:
        rates = {f"{st}__{e}": rd[R.render_key(st, v, 0)][e] for st in C1.STIMULI for e in EYES}
        chk = input_check(rates, tcode, fw_types)
        name = f"p0_v{v:03d}"
        for st in ROT:
            x = chk["per_stimulus"][st]["A_in_T4aT5a"]
            y, w = ic[name]["A_in_T4aT5a"][st], b0[name]["A_in_T4aT5a"][st]
            eq = x == y and x == w
            a_vals.setdefault(name, {})[st] = dict(recomputed=x, input_checks_json=y, records_json=w, equal=eq)
            comps.append(dict(test="c", what=f"A_in_T4aT5a {name} {st}: {x!r} == {y!r} == {w!r}", equal=eq))
        comps.append(dict(test="c", what=f"input check {name} passed", equal=bool(chk["passed"])))
    uncovered = [k for k in KEYS if not any(eq for _, eq in covered.get(k, []))]
    ok = (bool(comps) and all(c["equal"] for c in comps) and not uncovered
          and any(c["test"] in ("a", "b") for c in comps))
    return dict(comparisons=comps, covered={k: sorted({t for t, _ in covered.get(k, [])}) for k in KEYS},
                uncovered=uncovered, A_in_T4aT5a=a_vals, ok=ok)


def print_gate(g):
    for c in g["comparisons"]:
        print(f"[{c['test']}] {'EQUAL' if c['equal'] else 'DIFFERENT'}  {c['what']}")
    print("covered:", g["covered"], "uncovered:", g["uncovered"])
    print("GATE", "PASSED" if g["ok"] else "FAILED")


# ── export ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def eye_images_spec():
    """Interval indices and time formula of `eye_images()` in scripts/f5_stageA_preview.py, read from its source (the
    function draws; it does not return the luminance)."""
    src = PREVIEW_SCRIPT.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "eye_images")
    body = ast.get_source_segment(src, fn)
    ks = next(ast.literal_eval(n.value) for n in ast.walk(fn)
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "ks")
    consts = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple):
            for t, v in zip(n.targets[0].elts, n.value.elts):
                consts[t.id] = ast.literal_eval(v)
    for needle in ("t = None if k < N_GREY else (k - N_GREY + 1) * DT",
                   "lum = S.luminance(2, t, rays, tf_hz=60 / S.WAVELENGTH_DEG)",
                   "b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs=\"tuck\", enable_vision=True)",
                   "rays = S.camera_rays(b.m, b.d, b.root)"):
        if needle not in body:
            raise SystemExit(f"eye_images() in {PREVIEW_SCRIPT} changed: {needle!r} not found")
    return ks, consts["N_GREY"], consts["DT"]


def eye_images(n_intervals=None):
    """The raw eye images of `eye_images()` in scripts/f5_stageA_preview.py, recomputed by its code: (entry, lums,
    space_time). entry is the `eye_images` entry of visual_input.json (hashes, no arrays); lums the luminance arrays
    (eye, row, column) in the order of entry["images"]. With n_intervals, space_time holds Figure 7a's space-time data:
    row H // 2 of the full frame at every interval 0..n_intervals-1, each frame computed by the same S.luminance call
    and dropped after its row is kept (rows, (eye, interval, column)), and the azimuth of that row (deg, + = left, of
    S.azimuth); otherwise None. Used by `export` (entry) and `figure` (all three)."""
    import mujoco

    from body.flygym_body import FlightBody
    from tasks import t0_stimuli as S
    ks, n_grey, dt = eye_images_spec()
    key = R.render_key("rot_left", GATE_SPEEDS[0], 0)
    cond, tf, ph = R.key_params(key)
    if (cond, tf, ph) != (2, 60 / S.WAVELENGTH_DEG, 0.0):              # the arguments eye_images() uses
        raise SystemExit(f"{key} is not the stimulus of eye_images()")
    b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs="tuck", enable_vision=True)
    mujoco.mj_forward(b.m, b.d)
    rays = S.camera_rays(b.m, b.d, b.root)
    mid = rays.shape[1] // 2
    out, lums, rows = {}, {}, []
    for k in sorted(set(ks) | set(range(n_intervals or 0))):
        t = None if k < n_grey else (k - n_grey + 1) * dt
        lum = S.luminance(2, t, rays, tf_hz=60 / S.WAVELENGTH_DEG)
        if k < (n_intervals or 0):
            rows.append(lum[:, mid].copy())
        if k in ks:
            lums[k] = lum
            out[k] = dict(interval=k, t_s=t, shape=list(lum.shape), dtype=str(lum.dtype),
                          sha256=hashlib.sha256(np.ascontiguousarray(lum).tobytes()).hexdigest())
    entry = dict(key=key, source="scripts/f5_stageA_preview.py eye_images() (recomputed; the images are not stored)",
                 eyes=["left", "right"], hash_of="np.ascontiguousarray(lum).tobytes(), lum of shape (eye, row, column)",
                 images=[out[k] for k in ks])
    space_time = None if not n_intervals else dict(rows=np.stack(rows, axis=1), row=mid,
                                                   azimuth_deg=S.azimuth(rays[:, mid]))
    return entry, [lums[k] for k in ks], space_time


def export_csv():
    ic = json.loads(CHECKS.read_text())
    order = {"train": 0, "val": 1, "test": 2}
    rows = []
    for name, v in ic.items():
        g, sp = name.rsplit("_v", 1)
        seed = 0 if g == "p0" else int(g[1:])
        split = R.split_of(seed)
        a = v.get("A_in_T4aT5a")
        rows.append(dict(group=g, split=split, speed_deg_s=int(sp), phase_deg=repr(R.phase_of(seed)),
                         passed=v["passed"], A_in_rot_left="" if a is None else repr(a["rot_left"]),
                         A_in_rot_right="" if a is None else repr(a["rot_right"]), _k=(order[split], seed, int(sp))))
        if split == "test" and a is not None:
            raise SystemExit(f"{name}: a test group with kept values")
    rows.sort(key=lambda r: r["_k"])
    return rows


def export(overwrite=False):
    existing = [p for p in (OUT_JSON, OUT_CSV) if p.exists()]
    if existing and not overwrite:
        raise SystemExit(f"{', '.join(str(p) for p in existing)} exist; pass --overwrite to replace (nothing written)")
    from f3b1_brain import a_in
    g = gate()
    print_gate(g)
    if not g["ok"]:
        raise SystemExit("gate failed: nothing exported")
    tcode, fw_types, _ = load_meta()
    rd = {k: dict(np.load(RENDERS / f"{k}.npz")) for k in KEYS}
    subtypes = [t for t in fw_types if t.startswith(("T4", "T5"))]
    if subtypes != sorted(subtypes):
        raise SystemExit(f"unexpected T4/T5 order in fw_types: {subtypes}")
    sel = {e: {t: tcode[e] == fw_types.index(t) for t in subtypes} for e in EYES}
    t45 = {e: np.isin(tcode[e], [fw_types.index(t) for t in subtypes]) for e in EYES}
    counts = {e: {t: int(sel[e][t].sum()) for t in subtypes} for e in EYES}
    lo, hi = C1.WINDOW
    mean = {k: {e: {t: float(rd[k][e][lo:hi][:, sel[e][t]].mean(dtype=np.float64)) for t in subtypes} for e in EYES}
            for k in ROT_KEYS}
    asym = {k: {t: a_in(mean[k]["L"][t], mean[k]["R"][t]) for t in subtypes} for k in ROT_KEYS}
    lead = {k: {e: float(rd[k][e][:C1.N_GREY][:, t45[e]].mean(dtype=np.float64)) for e in EYES} for k in KEYS}
    hm_keys = [R.render_key(s, GATE_SPEEDS[0], 0) for s in ROT]
    pooled = np.concatenate([rd[k][e][:, t45[e]].ravel() for k in hm_keys for e in EYES])
    vmax = float(np.percentile(pooled, 99))
    from sense import vision_flyvis as VF                              # the boundary-layer rate clip
    r_clip = float(VF.R_CLIP)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    d = dict(
        provenance=dict(
            script="scripts/readme_visual_input.py", head_commit=head.strip(),
            renders={k: dict(file=f"runs/f5/stageA/readme_renders/{k}.npz", sha256=sha_file(RENDERS / f"{k}.npz"),
                             shape={e: list(rd[k][e].shape) for e in EYES}, dtype={e: str(rd[k][e].dtype) for e in EYES},
                             t0_condition=R.key_params(k)[0], tf_hz=R.key_params(k)[1], phase_deg=R.key_params(k)[2])
                     for k in KEYS},
            boundary_meta_sha256=sha_file(META),
            note=("renders re-made by `render` (not committed); `gate` showed them identical to the originals; the "
                  "rates are FlyVis boundary-layer rates in Hz, one row per 5 ms interval")),
        window=dict(window=list(C1.WINDOW), n_grey=C1.N_GREY, interval_ms=C1.INTERVAL_MS,
                    note=("input-check window of scripts/f3b1_common.py (interval indices, end exclusive: the whole "
                          "stimulus after the grey lead-in); not the stage-A analysis window of train/f5_stageA.py")),
        subtypes=subtypes,
        neurons={e: dict(per_subtype=counts[e], t4t5_total=int(t45[e].sum())) for e in EYES},
        mean_rate_hz=mean,
        asymmetry=dict(formula="f3b1_brain.a_in(mean left eye, mean right eye) = (l - r) / (l + r + 1), per subtype",
                       values=asym),
        lead_in_mean_hz=dict(intervals=[0, C1.N_GREY], note="mean T4/T5 rate over intervals 0..N_GREY-1",
                             values=lead),
        heatmap_vmax_hz=dict(value=vmax, keys=hm_keys,
                             note="numpy percentile 99 (default method) of the T4/T5 rates, both eyes, all intervals",
                             r_clip_hz=r_clip, fraction_at_clip=float((pooled == r_clip).mean()),
                             clip_source="sense/vision_flyvis.py R_CLIP (configs r_clip); rates are clipped to it"),
        eye_images=eye_images()[0],
        gate=dict(comparisons=g["comparisons"], covered=g["covered"], passed=g["ok"], A_in_T4aT5a=g["A_in_T4aT5a"]),
    )
    rows = export_csv()
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(OUT_JSON)
    tmp = OUT_CSV.with_suffix(".tmp.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "_k"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT_CSV)
    per = ", ".join(f"{s} {sum(r['split'] == s for r in rows)}" for s in ("train", "val", "test"))
    print(f"{OUT_JSON}: written; {OUT_CSV}: {len(rows)} rows ({per})")
    return 0


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def check_inputs(d, renders=RENDERS, meta=META):
    """Stop unless every render and boundary_meta.npz have the SHA-256 recorded in `provenance` of visual_input.json."""
    bad = []
    for k, v in d["provenance"]["renders"].items():
        p = Path(renders) / f"{k}.npz"
        if not p.exists():
            bad.append(f"{p}: missing")
        elif sha_file(p) != v["sha256"]:
            bad.append(f"{p}: SHA-256 differs from visual_input.json")
    if sha_file(meta) != d["provenance"]["boundary_meta_sha256"]:
        bad.append(f"{meta}: SHA-256 differs from visual_input.json")
    if bad:
        raise SystemExit("inputs differ from visual_input.json (nothing written):\n" + "\n".join(bad))


def check_space_time(entry, lums, space_time):
    """Stop unless, at every interval of entry["images"], the space-time row equals row space_time["row"] of the hashed
    frame (np.array_equal)."""
    for im, lum in zip(entry["images"], lums):
        a, b = space_time["rows"][:, im["interval"]], lum[:, space_time["row"]]
        if not (a.dtype == b.dtype and np.array_equal(a, b)):
            raise SystemExit(f"interval {im['interval']}: the space-time row differs from the hashed frame")


def read_csv(path=OUT_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def draw_fig7(d, rows, lums, space_time, rates, tcode, fw_types, held_out):
    """d: visual_input.json; rows: input_checks.csv; lums, space_time: of eye_images(); rates[key][eye]: the renders of
    the heat-map keys; tcode, fw_types: boundary_meta.npz; held_out: drum speeds."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import readme_figures as F
    W, H = F.WIDTH_MM, FIG7_HEIGHT_MM
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))

    def ax_mm(x, y, w, h):                                      # bottom-left corner and size in mm
        return fig.add_axes((x / W, y / H, w / W, h / H))

    def letter(x, y, lab):
        fig.text(x / W, y / H, lab, ha="left", va="bottom", fontsize=F.FONT_PT + 1, fontweight="bold")

    # rows placed from the bottom (mm): c and d, then b, then a
    yc, hc = 12.0, 30.0                                         # c, d axes
    hh, hw, gap = 23.0, 66.0, 3.0                               # b heat maps
    xb = (17.0, 17.0 + hw + 13.0)                               # x of each eye's column in a and b
    yb_top = yc + hc + 19.0 + 2 * hh + gap
    ys, hs = yb_top + 15.0, 20.0                                # a, space-time plots
    imgs = d["eye_images"]["images"]
    ih = 13.0                                                   # a, eye images
    iw = ih * imgs[0]["shape"][2] / imgs[0]["shape"][1]
    yi = ys + hs + 3.0
    n_grey, dt = d["window"]["n_grey"], d["window"]["interval_ms"] / 1000

    # a: what the eyes see; top, the grey lead-in and the first stimulus time; below, the middle image row over time
    shown = [next(j for j, im in enumerate(imgs) if im["t_s"] is None),
             next(j for j, im in enumerate(imgs) if im["t_s"] is not None)]
    rows_st, az = space_time["rows"], space_time["azimuth_deg"]
    t_edges = (np.arange(rows_st.shape[1] + 1) - n_grey) * dt         # interval k over the same bin as in b
    for c, side in enumerate(d["eye_images"]["eyes"]):
        xs = xb[c] + (hw - 2 * iw - 4.0) / 2
        for i, j in enumerate(shown):
            ax = ax_mm(xs + i * (iw + 4.0), yi, iw, ih)
            ax.imshow(lums[j][c], cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title("grey lead-in" if imgs[j]["t_s"] is None else f"t = {imgs[j]['t_s']:.3f} s", pad=2)
        fig.text((xb[c] + hw / 2) / W, (yi + ih + 5.0) / H, f"{side} eye", ha="center", va="bottom")
        if not (np.all(np.diff(az[c]) < 0) or np.all(np.diff(az[c]) > 0)):
            raise SystemExit(f"{side} eye: the azimuth of the middle row is not monotonic")
        mids = (az[c][1:] + az[c][:-1]) / 2
        a_edges = np.r_[2 * az[c][0] - mids[0], mids, 2 * az[c][-1] - mids[-1]]
        ax = ax_mm(xb[c], ys, hw, hs)
        st = ax.pcolormesh(t_edges, a_edges, rows_st[c].T, cmap="gray", vmin=0, vmax=1, shading="flat")
        ax.set_xlim(t_edges[0], t_edges[-1])
        ax.set_ylim(a_edges.min(), a_edges.max())                      # positive azimuth (left) up
        ax.axvline(0, color="white", lw=0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("time from drum onset (s)")
        ax.set_ylabel("azimuth (deg, + = left)")
    cax = ax_mm(xb[1] + hw + 3.0, ys, 2.2, hs)
    cb = fig.colorbar(st, cax=cax)
    cb.set_ticks(np.linspace(st.norm.vmin, st.norm.vmax, 3), labels=[f"{v:g}" for v in
                                                                      np.linspace(st.norm.vmin, st.norm.vmax, 3)])
    cb.set_label("luminance")
    cb.outline.set_linewidth(0.5)
    letter(1.5, yi + ih + 5.0, "a")

    # b: T4/T5 boundary rates
    subtypes = d["subtypes"]
    vmax = d["heatmap_vmax_hz"]["value"]
    if vmax != d["heatmap_vmax_hz"]["r_clip_hz"]:
        raise SystemExit("heatmap_vmax_hz is not the rate clip; the colour-bar label would be wrong")
    keys = d["heatmap_vmax_hz"]["keys"]
    t45 = [fw_types.index(t) for t in subtypes]
    for r, key in enumerate(keys):
        y = yb_top - (r + 1) * hh - r * gap
        st, v = key.split("_v")[0], int(key.split("_v")[1].split("_")[0])
        fig.text(2.0 / W, (y + hh / 2) / H, f"{st}, {v} deg/s", rotation=90, ha="left", va="center")
        for c, e in enumerate(EYES):
            ax = ax_mm(xb[c], y, hw, hh)
            tc = tcode[e]
            sel = np.flatnonzero(np.isin(tc, t45))
            order = sel[np.argsort(tc[sel], kind="stable")]                # as heatmaps() of f5_stageA_preview.py
            if [fw_types[k] for k in np.unique(tc[order])] != subtypes:
                raise SystemExit("subtype order differs from visual_input.json")
            z = rates[key][e][:, order].T
            if z.shape[0] != d["neurons"][e]["t4t5_total"]:
                raise SystemExit(f"{key} {e}: neuron count differs from visual_input.json")
            hm = ax.imshow(z, aspect="auto", cmap="magma", vmin=0, vmax=vmax,     # default interpolation
                           extent=[-n_grey * dt, (z.shape[1] - n_grey) * dt, z.shape[0], 0])
            ax.axvline(0, color="white", lw=0.6)
            bounds = np.flatnonzero(np.diff(tc[order])) + 1
            for b in bounds:
                ax.axhline(b, color="white", lw=0.3)
            mids = np.r_[0, bounds] + np.diff(np.r_[0, bounds, len(order)]) / 2
            ax.set_yticks(mids, subtypes)
            ax.tick_params(axis="y", length=0, pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"{'left' if e == 'L' else 'right'} eye", pad=3)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time from drum onset (s)")
    cax = ax_mm(xb[1] + hw + 3.0, yb_top - 2 * hh - gap, 2.2, 2 * hh + gap)
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label(f"T4/T5 rate (Hz), clipped at {vmax:g}")
    cb.outline.set_linewidth(0.5)
    letter(1.5, yb_top + 4.0, "b")

    # c: per-subtype eye asymmetry
    axc = ax_mm(17.0, yc, 68.0, hc)
    asym = d["asymmetry"]["values"]
    x = np.arange(len(subtypes))
    for key in asym:
        st, v = key.split("_v")[0], int(key.split("_v")[1].split("_")[0])
        dx = 0.13 * (-1 if v == min(SPEED_MARKER) else 1)
        axc.scatter(x + dx, [asym[key][t] for t in subtypes], s=10, color=DIR_COL[st], marker=SPEED_MARKER[v],
                    lw=0, zorder=3)
    axc.axhline(0, color=F.INK2, lw=0.5, zorder=1)
    axc.set_xticks(x, subtypes)
    axc.set_xlim(-0.6, len(subtypes) - 0.4)
    axc.set_xlabel("T4/T5 subtype")
    axc.set_ylabel(A_IN_LABEL["c"])
    hd = [Line2D([], [], ls="", marker="o", color=DIR_COL[s], ms=3, label=s) for s in ROT]
    hd += [Line2D([], [], ls="", marker=m, color=F.INK2, ms=3, label=f"{v} deg/s") for v, m in SPEED_MARKER.items()]
    axc.legend(handles=hd, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.1,
               columnspacing=0.8, borderaxespad=0.0)
    letter(1.5, yc + hc + 5.5, "c")

    # d: A_in of T4a + T5a over the training data
    axd = ax_mm(108.0, yc, 70.0, hc)
    used = [r for r in rows if r["split"] in ("train", "val")]
    speeds = sorted({int(r["speed_deg_s"]) for r in used})
    half = 0.4 * min(np.diff(speeds))
    for v in held_out:
        axd.axvspan(v - half, v + half, color=F.HELD_OUT_BAND, lw=0, zorder=0)
    rng = np.random.default_rng(0)                                     # horizontal jitter only
    for st in ROT:
        sp = np.array([int(r["speed_deg_s"]) for r in used])
        a = np.array([float(r[f"A_in_{st}"]) for r in used])
        axd.scatter(sp + rng.uniform(-0.25, 0.25, len(sp)) * half, a, s=3, color=DIR_COL[st], lw=0, alpha=0.6,
                    zorder=2)
    for sgn in (1, -1):
        axd.axhline(sgn * C1.INPUT_CHECK_MIN, color=F.INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axd.axhline(0, color=F.INK2, lw=0.5, zorder=1)
    axd.set_xticks(speeds, [f"{v}\n(held out)" if v in held_out else f"{v}" for v in speeds])
    axd.set_xlim(speeds[0] - 1.25 * half, speeds[-1] + 1.25 * half)
    axd.set_xlabel("drum speed (deg/s)")
    axd.set_ylabel(A_IN_LABEL["d"])
    hd = [Line2D([], [], ls="", marker="o", color=DIR_COL[s], ms=3, label=s) for s in ROT]
    hd += [Line2D([], [], color=F.INK2, lw=0.6, ls=(0, (3, 2)), label="input-check minimum")]
    axd.legend(handles=hd, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.1,
               columnspacing=0.8, borderaxespad=0.0)
    letter(94.0, yc + hc + 5.5, "d")
    return fig


def figure(out=None, overwrite=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import readme_figures as F
    out = Path(out) if out else OUT_JSON.parent
    if F.inside(out, F.PREVIEW):
        raise SystemExit(f"{out}: inside the local preview folder, which is never written")
    path = out / FIG7
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it (nothing written)")
    d = json.loads(OUT_JSON.read_text())
    check_inputs(d)
    keys = d["heatmap_vmax_hz"]["keys"]
    rates = {k: dict(np.load(RENDERS / f"{k}.npz")) for k in keys}
    n_int = {rates[k][e].shape[0] for k in keys for e in EYES}
    if len(n_int) != 1:
        raise SystemExit(f"the renders differ in length: {n_int}")
    entry, lums, space_time = eye_images(n_int.pop())
    if entry != d["eye_images"]:
        raise SystemExit("recomputed eye images differ from visual_input.json (nothing written)")
    check_space_time(entry, lums, space_time)
    print("renders, boundary_meta.npz and eye images equal visual_input.json; space-time rows equal the hashed frames")
    tcode, fw_types, _ = load_meta()
    from train import f5_stageA as A
    rows = read_csv()
    used = [r for r in rows if r["split"] in ("train", "val")]
    for v in sorted({int(r["speed_deg_s"]) for r in used}):
        print(f"panel d, {v} deg/s: n = {sum(int(r['speed_deg_s']) == v for r in used)} seed groups (train + val)")
    fam = F.font_setup()
    fig = draw_fig7(d, rows, lums, space_time, rates, tcode, fw_types, sorted(A.HELD_OUT.values()))
    png = F.render_png(fig)
    plt.close(fig)
    if len(png) > F.MAX_BYTES:
        raise SystemExit(f"{FIG7}: {len(png):,} bytes, over {F.MAX_BYTES:,} (nothing written)")
    from PIL import Image
    import io
    w, h = Image.open(io.BytesIO(png)).size
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    print(f"{path}: {w} x {h} px, {len(png):,} bytes, font {fam}, {F.WIDTH_MM} x {FIG7_HEIGHT_MM} mm")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["render", "render-worker", "gate", "export", "figure"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=None, help="figure: output folder (default docs/figures/)")
    a = ap.parse_args(argv)
    if a.command == "render":
        return render()
    if a.command == "render-worker":
        return worker()
    if a.command == "gate":
        g = gate()
        print_gate(g)
        return 0 if g["ok"] else 1
    if a.command == "figure":
        return figure(a.out, a.overwrite)
    return export(a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
