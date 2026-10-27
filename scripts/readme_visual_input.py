"""Visual input of stage A for the README: re-make six phase-0 renders, gate them, export the figure data.

The renders behind Figures S9-S11 (runs/f5/stageA/tmp/) were deleted batch by batch during the record generation; this
script re-makes the few the README needs, proves them identical to the originals and exports what the figure needs.

    env -u PYTHONPATH python scripts/readme_visual_input.py render                 # runs/f5/stageA/readme_renders/
    env -u PYTHONPATH python scripts/readme_visual_input.py gate                   # exit 1 unless identical
    env -u PYTHONPATH python scripts/readme_visual_input.py export [--overwrite]   # docs/figures/visual_input.json,
                                                                                    # docs/figures/input_checks.csv

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


def eye_images():
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
    out = []
    for k in ks:
        t = None if k < n_grey else (k - n_grey + 1) * dt
        lum = S.luminance(2, t, rays, tf_hz=60 / S.WAVELENGTH_DEG)
        out.append(dict(interval=k, t_s=t, shape=list(lum.shape), dtype=str(lum.dtype),
                        sha256=hashlib.sha256(np.ascontiguousarray(lum).tobytes()).hexdigest()))
    return dict(key=key, source="scripts/f5_stageA_preview.py eye_images() (recomputed; the images are not stored)",
                eyes=["left", "right"], hash_of="np.ascontiguousarray(lum).tobytes(), lum of shape (eye, row, column)",
                images=out)


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
        eye_images=eye_images(),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["render", "render-worker", "gate", "export"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "render":
        return render()
    if a.command == "render-worker":
        return worker()
    if a.command == "gate":
        g = gate()
        print_gate(g)
        return 0 if g["ok"] else 1
    return export(a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
