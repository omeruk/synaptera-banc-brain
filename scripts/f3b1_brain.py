"""Brain stage of prereg/09_f3b1_optomotor.md: stimulus render, input check, FlyWire v783 runs (sections 3-4).

    env -u PYTHONPATH python scripts/f3b1_brain.py --out runs/f3b1

1. Render (once, deterministic): body-frame stimuli of tasks/t0_stimuli.py -> raw FlyGym eye images -> Retina ->
   FlyVis (5 ms, 1 sub-frame, data/visual_transduction_5ms.json) -> boundary-layer rates; 100 grey + 200 stimulus
   intervals per stimulus -> OUT/stim_rates.npz and OUT/input_check.json. A failed input check stops here.
2. Brain: Shiu reference arm on FlyWire v783, inputs sugar (0 Hz), vbnd_L, vbnd_R; per (stimulus, seed) a fresh state,
   per-interval spike counts of all FlyWire DNs -> OUT/brain/<stimulus>_s<seed>.npz (existing files are skipped).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import f3b1_common as C  # noqa: E402

MIN_FREE_GB = 6.0
TRANSDUCTION = ROOT / "data" / "visual_transduction_5ms.json"


def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def a_in(rl, rr):
    return float((rl - rr) / (rl + rr + 1.0))


def input_check(rates, tcode, fw_types):
    sel = {s: np.isin(tcode[s], [fw_types.index(t) for t in ("T4a", "T5a")]) for s in ("L", "R")}
    lo, hi = C.WINDOW
    res = {}
    for name in C.STIMULI:
        L, R = rates[f"{name}__L"][lo:hi], rates[f"{name}__R"][lo:hi]
        res[name] = dict(A_in_T4aT5a=a_in(L[:, sel["L"]].mean(), R[:, sel["R"]].mean()), A_in_all=a_in(L.mean(), R.mean()),
                         mean_L=float(L.mean()), mean_R=float(R.mean()))
    y = (res["rot_left"]["A_in_T4aT5a"], res["rot_right"]["A_in_T4aT5a"])
    ok = bool(np.sign(y[0]) != np.sign(y[1]) and min(abs(v) for v in y) >= C.INPUT_CHECK_MIN)
    return dict(per_stimulus=res, passed=ok, min_abs=C.INPUT_CHECK_MIN,
                expected_signs_recorded_not_criteria=dict(rot_left_positive=bool(y[0] > 0), rot_right_negative=bool(y[1] < 0)))


def build(log=print):
    from connectome.adapters import flywire783 as fw
    from sense import vision_boundary as VB
    from sense import vision_flyvis as V
    cx = fw.load_connectome(hash_files=True)
    sugar_ids = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    table = VB.load_table(ROOT / "data" / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    vg = VB.boundary_groups(cx, table)
    groups_in = {"sugar": cx.index.group(sugar_ids), "vbnd_L": vg["vbnd_L"], "vbnd_R": vg["vbnd_R"]}
    tr = VB.load_transduction(TRANSDUCTION)
    interval_s = C.INTERVAL_MS * 1e-3
    assert abs(tr["dt"] - interval_s) < 1e-12, tr["dt"]
    columns = V.load_columns(ROOT / "data" / "column_assignment.csv.gz")
    eyes = VB.BoundaryEyes(table, {"L": cx.index.ids_of(groups_in["vbnd_L"]), "R": cx.index.ids_of(groups_in["vbnd_R"])},
                           tr, columns, cx.index, interval_s=interval_s, subframes=1, verbose=False)
    return cx, groups_in, eyes


def render_setup():
    """Camera rays of the fixed body and the FlyGym retina, shared by every render."""
    import mujoco
    from flygym.vision import Retina

    from body.flygym_body import FlightBody
    from tasks import t0_stimuli as S
    b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs="tuck", enable_vision=True)
    mujoco.mj_forward(b.m, b.d)
    return S.camera_rays(b.m, b.d, b.root), Retina()


def render_stimulus(eyes, cond, rays, retina, tf_hz=None, phase_deg=0.0):
    """Boundary-layer rates (L, R) of one t0 condition: 100 grey + 200 stimulus intervals, float32. tf_hz / phase_deg:
    the F5 drum temporal frequency and grating phase (tasks/t0_stimuli.luminance; defaults = the F3b render)."""
    from tasks import t0_stimuli as S
    tf_hz = S.TF_HZ if tf_hz is None else tf_hz
    dt = C.INTERVAL_MS * 1e-3
    eyes.reset()
    rl, rr = [], []
    for k in range(C.N_GREY + C.N_STIM):
        t = None if k < C.N_GREY else (k - C.N_GREY + 1) * dt
        act, _ = eyes.step([S.encode(S.luminance(cond, t, rays, tf_hz=tf_hz, phase_deg=phase_deg), retina)])
        r = eyes.rates(act)
        rl.append(r["L"].astype(np.float32))
        rr.append(r["R"].astype(np.float32))
    return np.array(rl), np.array(rr)


def render(eyes, log=print, tf_hz=None, phase_deg=0.0):
    rays, retina = render_setup()
    out = {}
    for name, (cond, _) in C.STIMULI.items():
        t1 = time.time()
        out[f"{name}__L"], out[f"{name}__R"] = render_stimulus(eyes, cond, rays, retina, tf_hz, phase_deg)
        lo, hi = C.WINDOW
        log(f"render {name:9s} (t0 cond {cond}): window mean L {out[f'{name}__L'][lo:hi].mean():.2f} "
            f"R {out[f'{name}__R'][lo:hi].mean():.2f} Hz ({time.time() - t1:.0f} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "f3b1"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "brain").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"free {free_gb():.1f} GB", flush=True)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb():.1f} GB available (< {MIN_FREE_GB}); stop")
    cx, groups_in, eyes = build()
    stim_path = out / "stim_rates.npz"
    if stim_path.exists():
        rates = dict(np.load(stim_path))
    else:
        rates = render(eyes, print)
        np.savez_compressed(stim_path, **rates)
        chk = input_check(rates, eyes.tcode, list(eyes.fw_types))
        chk.update(transduction=str(TRANSDUCTION.relative_to(ROOT)), interval_ms=C.INTERVAL_MS, subframes=1,
                   n_grey=C.N_GREY, n_stim=C.N_STIM, stimuli={k: v[0] for k, v in C.STIMULI.items()})
        (out / "input_check.json").write_text(json.dumps(chk, indent=1))
    chk = json.loads((out / "input_check.json").read_text())
    print("input check", "passed" if chk["passed"] else "FAILED",
          {k: round(v["A_in_T4aT5a"], 4) for k, v in chk["per_stimulus"].items()}, flush=True)
    if not chk["passed"]:
        raise SystemExit("input check failed: stop (prereg/09 section 3)")
    del eyes

    from engine.lif_brian2 import Brian2LIFEngine
    n = cx.neurons
    dn_idx = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    dn_tab = dict(index=dn_idx, cell_type=n["cell_type"].astype(str).to_numpy()[dn_idx],
                  side=n["side"].astype(str).to_numpy()[dn_idx])
    eng = Brian2LIFEngine(cx, input_groups=groups_in, seed=0)
    print(f"engine {eng.n_neurons:,} neurons, {len(eng.syn):,} synapses; free {free_gb():.1f} GB "
          f"({time.time() - t0:.0f} s)", flush=True)
    nI = C.N_GREY + C.N_STIM
    for name in C.STIMULI:
        L, R = rates[f"{name}__L"].astype(float), rates[f"{name}__R"].astype(float)
        for s in C.SEEDS:
            path = out / "brain" / f"{name}_s{s}.npz"
            if path.exists():
                continue
            t1 = time.time()
            eng.reset(s)
            cnt = np.zeros((nI, len(dn_idx)), np.int16)
            net = np.zeros(nI, np.int64)
            st = {"prev": eng.spike_counts()}

            def on_boundary(j, e):
                if j > 0:
                    cur = e.spike_counts()
                    d = cur - st["prev"]
                    cnt[j - 1] = d[dn_idx]
                    net[j - 1] = d.sum()
                    st["prev"] = cur
                if j < nI:
                    e.set_inputs({"sugar": 0.0, "vbnd_L": L[j], "vbnd_R": R[j]})

            eng.run_coupled(nI, C.INTERVAL_MS, on_boundary)
            np.savez_compressed(path, stimulus=name, t0_condition=C.STIMULI[name][0], seed=s, dn_counts=cnt,
                                net_counts=net, input_mean_L=L.mean(1), input_mean_R=R.mean(1), interval_ms=C.INTERVAL_MS,
                                weights_sha256=eng.state_dict()["weights_sha256"], model_sha256=eng.model_sha256,
                                **{f"dn_{k}": v for k, v in dn_tab.items()})
            lo, hi = C.WINDOW
            print(f"{name} s{s}: DN spikes {int(cnt.sum())}, net {net[lo:hi].sum() / (eng.n_neurons * 1.0):.2f} Hz "
                  f"({time.time() - t1:.0f} s, free {free_gb():.1f} GB)", flush=True)
    (out / "brain" / "DONE").write_text(json.dumps(dict(wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb())) + "\n")
    print("done", round(time.time() - t0), "s", flush=True)
    return 0


def peak_rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
