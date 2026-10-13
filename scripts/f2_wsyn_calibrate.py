"""Calibration and sanity gates of prereg/03_banc_wsyn.md (open loop; no behaviour, no F2 tests).

    env -u PYTHONPATH python scripts/f2_wsyn_calibrate.py --stage flywire --out runs/f2/flywire_ref.json
    env -u PYTHONPATH python scripts/f2_wsyn_calibrate.py --stage banc --out runs/f2/banc_calib.json

flywire: gate-3 reference rates (labellar / leg sugar GRNs at 100 Hz) and the validation of the operational
         definition of Shiu's criterion on the FlyWire graph (c(s) = r(100) / max_f r(f) at s = 1).
banc:    s_B by bisection (variant B), then for R, A, A2, B: gate 1 (no input), gate 2 (ignition, 20 seeds),
         and the sugar-drive run list (labellar, leg, labellar-left, labellar-right; MN9 per neuron and side).
Seeds: calibration 1000-1001, gate 1 seeds 0-4, gate 2 seeds 200-219, run list seeds 0-2 (prereg section 4-6).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402

STEP_MS = 25.0
FREQS = tuple(range(20, 201, 20))
CAL_SEEDS = (1000, 1001)
GATE1_SEEDS = tuple(range(5))
GATE2_SEEDS = tuple(range(200, 220))
RUN_SEEDS = (0, 1, 2)
TARGET, TOL, S_MIN, S_MAX = 0.80, 0.02, 1.0, 12.0
MN9_FLYWIRE_ID = 720575940660219265
CLASSES = ("sensory", "central", "optic", "visual_projection", "descending", "ascending", "motor")
BANC_SC = {"optic_lobe_intrinsic": "optic", "central_brain_intrinsic": "central"}


def peak_rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def run_once(eng, drive, seed, n_on=40, n_cut=0, n_idle=0):
    """n_on steps with ``drive``, then n_cut steps with all inputs 0, then n_idle. Returns per-step network rates
    (Hz, all neurons) and the spike counts of the driven phase."""
    eng.reset(seed)
    eng.set_inputs(drive)
    before = eng.spike_counts()
    n = eng.n_neurons
    steps = []
    for k in range(n_on + n_cut + n_idle):
        if k == n_on:
            eng.set_inputs({g: 0.0 for g in drive})
            on_counts = eng.spike_counts() - before
        b = eng.spike_counts().sum()
        eng.advance(STEP_MS)
        steps.append(float((eng.spike_counts().sum() - b) / (n * STEP_MS * 1e-3)))
    if n_cut + n_idle == 0:
        on_counts = eng.spike_counts() - before
    return steps, on_counts


def drive_phase(eng, drive, seed, mn9, driven_idx, dur_steps=40):
    steps, c = run_once(eng, drive, seed, n_on=dur_steps)
    dur = dur_steps * STEP_MS * 1e-3
    nd = np.ones(eng.n_neurons, bool)
    nd[driven_idx] = False
    return dict(seed=seed, mn9_rate_hz_per_neuron=(c[mn9] / dur).tolist(), mn9_rate_hz=float(c[mn9].mean() / dur),
                network_rate_hz=float(c.sum() / (eng.n_neurons * dur)),
                nondriven_rate_hz=float(c[nd].sum() / (nd.sum() * dur)),
                nondriven_rate_hz_per_driven_grn=float(c[nd].sum() / (nd.sum() * dur) / max(len(driven_idx), 1)),
                n_driven=int(len(driven_idx)))


def criterion(eng, labellar_groups, mn9, seeds=CAL_SEEDS):
    """c(s) of prereg section 4 at the engine's current scale: r(100) / max over FREQS of the MN9 rate."""
    r = {}
    for f in FREQS:
        drive = {g: float(f) for g in labellar_groups}
        r[f] = float(np.mean([drive_phase(eng, drive, s, mn9, np.concatenate([eng.groups[g] for g in labellar_groups]))
                              ["mn9_rate_hz"] for s in seeds]))
    rmax = max(r.values())
    c = r[100] / rmax if rmax > 0 else None
    return dict(c=c, r_by_freq=r, r100=r[100], rmax=rmax)


def find_scale(eng, labellar_groups, mn9, apply_scale, log=print):
    """Smallest s in [S_MIN, S_MAX] with c(s) >= TARGET (bisection on log s, tolerance TOL); None if none."""
    trace = []

    def c_at(s):
        apply_scale(s)
        t0 = time.time()
        d = criterion(eng, labellar_groups, mn9)
        d.update(s=float(s), wall_s=time.time() - t0)
        trace.append(d)
        log(f"  s = {s:.4f}: c = {d['c']}, r100 {d['r100']:.1f}, rmax {d['rmax']:.1f} ({d['wall_s']:.0f} s)", flush=True)
        return d["c"] is not None and d["c"] >= TARGET

    if c_at(S_MIN):
        return S_MIN, trace
    if not c_at(S_MAX):
        return None, trace
    lo, hi = S_MIN, S_MAX
    while hi / lo > 1 + TOL:
        mid = float(np.sqrt(lo * hi))
        if c_at(mid):
            hi = mid
        else:
            lo = mid
    return hi, trace


def stage_flywire(a):
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=True)
    n = cx.neurons
    sugar = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    leg_t = pd.read_csv(ROOT / "data" / "leg_sugar_grn_783.csv")
    leg = leg_t.loc[leg_t["selected"], "root_id"].to_numpy()
    groups = dict(labellar=cx.index.group(sugar), leg=cx.index.group(leg))
    mn9 = cx.index.group(np.array([MN9_FLYWIRE_ID]))
    eng = Brian2LIFEngine(cx, input_groups=groups, groups={"mn9": mn9}, seed=0)
    res = dict(stage="flywire", n_neurons=eng.n_neurons, groups={k: len(v) for k, v in groups.items()},
               mn9=int(len(mn9)), model_sha256=eng.model_sha256, runs=[])
    for cname, drive in dict(labellar=dict(labellar=100.0, leg=0.0), leg=dict(labellar=0.0, leg=100.0)).items():
        driven = groups["labellar" if cname == "labellar" else "leg"]
        for seed in RUN_SEEDS:
            r = drive_phase(eng, drive, seed, mn9, driven)
            r["condition"] = cname
            res["runs"].append(r)
            print("flywire", cname, seed, f"MN9 {r['mn9_rate_hz']:.1f} Hz net {r['network_rate_hz']:.3f} Hz", flush=True)
    # validation of the operational definition (prereg section 4): s = 1 is the Shiu graph at w_syn = 0.275 mV
    eng.set_weight_scale(1.0)
    v = criterion(eng, ["labellar"], mn9)
    res["validation"] = dict(s=1.0, **v, passes=bool(v["c"] is not None and v["c"] >= TARGET))
    print("flywire validation c(1) =", v["c"], flush=True)
    # supplement (not part of the registered validation): c at s = 0.9
    eng.set_weight_scale(0.9)
    res["supplement_c_at_0p9"] = criterion(eng, ["labellar"], mn9)
    res.update(peak_rss_gb=peak_rss_gb())
    return res


def stage_flywire_search(a):
    """The registered validation of prereg section 4 in full: the same bisection as variant B, on the FlyWire graph.
    The validation passes if s_B(FlyWire) lies within [0.9, 1.1]."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=True)
    sugar = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    groups = dict(labellar=cx.index.group(sugar))
    mn9 = cx.index.group(np.array([MN9_FLYWIRE_ID]))
    eng = Brian2LIFEngine(cx, input_groups=groups, groups={"mn9": mn9}, seed=0)
    s_b, trace = find_scale(eng, ["labellar"], mn9, eng.set_weight_scale)
    return dict(stage="flywire_search", s_flywire=s_b, trace=trace, window=[0.9, 1.1],
                passes=bool(s_b is not None and 0.9 <= s_b <= 1.1), peak_rss_gb=peak_rss_gb())


def banc_setup():
    from connectome.adapters import banc
    cx = banc.load_connectome(ROOT / "banc" / "neurons-banc.csv.gz", ROOT / "banc" / "connections_princeton-banc.csv.gz",
                              hash_files=True)
    n = cx.neurons
    sugar = n["function"].astype(str).str.contains("sugar") & n["super_class"].isin(["sensory", "sensory_ascending"])
    lab = sugar & (n["body_part"] == "labellum")
    leg = sugar & n["body_part"].isin(["front_leg", "middle_leg", "hind_leg"])
    side = n["side"]
    groups = {}
    for nm, m in (("labellar", lab), ("leg", leg)):
        for sd in ("left", "right"):
            groups[f"{nm}_{sd[0].upper()}"] = np.flatnonzero((m & (side == sd)).to_numpy())
    mn9 = np.flatnonzero((n["cell_type"] == "MN9").to_numpy())
    return cx, groups, mn9


def class_factors(cx, kc, kg):
    sc = cx.neurons["super_class"].replace(BANC_SC).astype(object).to_numpy()
    f = np.full(cx.n_neurons, kg, np.float64)
    for c in CLASSES:
        f[sc == c] = kc[c]
    return f


def variant_report(eng, cx, groups, mn9, label, log=print, gate1_seeds=GATE1_SEEDS, gate2_seeds=GATE2_SEEDS,
                   run_seeds=RUN_SEEDS):
    sides = cx.neurons["side"].to_numpy()[mn9].tolist()
    allg = np.concatenate(list(groups.values()))
    out = dict(variant=label, state=eng.state_dict()["weight_scale"], post_scale_sha256=eng.state_dict()["post_scale_sha256"],
               weights_sha256=eng.state_dict()["weights_sha256"], mn9_sides=sides)
    t0 = time.time()
    # gate 1: no input
    g1 = []
    for s in gate1_seeds:
        steps, c = run_once(eng, {g: 0.0 for g in groups}, s, n_on=40)
        g1.append(int(c.sum()))
    out["gate1_no_input_spikes_per_seed"] = g1
    out["gate1_pass"] = bool(sum(g1) == 0)
    # gate 2: ignition after the cut, labellar + leg both sides at 100 Hz (prereg section 5)
    g2 = []
    for s in gate2_seeds:
        steps, _ = run_once(eng, {g: 100.0 for g in groups}, s, n_on=10, n_cut=8, n_idle=3)
        g2.append(dict(seed=s, last_cut_step_hz=steps[17], ignited=bool(steps[17] > 1.0), idle_mean_hz=float(np.mean(steps[18:]))))
    out["gate2_runs"] = g2
    out["gate2_ignited"] = int(sum(r["ignited"] for r in g2))
    out["gate2_pass"] = bool(out["gate2_ignited"] == 0)
    # run list
    conds = dict(labellar={"labellar_L": 100.0, "labellar_R": 100.0}, leg={"leg_L": 100.0, "leg_R": 100.0},
                 labellar_left={"labellar_L": 100.0}, labellar_right={"labellar_R": 100.0})
    out["runs"] = []
    for cname, on in conds.items():
        drive = {g: on.get(g, 0.0) for g in groups}
        driven = np.concatenate([groups[g] for g in on])
        for s in run_seeds:
            r = drive_phase(eng, drive, s, mn9, driven)
            r["condition"] = cname
            out["runs"].append(r)
    out["wall_s"] = time.time() - t0
    log(f"{label}: gate1 {out['gate1_pass']} gate2 {out['gate2_ignited']}/{len(gate2_seeds)} "
        + ", ".join(f"{c} MN9 {np.mean([r['mn9_rate_hz'] for r in out['runs'] if r['condition'] == c]):.1f} Hz"
                    for c in conds) + f" ({out['wall_s']:.0f} s)", flush=True)
    return out


def stage_banc(a):
    import yaml
    cfg = {k: v for g in yaml.safe_load((ROOT / "configs" / "banc_wsyn.yaml").read_text())["groups"]
           for k, v in g["values"].items()}
    kg = cfg["wsyn_struct_k_global"]
    kc = {c: cfg[f"wsyn_struct_class_k_{c}"] for c in CLASSES}
    cx, groups, mn9 = banc_setup()
    eng = Brian2LIFEngine(cx, input_groups=groups, groups={"mn9": mn9}, seed=0)
    res = dict(stage="banc", n_neurons=eng.n_neurons, groups={k: len(v) for k, v in groups.items()}, mn9=int(len(mn9)),
               model_sha256=eng.model_sha256, k_global=kg, k_class=kc, variants={})
    lab = ["labellar_L", "labellar_R"]
    # variant B: Shiu's criterion (prereg section 4)
    print("variant B: bisection ...", flush=True)
    sB, trace = find_scale(eng, lab, mn9, eng.set_weight_scale)
    res["B_search"] = dict(s_B=sB, trace=trace, target=TARGET, tol=TOL, bounds=[S_MIN, S_MAX], seeds=list(CAL_SEEDS))
    print("s_B =", sB, flush=True)
    plan = [("R", lambda: eng.set_weight_scale(1.0)), ("A_wsyn_struct", lambda: eng.set_weight_scale(kg)),
            ("A2_wsyn_struct_class", lambda: eng.set_weight_scale_per_post(class_factors(cx, kc, kg)))]
    if sB is not None:
        plan.append(("B_wsyn_fit_mn9", lambda: eng.set_weight_scale(sB)))
    for label, setter in plan:
        setter()
        res["variants"][label] = variant_report(eng, cx, groups, mn9, label)
        Path(a.out).with_suffix(".partial.json").write_text(json.dumps(res, indent=1))
    res["peak_rss_gb"] = peak_rss_gb()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("flywire", "flywire_search", "banc"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    res = dict(flywire=stage_flywire, flywire_search=stage_flywire_search, banc=stage_banc)[a.stage](a)
    res["total_wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
