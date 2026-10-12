"""F0 bisection step c2: drive the v1 engine (read-only scratch copy) and the new engine with the SAME recorded
perch + cut input arrays (drive.npz from f0_bisect_inputs.py), then idle for N intervals.

    --engine v1 --v1-root <scratch copy> | --engine new [--delay]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=("v1", "new"), required=True)
ap.add_argument("--v1-root")
ap.add_argument("--drive", required=True)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--idle", type=int, default=12)
ap.add_argument("--delay", action="store_true")
ap.add_argument("--v1-restore", action="store_true", help="v1: store right after the build and restore before driving")
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = np.load(a.drive)
n_in = len(d["L"])
rates = []
if a.engine == "v1":
    sys.path.insert(0, a.v1_root)
    from flight.brain import FlightBrain
    b = FlightBrain(seed=a.seed, olfaction=False, vision_boundary=True, verbose=False)
    assert np.array_equal(b.groups["vbnd_L"], d["idx_L"]) and np.array_equal(b.groups["vbnd_R"], d["idx_R"])
    if a.v1_restore:
        from brian2 import seed as bseed
        b.net.store("fresh")
        b.net.restore("fresh")
        bseed(a.seed)
    for k in range(n_in):
        b.set_rates(sugar=float(d["sugar"][k]), vbnd_L=d["L"][k], vbnd_R=d["R"][k])
        rates.append(float(b.step()[1].sum() / (b.n * 0.025)))
    b.silence_inputs()
    for k in range(a.idle):
        rates.append(float(b.step()[1].sum() / (b.n * 0.025)))
    names = sorted(b.inputs)
else:
    sys.path.insert(0, str(ROOT))
    from connectome.adapters import flywire783 as fw
    from engine.lif_brian2 import Brian2LIFEngine
    cx = fw.load_connectome(hash_files=False)
    eng = Brian2LIFEngine(cx, input_groups={"sugar": d["idx_sugar"], "vbnd_L": d["idx_L"], "vbnd_R": d["idx_R"]},
                          seed=a.seed, poisson_delay_v1=a.delay)
    eng.reset(a.seed)
    for k in range(n_in):
        eng.set_inputs({"sugar": float(d["sugar"][k]), "vbnd_L": d["L"][k], "vbnd_R": d["R"][k]})
        b0 = eng.spike_counts().sum()
        eng.advance(25.0)
        rates.append(float((eng.spike_counts().sum() - b0) / (eng.n_neurons * 0.025)))
    eng.set_inputs({"sugar": 0.0, "vbnd_L": 0.0, "vbnd_R": 0.0})
    for k in range(a.idle):
        b0 = eng.spike_counts().sum()
        eng.advance(25.0)
        rates.append(float((eng.spike_counts().sum() - b0) / (eng.n_neurons * 0.025)))
    names = sorted(eng._poisson)
out = dict(engine=a.engine, delay=a.delay, seed=a.seed, n_drive_intervals=n_in, inputs=names,
           net_rate_hz=[round(r, 4) for r in rates])
Path(a.out).write_text(json.dumps(out, indent=1))
print(a.engine, "drive", np.round(rates[:n_in], 2).tolist(), "idle", np.round(rates[n_in:], 3).tolist())
