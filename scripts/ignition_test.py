"""Pre-registered ignition-frequency test (prereg/02_ignition_frequency.md): one arm, seeds 100-199.

    env -u PYTHONPATH python scripts/ignition_test.py --arm v1|new|new_delay [--v1-root <scratch copy>]

Appends one JSON line per seed to runs/ignition/<arm>.jsonl (a seed already present is skipped, never rewritten).
Primary endpoint: network rate in the last cut step (interval 17) > 1 Hz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRIVE = ROOT / "runs" / "bisect" / "f0_drive.npz"
DRIVE_SHA256 = "ca1c77ec7f05926b6b7a1b52da6e3693e913a9b47082396a9ffe042e59fee8e9"
N_CUT, N_IDLE, SEEDS = 8, 3, range(100, 200)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("v1", "new", "new_delay"), required=True)
    ap.add_argument("--v1-root")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "ignition"))
    a = ap.parse_args(argv)
    import hashlib
    assert hashlib.sha256(DRIVE.read_bytes()).hexdigest() == DRIVE_SHA256, "drive file differs from the registered one"
    d = np.load(DRIVE)
    n_in = len(d["L"])
    out = Path(a.out_dir) / f"{a.arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["seed"] for l in out.read_text().splitlines() if l.strip()}
    todo = [s for s in SEEDS if s not in done]
    if not todo:
        print("nothing to do")
        return
    if a.arm == "v1":
        sys.path.insert(0, a.v1_root)
        from brian2 import seed as bseed
        from flight.brain import FlightBrain, SpikeCounter
        b = FlightBrain(seed=todo[0], olfaction=False, vision_boundary=True, verbose=False)
        assert np.array_equal(b.groups["vbnd_L"], d["idx_L"]) and np.array_equal(b.groups["vbnd_R"], d["idx_R"])
        b.net.store("fresh")
        n_all = b.n

        def run(seed):
            b.net.restore("fresh")
            bseed(seed)
            b.counter = SpikeCounter(b.spk_mon, b.n)
            b.silence_inputs()
            r = []
            for k in range(n_in):
                b.set_rates(sugar=float(d["sugar"][k]), vbnd_L=d["L"][k], vbnd_R=d["R"][k])
                r.append(float(b.step()[1].sum() / (n_all * 0.025)))
            b.silence_inputs()
            for _ in range(N_IDLE):
                r.append(float(b.step()[1].sum() / (n_all * 0.025)))
            return r
    else:
        sys.path.insert(0, str(ROOT))
        from connectome.adapters import flywire783 as fw
        from engine.lif_brian2 import Brian2LIFEngine
        cx = fw.load_connectome(hash_files=False)
        eng = Brian2LIFEngine(cx, input_groups={"sugar": d["idx_sugar"], "vbnd_L": d["idx_L"], "vbnd_R": d["idx_R"]},
                              seed=todo[0], poisson_delay_v1=(a.arm == "new_delay"))
        n_all = eng.n_neurons

        def run(seed):
            eng.reset(seed)
            r = []
            for k in range(n_in):
                eng.set_inputs({"sugar": float(d["sugar"][k]), "vbnd_L": d["L"][k], "vbnd_R": d["R"][k]})
                b0 = eng.spike_counts().sum()
                eng.advance(25.0)
                r.append(float((eng.spike_counts().sum() - b0) / (n_all * 0.025)))
            eng.set_inputs({"sugar": 0.0, "vbnd_L": 0.0, "vbnd_R": 0.0})
            for _ in range(N_IDLE):
                b0 = eng.spike_counts().sum()
                eng.advance(25.0)
                r.append(float((eng.spike_counts().sum() - b0) / (n_all * 0.025)))
            return r
    t0 = time.time()
    with open(out, "a") as f:
        for s in todo:
            r = run(s)
            last_cut = r[17]            # interval 17 = the 8th cut interval (10 perch + 8 cut)
            idle = r[18:18 + N_IDLE]
            f.write(json.dumps(dict(arm=a.arm, seed=s, last_cut_hz=last_cut, idle_mean_hz=float(np.mean(idle)),
                                    ignited_primary=bool(last_cut > 1.0),
                                    ignited_secondary=bool(np.mean(idle) > 0.5), rates=[round(x, 4) for x in r]))
                    + "\n")
            f.flush()
            print(a.arm, s, round(last_cut, 3), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
