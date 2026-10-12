"""F0 bisection step b + interval check: full brain, 1 s open loop (sugar GRNs 100 Hz, no feedback), same seed.

Compares spike trains (t, i) exactly: per-interval ``advance`` vs ``run_coupled`` at 25 ms, and run_coupled at 5 ms
vs 25 ms. Without feedback the coupling interval must not change the spikes.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from connectome.adapters import flywire783 as fw  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402

cx = fw.load_connectome(hash_files=False)
sugar = cx.index.group(pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy())
got = {"t": [], "i": []}
eng = Brian2LIFEngine(cx, input_groups={"sugar": sugar}, seed=0,
                      spike_sink=lambda t, i: (got["t"].append(t.copy()), got["i"].append(i.copy())))


def run(mode, interval_ms, seed=5, total_ms=1000.0):
    got["t"].clear()
    got["i"].clear()
    eng.reset(seed)
    eng.set_inputs({"sugar": 100.0})
    n = int(round(total_ms / interval_ms))
    t0 = time.time()
    if mode == "advance":
        for _ in range(n):
            eng.advance(interval_ms)
    else:
        eng.run_coupled(n, interval_ms, lambda j, e: None)
    return np.concatenate(got["t"]), np.concatenate(got["i"]), time.time() - t0


res = {}
runs = {"advance_25": ("advance", 25.0), "coupled_25": ("coupled", 25.0), "coupled_5": ("coupled", 5.0),
        "advance_5": ("advance", 5.0)}
out = {k: run(*v) for k, v in runs.items()}
ref = out["advance_25"]
for k, (t, i, w) in out.items():
    res[k] = dict(n_spikes=int(len(t)), wall_s=round(w, 1),
                  identical_to_advance_25=bool(len(t) == len(ref[0]) and np.array_equal(t, ref[0])
                                               and np.array_equal(i, ref[1])))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
