"""F0 brain-mismatch bisection, step a: open loop, no body (docs/results/f0_brain_mismatch.md).

One protocol for three engines: sugar GRNs (36) at 100 Hz for 1 s in 25 ms steps, then all inputs cut for 200 ms;
3 seeds; reports the MN9 (CB0701) rate, the network rate, the DN rate and the network rate in the last 100 ms of the
cut. The inputs that exist in the n1 configuration (t45 groups at 0 Hz keep rfc = 0 in v1) are declared in all
engines.

    --engine v1          the v1 FlightBrain from a read-only scratch copy (--v1-root)
    --engine new         engine/lif_brian2.py (Poisson input without delay, review M4)
    --engine new_delay   the same with the named variant poisson_delay_v1 (1.8 ms delay on the input)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEP_MS, N_ON, N_CUT = 25.0, 40, 8


def protocol(step, counts, set_sugar, silence, reset, seeds, groups):
    out = []
    for seed in seeds:
        reset(seed)
        tot_on = np.zeros(groups["n"])
        for k in range(N_ON):
            set_sugar(100.0)
            tot_on += step()
        last = []
        silence()
        for k in range(N_CUT):
            last.append(step())
        n = groups["n"]
        dur = N_ON * STEP_MS * 1e-3
        out.append(dict(
            seed=seed, net_rate_hz=float(tot_on.sum() / (n * dur)),
            mn9_rate_hz=float(tot_on[groups["mn9"]].mean() / dur),
            dn_rate_hz=float(tot_on[groups["dn"]].mean() / dur),
            sugar_rate_hz=float(tot_on[groups["sugar"]].mean() / dur),
            cut_last100ms_net_hz=float(np.sum(last[-4:]) / (n * 0.1)),
            cut_first25ms_net_hz=float(np.sum(last[0]) / (n * 0.025))))
    return out


def run_v1(v1_root, seeds):
    sys.path.insert(0, str(v1_root))
    from flight import groups as G
    from flight.brain import FlightBrain
    import pandas as pd
    b = FlightBrain(seed=seeds[0], olfaction=False, verbose=False)
    b.net.store("fresh")
    from flight.brain import SpikeCounter
    ann = pd.read_csv(G.PATH_ANN, sep="\t", low_memory=False, usecols=["root_id", "cell_type"])
    r2i = G.root_to_index(b.root_ids)
    mn9 = np.array([r2i[r] for r in ann.loc[ann["cell_type"] == "CB0701", "root_id"] if r in r2i])
    groups = dict(n=b.n, mn9=mn9, dn=np.concatenate([b.groups["dn_L"], b.groups["dn_R"], b.groups["dn_C"]]),
                  sugar=b.groups["sugar"])
    from brian2 import seed as bseed

    def reset(seed):
        b.net.restore("fresh")
        bseed(seed)
        b.counter = SpikeCounter(b.spk_mon, b.n)
        b.silence_inputs()

    def step():
        return b.step()[1].copy()
    return protocol(step, None, lambda r: b.set_rates(sugar=r), b.silence_inputs, reset, seeds, groups), \
        dict(engine="v1", n_neurons=b.n, inputs=sorted(b.inputs))


def run_new(delay, seeds):
    sys.path.insert(0, str(ROOT))
    from connectome.adapters import flywire783 as fw
    from engine.lif_brian2 import Brian2LIFEngine
    import pandas as pd
    cx = fw.load_connectome(hash_files=False)
    n = cx.neurons
    sugar = cx.index.group(pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy())
    col = pd.read_csv(ROOT / "data" / "column_assignment.csv.gz")
    t45 = col[col["type"].isin(["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])]
    inp = {"sugar": sugar}
    for s, sd in (("L", "left"), ("R", "right")):
        inp[f"t45_{s}"] = cx.index.group(t45.loc[t45["hemisphere"] == sd, "root_id"].to_numpy())
    eng = Brian2LIFEngine(cx, input_groups=inp, seed=seeds[0], poisson_delay_v1=delay)
    mn9 = cx.by_type(["CB0701"])
    dn = cx.select((n["super_class"] == "descending").to_numpy())
    groups = dict(n=eng.n_neurons, mn9=mn9, dn=dn, sugar=sugar)

    def step():
        before = eng.spike_counts()
        eng.advance(STEP_MS)
        return eng.spike_counts() - before

    def reset(seed):
        eng.reset(seed)

    def set_sugar(r):
        eng.set_inputs({"sugar": r})

    def silence():
        eng.set_inputs({"sugar": 0.0})
    return protocol(step, None, set_sugar, silence, reset, seeds, groups), \
        dict(engine="new_delay" if delay else "new", n_neurons=eng.n_neurons, inputs=sorted(inp),
             state=dict(variant="poisson_delay_v1" if delay else None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("v1", "new", "new_delay"), required=True)
    ap.add_argument("--v1-root", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    t = time.time()
    res, info = run_v1(Path(a.v1_root), a.seeds) if a.engine == "v1" else run_new(a.engine == "new_delay", a.seeds)
    doc = dict(info, results=res, wall_s=time.time() - t,
               mean={k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "seed"})
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["mean"], indent=1))


if __name__ == "__main__":
    main()
