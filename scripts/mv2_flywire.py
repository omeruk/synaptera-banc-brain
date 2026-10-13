"""Test (i) of prereg/12_neuron_model_v2.md: the FlyWire v783 brain does not break.

    env -u PYTHONPATH python scripts/mv2_flywire.py --arm R|A|AC|ACG --out runs/mv2/t1_<arm>.json

The 36 labellar sugar GRNs (data/sugar_grn_783.csv) are driven at 100 Hz (Poisson input path of the reference arm)
for 1 s (40 steps of 25 ms), then all input is cut for 200 ms (8 steps). Seeds 300-349, a fresh state per seed.
Recorded per seed: MN9 (CB0701, both neurons) rate in the drive window, the driven GRNs' rate, the network rate per
step, the number of active neurons; ignition = network rate > 1 Hz in the last cut step. No-input check: seeds
300-302 without drive must give 0 spikes.
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
sys.path.insert(0, str(ROOT / "scripts"))
import mv2_common as M  # noqa: E402
from f3b1_brain import MIN_FREE_GB, free_gb, peak_rss_gb  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=M.ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None, help="first-last (default: the registered 300-349)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = M.T1_SEEDS if a.seeds is None else tuple(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1))
    t0 = time.time()
    print(f"free {free_gb():.1f} GB", flush=True)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb():.1f} GB available (< {MIN_FREE_GB}); stop")
    from connectome.adapters import flywire783 as fw
    from engine.lif_brian2 import Brian2LIFEngine
    cx = fw.load_connectome(hash_files=True)
    sugar = cx.index.group(pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy())
    n = cx.neurons
    mn9 = np.flatnonzero((n["cell_type"].astype(str) == M.T1_MN9_TYPE).to_numpy())
    if len(sugar) != 36 or len(mn9) != 2:
        raise SystemExit(f"expected 36 GRNs and 2 MN9, got {len(sugar)} and {len(mn9)}")
    mv = M.model_variant(a.arm, cx)
    eng = Brian2LIFEngine(cx, input_groups={"sugar": sugar}, groups={"mn9": mn9}, seed=0, model_variant=mv)
    sd = eng.state_dict()
    res = dict(stage="prereg12_t1", arm=a.arm, model_variant=sd["model_variant"],
               gap_pairs=[list(p) for p in (mv.gap_pairs if mv is not None else ())],
               n_neurons=eng.n_neurons, n_sugar=int(len(sugar)), mn9_index=mn9.tolist(),
               mn9_side=n["side"].astype(str).to_numpy()[mn9].tolist(), seeds=list(seeds),
               step_ms=M.T1_STEP_MS, n_on=M.T1_N_ON, n_cut=M.T1_N_CUT, rate_hz=M.T1_RATE_HZ,
               weights_sha256=sd["weights_sha256"], model_sha256=eng.model_sha256, runs=[], no_input=[])
    print(f"arm {a.arm}: engine {eng.n_neurons:,} neurons ({time.time() - t0:.0f} s, free {free_gb():.1f} GB)", flush=True)
    nI = M.T1_N_ON + M.T1_N_CUT
    dur_on = M.T1_N_ON * M.T1_STEP_MS * 1e-3
    step_s = M.T1_STEP_MS * 1e-3

    def run(seed, rate):
        eng.reset(seed)
        st = {"prev": eng.spike_counts(), "on": None}
        net = []

        def on_boundary(j, e):
            if j > 0:
                cur = e.spike_counts()
                net.append(float((cur - st["prev"]).sum() / (eng.n_neurons * step_s)))
                st["prev"] = cur
                if j == M.T1_N_ON:
                    st["on"] = cur.copy()
            if j < nI:
                e.set_inputs({"sugar": rate if j < M.T1_N_ON else 0.0})

        eng.run_coupled(nI, M.T1_STEP_MS, on_boundary)
        return st["on"], eng.spike_counts(), net

    for s in seeds[:3]:
        _, end, _ = run(s, 0.0)
        res["no_input"].append(dict(seed=s, spikes=int(end.sum())))
    print("no-input spikes", sum(r["spikes"] for r in res["no_input"]), flush=True)
    for s in seeds:
        t1 = time.time()
        on, end, net = run(s, M.T1_RATE_HZ)
        res["runs"].append(dict(seed=s, mn9_hz=[float(on[i] / dur_on) for i in mn9],
                                sugar_hz=float(on[sugar].sum() / (len(sugar) * dur_on)),
                                network_on_hz=float(on.sum() / (eng.n_neurons * dur_on)),
                                n_active_on=int((on > 0).sum()), cut_spikes=int((end - on).sum()),
                                network_step_hz=[round(x, 5) for x in net], last_cut_hz=net[-1],
                                ignited=bool(net[-1] > M.T1_IGNITION_HZ)))
        r = res["runs"][-1]
        print(f"{a.arm} s{s}: MN9 {r['mn9_hz']} Hz, net {r['network_on_hz']:.3f} Hz, last cut {r['last_cut_hz']:.3f} Hz "
              f"({time.time() - t1:.0f} s)", flush=True)
        out.with_suffix(".partial.json").write_text(json.dumps(res))
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print("done", round(res["total_wall_s"]), "s ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
