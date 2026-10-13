"""F1 open-loop smoke run on BANC with the reference Shiu parameters (w_syn unchanged, reference sign rule).

Sugar GRNs (labellar; leg; both; and a no-input control) at 100 Hz for 1 s in 25 ms steps, then all inputs cut for
200 ms; 3 seeds. Reports MN9 rate, network rate, the activity after the cut, peak RAM and wall time. Any
recalibration of w_syn for BANC is a pre-registered F2 decision, not made here.

    env -u PYTHONPATH python scripts/f1_banc_smoke.py --out runs/f1/smoke.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome.adapters import banc  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402

STEP_MS, N_ON, N_CUT = 25.0, 40, 8


def peak_rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def groups(cx):
    n = cx.neurons
    sugar = n["function"].astype(str).str.contains("sugar") & n["super_class"].isin(["sensory", "sensory_ascending"])
    lab = (sugar & (n["body_part"] == "labellum")).to_numpy()
    leg = (sugar & n["body_part"].isin(["front_leg", "middle_leg", "hind_leg"])).to_numpy()
    return dict(labellar=np.flatnonzero(lab), leg=np.flatnonzero(leg),
                mn9=np.flatnonzero((n["cell_type"] == "MN9").to_numpy()),
                dn=np.flatnonzero((n["super_class"] == "descending").to_numpy()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-syn", type=int, default=None,
                    help="keep only pairs with >= this many synapses (sensitivity run; default: graph as exported)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    cx = banc.load_connectome(ROOT / "banc" / "neurons-banc.csv.gz", ROOT / "banc" / "connections_princeton-banc.csv.gz",
                              hash_files=True, min_syn=a.min_syn)
    g = groups(cx)
    inp = {"labellar": g["labellar"], "leg": g["leg"]}
    eng = Brian2LIFEngine(cx, input_groups=inp, groups={"mn9": g["mn9"], "dn": g["dn"]}, seed=0)
    n_all = eng.n_neurons
    setup_s = time.time() - t0
    conds = {"labellar": dict(labellar=100.0, leg=0.0), "leg": dict(labellar=0.0, leg=100.0),
             "both": dict(labellar=100.0, leg=100.0), "none": dict(labellar=0.0, leg=0.0)}
    runs = []
    for cname, drive in conds.items():
        for seed in a.seeds:
            t1 = time.time()
            eng.reset(seed)
            eng.set_inputs(drive)
            before = eng.spike_counts()
            net_steps = []
            for _ in range(N_ON):
                b = eng.spike_counts().sum()
                eng.advance(STEP_MS)
                net_steps.append(float((eng.spike_counts().sum() - b) / (n_all * STEP_MS * 1e-3)))
            on = eng.spike_counts() - before
            dur = N_ON * STEP_MS * 1e-3
            eng.set_inputs({"labellar": 0.0, "leg": 0.0})
            cut = []
            for _ in range(N_CUT):
                b = eng.spike_counts().sum()
                eng.advance(STEP_MS)
                cut.append(float((eng.spike_counts().sum() - b) / (n_all * STEP_MS * 1e-3)))
            runs.append(dict(
                condition=cname, seed=seed, mn9_rate_hz=float(on[g["mn9"]].mean() / dur),
                mn9_rate_hz_per_neuron=(on[g["mn9"]] / dur).tolist(),
                network_rate_hz=float(on.sum() / (n_all * dur)), dn_rate_hz=float(on[g["dn"]].mean() / dur),
                input_rate_hz=float(on[inp["labellar"]].mean() / dur) if drive["labellar"] else None,
                network_rate_last_200ms_of_drive_hz=float(np.mean(net_steps[-8:])),
                cut_first_step_hz=cut[0], cut_last_100ms_hz=float(np.mean(cut[-4:])), wall_s=time.time() - t1))
            print(cname, seed, f"MN9 {runs[-1]['mn9_rate_hz']:.1f} Hz, net {runs[-1]['network_rate_hz']:.2f} Hz, "
                  f"cut-last {runs[-1]['cut_last_100ms_hz']:.3f} Hz", flush=True)
    res = dict(n_neurons=n_all, n_pairs=len(eng.syn), n_synapses=int(cx.edges['n_syn'].sum()),
               graph={k: cx.meta[k] for k in ('min_syn', 'pairs_before_threshold', 'pairs_kept', 'synapses_kept')}, groups={k: len(v) for k, v in g.items()},
               params=eng.state_dict()["params"], sign_rule=cx.meta["sign_rule"], setup_s=setup_s,
               total_wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb(), model_sha256=eng.model_sha256,
               engine_weights_sha256=eng.state_dict()["weights_sha256"], runs=runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
