"""Read-only diagnostic of the BANC smoke run (why is MN9 silent?). No parameter is changed.

Reports: hop distance from the driven sugar GRNs to MN9 on the BANC graph and, for comparison, from the Shiu
LB3 sugar GRNs to MN9 on FlyWire v783; the MN9 input synapses by presynaptic class; and from one engine run
(labellar + leg at 100 Hz, 1 s, seed 0) how many neurons other than the driven ones fire, by hop distance.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome.adapters import banc, flywire783 as fw  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402


def hops(cx, seeds, max_hops=8):
    pre, post = cx.edge_endpoints()
    A = coo_matrix((np.ones(len(pre), np.int8), (pre, post)), shape=(cx.n_neurons,) * 2).tocsr()
    dist = np.full(cx.n_neurons, -1)
    dist[seeds] = 0
    frontier = np.zeros(cx.n_neurons, bool)
    frontier[seeds] = True
    for h in range(1, max_hops + 1):
        nxt = (A.T @ frontier.astype(np.int32)) > 0
        nxt &= dist < 0
        dist[nxt] = h
        frontier = nxt
        if not nxt.any():
            break
    return dist


def main():
    out = {}
    cx = banc.load_connectome(ROOT / "banc/neurons-banc.csv.gz", ROOT / "banc/connections_princeton-banc.csv.gz",
                              hash_files=False)
    n = cx.neurons
    sugar = n["function"].astype(str).str.contains("sugar") & n["super_class"].isin(["sensory", "sensory_ascending"])
    lab = np.flatnonzero((sugar & (n["body_part"] == "labellum")).to_numpy())
    leg = np.flatnonzero((sugar & n["body_part"].isin(["front_leg", "middle_leg", "hind_leg"])).to_numpy())
    mn9 = np.flatnonzero((n["cell_type"] == "MN9").to_numpy())
    both = np.concatenate([lab, leg])
    d_lab, d_leg = hops(cx, lab), hops(cx, leg)
    out["banc"] = dict(n_labellar=len(lab), n_leg=len(leg), mn9_hops_from_labellar=d_lab[mn9].tolist(),
                       mn9_hops_from_leg=d_leg[mn9].tolist(),
                       reachable_within_hops_labellar={int(h): int((d_lab == h).sum()) for h in range(1, 7)})
    _, post = cx.edge_endpoints()
    tot = np.bincount(post, weights=cx.edges["n_syn"].to_numpy(float), minlength=cx.n_neurons)
    out["banc"]["mn9_input_synapses"] = tot[mn9].tolist()
    pre, post = cx.edge_endpoints()
    m = np.isin(post, mn9)
    top = (pd.DataFrame({"pre": pre[m], "w": cx.edges["n_syn"].to_numpy()[m]}).groupby("pre")["w"].sum()
           .sort_values(ascending=False).head(10))
    out["banc"]["mn9_top_presynaptic"] = [dict(type=str(n["cell_type"].iloc[i]), super_class=str(n["super_class"].iloc[i]),
                                               syn=int(w)) for i, w in top.items()]
    # engine: who fires besides the driven neurons
    eng = Brian2LIFEngine(cx, input_groups={"labellar": lab, "leg": leg}, seed=0)
    eng.set_inputs({"labellar": 100.0, "leg": 100.0})
    for _ in range(40):
        eng.advance(25.0)
    c = eng.spike_counts()
    driven = np.zeros(cx.n_neurons, bool)
    driven[both] = True
    d_both = hops(cx, both)
    act = (c > 0) & ~driven
    out["banc"]["engine_both_1s"] = dict(
        driven_neurons=int(driven.sum()), driven_mean_rate_hz=float(c[both].mean()),
        other_neurons_spiking=int(act.sum()), other_spikes=int(c[~driven].sum()),
        other_spiking_by_hop={int(h): int((act & (d_both == h)).sum()) for h in range(1, 6)},
        mn9_spikes=int(c[mn9].sum()))
    del eng
    f = fw.load_connectome(hash_files=False)
    ids = pd.read_csv(ROOT / "data/sugar_grn_783.csv")["root_id"].to_numpy()
    s = f.index.group(ids)
    fmn9 = f.by_type(["CB0701"])
    dl = hops(f, s)
    out["flywire"] = dict(n_sugar=len(s), mn9_hops=dl[fmn9].tolist(),
                          reachable_within_hops={int(h): int((dl == h).sum()) for h in range(1, 7)})
    _, fpost = f.edge_endpoints()
    ftot = np.bincount(fpost, weights=f.edges["n_syn"].to_numpy(float), minlength=f.n_neurons)
    out["flywire"]["mn9_input_synapses"] = ftot[fmn9].tolist()
    p = Path(sys.argv[1])
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
