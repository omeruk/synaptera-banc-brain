"""Structural k for the BANC w_syn pre-registration (data only, no simulation).

k = FlyWire / BANC ratio of the mean number of input synapses per postsynaptic neuron, on the graphs as the engine
runs them (``as_run``; ``as_run_brain_neuropils`` counts BANC inputs in brain / SEZ neuropils only, as FlyWire is brain only) (FlyWire: Shiu model, every pair; BANC: the Codex export, pairs with >= 3 synapses), overall and per
postsynaptic super class, on a matched population: neurons whose cell type name exists in both datasets and whose
super class (after the name mapping below) is the same. Sensitivity views: all neurons; both graphs at >= 3; both
graphs at >= 5.

    env -u PYTHONPATH python scripts/f2_wsyn_k.py --out docs/results/f2_wsyn_k.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome.adapters import banc, flywire783 as fw  # noqa: E402

CLASSES = ("sensory", "central", "optic", "visual_projection", "descending", "ascending", "motor")
BANC_SC = {"optic_lobe_intrinsic": "optic", "central_brain_intrinsic": "central"}


def in_synapses(cx, min_syn=0):
    _, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.int64)
    keep = w >= min_syn
    return np.bincount(post[keep], weights=w[keep], minlength=cx.n_neurons)


def ratio_table(fx, bx, fsc, bsc, ftype, btype, min_syn, bin_override=None):
    fin = in_synapses(fx, min_syn)
    bin_ = in_synapses(bx, min_syn) if bin_override is None else bin_override
    common = set(pd.unique(ftype[pd.notna(ftype)])) & set(pd.unique(btype[pd.notna(btype)]))
    out = {}
    for view in ("matched", "all"):
        res = {}
        for c in ("__all__",) + CLASSES:
            fm = (fsc == c) if c != "__all__" else np.ones(len(fsc), bool)
            bm = (bsc == c) if c != "__all__" else np.ones(len(bsc), bool)
            if view == "matched":
                fm = fm & pd.Series(ftype).isin(common).to_numpy()
                bm = bm & pd.Series(btype).isin(common).to_numpy()
            nf, nb = int(fm.sum()), int(bm.sum())
            if min(nf, nb) < 1:
                res[c] = dict(n_flywire=nf, n_banc=nb, k=None)
                continue
            mf, mb = float(fin[fm].mean()), float(bin_[bm].mean())
            res[c] = dict(n_flywire=nf, n_banc=nb, mean_in_syn_flywire=mf, mean_in_syn_banc=mb,
                          k=mf / mb if mb > 0 else None)
        out[view] = res
    out["n_common_types"] = len(common)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists")
    fx = fw.load_connectome(hash_files=False)
    bx = banc.load_connectome(ROOT / "banc" / "neurons-banc.csv.gz",
                              ROOT / "banc" / "connections_princeton-banc.csv.gz", hash_files=False)
    fsc = fx.neurons["super_class"].astype(object).to_numpy()
    bsc = bx.neurons["super_class"].replace(BANC_SC).astype(object).to_numpy()
    ftype = fx.neurons["cell_type"].astype(object).to_numpy()
    btype = bx.neurons["cell_type"].astype(object).to_numpy()
    # BANC input synapses counted in brain and SEZ neuropils only: the FlyWire v783 graph is brain only, so the
    # dendritic inputs that BANC neurons receive in the VNC (ascending, motor, VNC-intrinsic) have no counterpart
    conn = banc.load_connections(ROOT / "banc" / "connections_princeton-banc.csv.gz")
    reg = conn["neuropil"].map({n: banc.region_of_neuropil(n) for n in conn["neuropil"].unique()})
    keep = reg.isin(["brain", "SEZ"]).to_numpy()
    post_idx = bx.index.index_of(conn["post_root_id"].to_numpy()[keep])
    bin_brain = np.bincount(post_idx, weights=conn["syn_count"].to_numpy()[keep], minlength=bx.n_neurons)
    res = dict(
        definition="k = mean input synapses per postsynaptic neuron, FlyWire / BANC",
        graphs=dict(flywire=dict(pairs=len(fx.edges), synapses=int(fx.edges["n_syn"].sum())),
                    banc=dict(pairs=len(bx.edges), synapses=int(bx.edges["n_syn"].sum()))),
        as_run=ratio_table(fx, bx, fsc, bsc, ftype, btype, 0),
        as_run_brain_neuropils=ratio_table(fx, bx, fsc, bsc, ftype, btype, 0, bin_brain),
        both_ge3=ratio_table(fx, bx, fsc, bsc, ftype, btype, 3),
        both_ge5=ratio_table(fx, bx, fsc, bsc, ftype, btype, 5))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for view in ("as_run", "as_run_brain_neuropils", "both_ge3", "both_ge5"):
        for pop in ("matched", "all"):
            print(view, pop, {c: (round(v["k"], 2) if v["k"] else None) for c, v in res[view][pop].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
