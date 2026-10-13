"""Test (iv) of prereg/12_neuron_model_v2.md: giant fiber -> TTMn on the BANC v3 VNC subgraph.

    env -u PYTHONPATH python scripts/mv2_gf.py --arm R|ACG --out runs/mv2/t4_<arm>.json

Subgraph of prereg/08 (`f3a3_subgraph.vnc_subgraph`); the two GFs (DNp01, one per side, input nodes) are driven at
100 Hz (Poisson) for 1 s (40 x 25 ms), then 200 ms without input. Seeds 0-9. Recorded: spike times of GF, TTMn and
PSI; per side the TTMn / GF spike ratio in the drive window (following), the GF -> TTMn latency (each TTMn spike minus
the latest preceding spike of the same-side GF, if that GF spike is < 5 ms earlier), PSI, DLM and TTMn rates, the
network rate. No other input (no DNg02 baseline).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mv2_common as M  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb  # noqa: E402
from f3a3_subgraph import vnc_subgraph  # noqa: E402
import f3a_vnc_run as F  # noqa: E402

LAT_WINDOW_S = 0.005


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=M.ARMS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from engine.lif_brian2 import Brian2LIFEngine
    whole = F.load()
    cx, _ = vnc_subgraph(whole)
    del whole
    n = cx.neurons
    ct, side = n["cell_type"].astype(str).to_numpy(), n["side"].astype(str).to_numpy()

    def one(t, s):
        ii = np.flatnonzero((ct == t) & (side == s))
        if len(ii) != 1:
            raise SystemExit(f"{t} {s}: {len(ii)} neurons")
        return int(ii[0])

    gf = {s: one("DNp01", s) for s in ("left", "right")}
    ttmn = {s: one("TTMn", s) for s in ("left", "right")}
    psi = {s: one("PSI", s) for s in ("left", "right")}
    dlm = {s: np.flatnonzero(np.char.startswith(ct.astype(str), "DLM") & (side == s)) for s in ("left", "right")}
    watch = {**{f"GF|{s}": i for s, i in gf.items()}, **{f"TTMn|{s}": i for s, i in ttmn.items()},
             **{f"PSI|{s}": i for s, i in psi.items()}}
    glob = np.asarray(cx.meta["global_index"])
    g2name = {int(glob[i]): k for k, i in watch.items()}
    buf = []

    def sink(t, gidx):
        m = np.isin(gidx, list(g2name))
        buf.extend(zip(t[m].tolist(), gidx[m].tolist()))

    mv = M.model_variant(a.arm, cx)
    groups = {"GF_L": [gf["left"]], "GF_R": [gf["right"]]}
    eng = Brian2LIFEngine(cx, input_groups=groups, seed=0, model_variant=mv, spike_sink=sink)
    sd = eng.state_dict()
    res = dict(stage="prereg12_t4", arm=a.arm, model_variant=sd["model_variant"],
               gap_pairs=[dict(pre=str(ct[i]), pre_side=str(side[i]), post=str(ct[j]), post_side=str(side[j]))
                          for i, j in (mv.gap_pairs if mv is not None else ())],
               chemical_synapses={f"GF|{s}->{t}|{u}": int(cx.edges["n_syn"].to_numpy()[
                   (cx.edge_endpoints()[0] == gf[s]) & (cx.edge_endpoints()[1] == tgt[u])].sum())
                   for s in ("left", "right") for t, tgt in (("TTMn", ttmn), ("PSI", psi)) for u in ("left", "right")},
               n_neurons=cx.n_neurons, seeds=list(M.T4_SEEDS), rate_hz=M.T4_RATE_HZ, n_on=M.T4_N_ON, n_cut=M.T4_N_CUT,
               step_ms=M.T4_STEP_MS, weights_sha256=sd["weights_sha256"], model_sha256=eng.model_sha256, runs=[])
    nI = M.T4_N_ON + M.T4_N_CUT
    t_on = M.T4_N_ON * M.T4_STEP_MS * 1e-3
    for s in M.T4_SEEDS:
        eng.reset(s)
        buf.clear()
        st = {}

        def on_boundary(j, e):
            if j == M.T4_N_ON:
                st["on"] = e.spike_counts()
            if j < nI:
                r = M.T4_RATE_HZ if j < M.T4_N_ON else 0.0
                e.set_inputs({"GF_L": r, "GF_R": r})

        eng.run_coupled(nI, M.T4_STEP_MS, on_boundary)
        on = st["on"]
        spk = {k: np.array(sorted(t for t, g in buf if g2name[g] == k)) for k in watch}
        rec = dict(seed=s, network_on_hz=float(on.sum() / (cx.n_neurons * t_on)), per_side={})
        for sd_ in ("left", "right"):
            g = spk[f"GF|{sd_}"]
            tt = spk[f"TTMn|{sd_}"]
            g_on, tt_on = g[g < t_on], tt[tt < t_on]
            lat = []
            for x in tt_on:
                prev = g[g <= x]
                if len(prev) and x - prev[-1] < LAT_WINDOW_S:
                    lat.append(float(x - prev[-1]) * 1e3)
            rec["per_side"][sd_] = dict(
                gf_spikes=int(len(g_on)), ttmn_spikes=int(len(tt_on)),
                following=(float(len(tt_on) / len(g_on)) if len(g_on) else None),
                latency_ms=lat, latency_median_ms=(float(np.median(lat)) if lat else None),
                n_ttmn_after_gf=len(lat), psi_hz=float((spk[f"PSI|{sd_}"] < t_on).sum() / t_on),
                ttmn_hz=float(len(tt_on) / t_on), gf_hz=float(len(g_on) / t_on),
                dlm_hz=float(on[dlm[sd_]].sum() / (len(dlm[sd_]) * t_on)) if len(dlm[sd_]) else None,
                ttmn_cut_spikes=int((tt >= t_on).sum()))
        res["runs"].append(rec)
        print(a.arm, s, {k: (v["following"], v["latency_median_ms"], round(v["psi_hz"], 1)) for k, v in rec["per_side"].items()},
              f"({time.time() - t0:.0f} s)", flush=True)
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print("done", round(res["total_wall_s"]), "s ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
