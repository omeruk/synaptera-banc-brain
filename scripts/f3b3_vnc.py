"""VNC stage of prereg/11_f3b3_specificity.md: gates and the replay of the fresh FlyWire DN records (seeds 10-29)
into the BANC v3 VNC subgraph through coupling/dn_bridge.py, for `ensemble`, `bridge_off` and the 39 random
derangements `random_k` (sections 3-5, 10).

    env -u PYTHONPATH python scripts/f3b3_vnc.py --brain runs/f3b3/brain --out runs/f3b3/f3b3_v3.json

Subgraph of prereg/08 (`f3a3_subgraph.vnc_subgraph`), reference arm (w_syn 0.275 mV, scale 1.0). Input groups DNg02_L,
DNg02_R (HAND 100 Hz in every replay) and `steer` (the 305 BANC neurons of the 112 ensemble families, one rate per
neuron, set at every 5 ms boundary). Every arm replays the same brain records with the same VNC seeds. Records of the
random arms keep the readout rates and the spike accounting only (no bins, no per-family DN rates).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import qa_report as Q  # noqa: E402
from coupling.dn_bridge import DNBridge, family  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb  # noqa: E402
from f3a2_vnc_run import groups_of, run  # noqa: E402
from f3a3_subgraph import vnc_subgraph  # noqa: E402
import f3a_vnc_run as F  # noqa: E402
import f3b3_common as C  # noqa: E402
from f3b2_vnc import FW_N_NEURONS, brain_summary  # noqa: E402

GATE_N_CUT = 8
BIN = 5                                   # intervals per 25 ms bin
FULL_ARMS = ("ensemble", "bridge_off")    # arms with full records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=str(ROOT / "runs" / "f3b3" / "brain"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    bdir = Path(a.brain)
    t0 = time.time()
    perms = C.permutations()
    if C.permutations_sha256() != C.PERMUTATIONS_SHA256:
        raise SystemExit("permutations differ from prereg/11")
    brain, brain_sha = {}, {}
    for name in C.STIMULI:
        for s in C.SEEDS:
            p = bdir / f"{name}_s{s}.npz"
            if not p.exists():
                raise SystemExit(f"missing brain record {p}")
            brain[(name, s)] = dict(np.load(p, allow_pickle=True))          # own records (str object arrays)
            brain_sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    z0 = brain[("rot_left", C.SEEDS[0])]
    fw_tab = pd.DataFrame(dict(cell_type=z0["dn_cell_type"], side=z0["dn_side"]))

    whole = F.load()
    cx, sg = vnc_subgraph(whole)
    del whole
    n = cx.neurons
    g_all = groups_of(n)
    dn = (n["super_class"] == "descending").to_numpy()
    banc_tab = pd.DataFrame(dict(cell_type=n["cell_type"].astype(str).to_numpy()[dn],
                                 side=n["side"].astype(str).to_numpy()[dn], index=np.flatnonzero(dn)))
    banc_tab["family"] = banc_tab["cell_type"].map(family)
    ens = banc_tab[banc_tab["family"].isin(C.ENSEMBLE)]
    steer_idx = np.sort(ens["index"].to_numpy(np.int64))
    dng02 = np.concatenate([g_all["DNg02_L"], g_all["DNg02_R"]])
    if np.intersect1d(steer_idx, dng02).size:
        raise SystemExit("ensemble overlaps the DNg02 baseline")
    groups = {"DNg02_L": g_all["DNg02_L"], "DNg02_R": g_all["DNg02_R"], "steer": steer_idx}
    pos_in_steer = {int(i): k for k, i in enumerate(steer_idx)}
    bridges = {arm: (DNBridge(fw_tab, banc_tab[["cell_type", "side", "index"]], C.bridge_config(arm, perms))
                     if arm != "bridge_off" else None) for arm in C.ARMS}
    for arm, b in bridges.items():
        if b is not None and sorted(b.target_index.tolist()) != steer_idx.tolist():
            raise SystemExit(f"{arm}: bridge targets differ from the BANC ensemble neurons")
    types = F.readout_types(n)
    sym = Q.type_symmetry(cx)
    idx_ts = {}
    for t, (_, msk) in types.items():
        for s in ("left", "right"):
            ii = np.flatnonzero((msk & (n["side"] == s)).to_numpy())
            if len(ii):
                idx_ts[f"{t}|{s}"] = ii
    fam_side = {f"{f}|{s}": np.sort(ens[(ens["family"] == f) & (ens["side"] == s)]["index"].to_numpy(np.int64))
                for f in C.ENSEMBLE for s in ("left", "right")}
    fam_side = {k: v for k, v in fam_side.items() if len(v)}
    nondriven = np.ones(cx.n_neurons, bool)
    for v in groups.values():
        nondriven[v] = False

    res = dict(stage="prereg11", subgraph=dict(n_neurons=cx.n_neurons, pairs=int(len(cx.edges)),
                                               synapses=int(cx.edges["n_syn"].sum()), counts=sg),
               groups={k: dict(n=len(v), cell_types=sorted(set(n["cell_type"].astype(str).iloc[v]))) for k, v in groups.items()},
               ensemble_banc=dict(n=int(len(steer_idx)), by_side={k: int(v) for k, v in ens["side"].value_counts().items()},
                                  family_side_n={k: int(len(v)) for k, v in fam_side.items()}),
               bridges={arm: (b.record() if b is not None else None) for arm, b in bridges.items()},
               constants=dict(ensemble=list(C.ENSEMBLE), window_intervals=C.BRIDGE_WINDOW_INTERVALS,
                              latency_intervals=C.BRIDGE_LATENCY_INTERVALS, permutations=perms,
                              perm_seeds=dict(zip(C.RANDOM_ARMS, C.PERM_SEEDS)),
                              permutations_sha256=C.permutations_sha256(), interval_ms=C.INTERVAL_MS,
                              n_grey=C.N_GREY, n_stim=C.N_STIM, window=list(C.WINDOW), baseline_hz=C.BASELINE_HZ,
                              baseline_label="HAND", primary_mn=C.PRIMARY_MN, steer_mn=C.STEER_MN, arms=C.ARMS,
                              stimuli=C.STIMULI, seeds=list(C.SEEDS), gate1_seeds=list(C.GATE1_SEEDS),
                              gate2_seeds=list(C.GATE2_SEEDS), perm_alpha=C.PERM_ALPHA),
               brain_records=dict(dir=str(bdir.resolve().relative_to(ROOT)) if bdir.resolve().is_relative_to(ROOT)
                                  else str(bdir), sha256=brain_sha, n_dn=int(z0["dn_counts"].shape[1])),
               symmetry={r["type"]: dict(n_left=int(r["n_left"]), n_right=int(r["n_right"]),
                                         log2_total=float(r["log2_total"]), log2_exc=float(r["log2_exc"]),
                                         passed=bool(r["pass"])) for _, r in sym.iterrows() if r["type"] in types},
               brain=[dict(stimulus=k[0], seed=k[1], rate_hz=brain_summary(z),
                           net_rate_hz=float(z["net_counts"][C.WINDOW[0]:C.WINDOW[1]].sum() / FW_N_NEURONS / 1.0),
                           weights_sha256=str(z["weights_sha256"]), model_sha256=str(z["model_sha256"]))
                      for k, z in brain.items()])
    eng = Brian2LIFEngine(cx, input_groups=groups, seed=0)
    eng.set_weight_scale(1.0)
    res["model_sha256"] = eng.model_sha256
    res["weights_sha256"] = eng.state_dict()["weights_sha256"]
    print(f"VNC engine {cx.n_neurons:,} neurons, ensemble {len(steer_idx)} BANC DNs, {len(C.ARMS)} arms "
          f"({time.time() - t0:.0f} s)", flush=True)

    def save():
        out.with_suffix(".partial.json").write_text(json.dumps(res))

    # gate 1
    res["gate1"] = []
    for s in C.GATE1_SEEDS:
        _, end, _, _ = run(eng, {}, s, GATE_N_CUT, groups)
        res["gate1"].append(dict(seed=s, spikes=int(end.sum())))
    print("gate1 spikes", sum(r["spikes"] for r in res["gate1"]), flush=True)

    # replays
    nI = C.N_GREY + C.N_STIM
    lo, hi = C.WINDOW
    dur = (hi - lo) * C.INTERVAL_MS * 1e-3
    bin_s = BIN * C.INTERVAL_MS * 1e-3
    prim = {f"{m}|{s}": idx_ts[f"{m}|{s}"] for m in C.PRIMARY_MN for s in ("left", "right")}
    res["replay"] = []
    for arm in C.ARMS:
        b = bridges[arm]
        full = arm in FULL_ARMS
        for name in C.STIMULI:
            for s in C.SEEDS:
                z = brain[(name, s)]
                steer = np.zeros((nI, len(steer_idx)))
                transfer = None
                if b is not None:
                    r = b.rates(z["dn_counts"])
                    cols = [pos_in_steer[int(i)] for i in b.target_index]
                    steer[:, cols] = r
                    transfer = b.transfer(z["dn_counts"])
                applied = steer[lo:hi].mean(0)
                eng.reset(C.vnc_seed(name, s))
                st = {"prev": eng.spike_counts(), "lo": None, "hi": None}
                net_bins, prim_bins = [], {k: [] for k in prim}

                def on_boundary(j, e):
                    if full and j > 0 and j % BIN == 0:
                        cur = e.spike_counts()
                        d = cur - st["prev"]
                        st["prev"] = cur
                        net_bins.append(float(d.sum() / (cx.n_neurons * bin_s)))
                        for k, ii in prim.items():
                            prim_bins[k].append(float(d[ii].sum() / (len(ii) * bin_s)))
                    if j == lo:
                        st["lo"] = e.spike_counts()
                    if j == hi:
                        st["hi"] = e.spike_counts()
                    if j < nI:
                        e.set_inputs({"DNg02_L": C.BASELINE_HZ, "DNg02_R": C.BASELINE_HZ, "steer": steer[j]})

                eng.run_coupled(nI, C.INTERVAL_MS, on_boundary)
                on = st["hi"] - st["lo"]
                rec = dict(arm=arm, stimulus=name, seed=s, vnc_seed=C.vnc_seed(name, s),
                           rate_hz={k: float(on[ii].sum() / (len(ii) * dur)) for k, ii in idx_ts.items()},
                           network_rate_hz=float(on.sum() / (cx.n_neurons * dur)),
                           nondriven_rate_hz=float(on[nondriven].sum() / (nondriven.sum() * dur)),
                           steer_applied_mean_hz=float(applied.mean()),
                           steer_realised_hz=float(on[steer_idx].sum() / (len(steer_idx) * dur)), transfer=transfer)
                if full:
                    rec.update(
                        dn_realised_hz={**{k: float(on[ii].sum() / (len(ii) * dur)) for k, ii in fam_side.items()},
                                        "DNg02|left": float(on[groups["DNg02_L"]].sum() / (len(groups["DNg02_L"]) * dur)),
                                        "DNg02|right": float(on[groups["DNg02_R"]].sum() / (len(groups["DNg02_R"]) * dur))},
                        dn_applied_hz={k: float(np.mean([applied[pos_in_steer[int(i)]] for i in ii]))
                                       for k, ii in fam_side.items()},
                        net_bin25_hz=net_bins, primary_bin25_hz=prim_bins)
                res["replay"].append(rec)
            print(f"{arm:11s} {name:9s} net {np.mean([r['network_rate_hz'] for r in res['replay'] if r['arm'] == arm and r['stimulus'] == name]):.3f} Hz "
                  f"({time.time() - t0:.0f} s)", flush=True)
        save()

    # gate 2: every bridge input group (DNg02 both sides and all ensemble neurons) at 100 Hz; the same for every arm
    all_on = {g: 100.0 for g in groups}
    res["gate2_runs"] = []
    for s in C.GATE2_SEEDS:
        on, _, net, _ = run(eng, all_on, s, GATE_N_CUT, groups)
        res["gate2_runs"].append(dict(seed=s, last_cut_step_hz=net[-1], ignited=bool(net[-1] > 1.0),
                                      network_rate_hz=float(on.sum() / (cx.n_neurons * 1.0))))
    print("gate2 ignited", sum(r["ignited"] for r in res["gate2_runs"]), "/", len(C.GATE2_SEEDS), flush=True)
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.write_text(json.dumps(res))
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
