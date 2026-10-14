"""R1, R2, R3 and gates 1, 2, 3' of prereg/08_f3a3_vnc_subgraph.md on the BANC v3 VNC subgraph.

    env -u PYTHONPATH python scripts/f3a3_vnc_run.py --out runs/f3a3/f3a3_v3.json

Subgraph: `scripts/f3a3_subgraph.py:vnc_subgraph` of `banc888_full_v3` (VNC/neck region + motor + DN input nodes +
ascending outputs; no edges onto DNs). Reference arm unchanged (w_syn 0.275 mV, scale 1.0); 25 ms steps; seeds 0-9;
gate 2 seeds 200-249.
R1: F3a DNg02 drives G20/G50/G100/GL100/GR100. R2: DNg02 100 Hz 1 s, 1 s cut. R3: HAND baseline DNg02 100 Hz plus
unilateral DNa04 / DNa10 / DNa15 at 10, 25, 50, 100 Hz (25 conditions).
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
from connectome import qa_report as Q  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb  # noqa: E402
from f3a2_vnc_run import groups_of, run  # noqa: E402
from f3a3_subgraph import vnc_subgraph  # noqa: E402
import f3a_vnc_run as F  # noqa: E402

STEP_MS, N_ON, N_CUT_R2, N_CUT = 25.0, 40, 40, 8
RUN_SEEDS, GATE1_SEEDS, GATE2_SEEDS = tuple(range(10)), tuple(range(10)), tuple(range(200, 250))
STEER_FAMILIES = ("DNa04", "DNa10", "DNa15")
STEER_RATES = (10, 25, 50, 100)
STEER_MN = ("b1 b2 b3 i1 i2 iii1 iii3 iii4 hg1 hg2 hg3 hg4 tp1 tp2 tpn ps1 ps2").split()
BASE = {"DNg02_L": 100, "DNg02_R": 100}
R1 = {"G20": {"DNg02_L": 20, "DNg02_R": 20}, "G50": {"DNg02_L": 50, "DNg02_R": 50}, "G100": dict(BASE),
      "GL100": {"DNg02_L": 100}, "GR100": {"DNg02_R": 100}}
R3 = {"B": dict(BASE)}
for _f in STEER_FAMILIES:
    for _s in ("L", "R"):
        for _r in STEER_RATES:
            R3[f"{_f}_{_s}_{_r}"] = dict(BASE, **{f"{_f}_{_s}": _r})
GATE2_GROUPS = ("DNg02",) + STEER_FAMILIES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="R", choices=("R", "A", "AC", "ACG"),
                    help="neuron model arm (prereg/12; R = the unmodified reference of prereg/08)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    whole = F.load()
    sym_whole = Q.type_symmetry(whole)
    cx, sg = vnc_subgraph(whole)
    glob = np.asarray(cx.meta["global_index"])
    del whole
    n = cx.neurons
    groups = {k: v for k, v in groups_of(n).items() if k.rsplit("_", 1)[0] in GATE2_GROUPS}
    if any(len(v) == 0 for v in groups.values()):
        raise SystemExit(f"empty DN group: {[k for k, v in groups.items() if len(v) == 0]}")
    types = F.readout_types(n)
    sym = Q.type_symmetry(cx)

    def symd(s):
        return {r["type"]: dict(n_left=int(r["n_left"]), n_right=int(r["n_right"]), log2_total=float(r["log2_total"]),
                                log2_exc=float(r["log2_exc"]), passed=bool(r["pass"])) for _, r in s.iterrows() if r["type"] in types}
    res = dict(stage="prereg08", subgraph=sg, n_neurons=cx.n_neurons, pairs=int(len(cx.edges)),
               synapses=int(cx.edges["n_syn"].sum()),
               graph_meta={k: v for k, v in cx.meta.items() if k not in ("sign_rules", "global_index")},
               dn_groups={k: dict(n=len(v), cell_types=sorted(set(n["cell_type"].astype(str).iloc[v]))) for k, v in groups.items()},
               baseline=BASE, baseline_label="HAND", r1_drives=R1, r3_conditions=R3, steer_mn=STEER_MN,
               steer_families=STEER_FAMILIES, steer_rates=STEER_RATES,
               seeds=dict(run=list(RUN_SEEDS), gate1=list(GATE1_SEEDS), gate2=[GATE2_SEEDS[0], GATE2_SEEDS[-1]]),
               step_ms=STEP_MS, n_on=N_ON, n_cut_r2=N_CUT_R2, n_cut=N_CUT,
               types={t: dict(category=c, n_left=int((m & (n["side"] == "left")).sum()),
                              n_right=int((m & (n["side"] == "right")).sum())) for t, (c, m) in types.items()},
               symmetry=symd(sym), symmetry_whole=symd(sym_whole))
    idx_ts = {}
    for t, (_, msk) in types.items():
        for s in ("left", "right"):
            ii = np.flatnonzero((msk & (n["side"] == s)).to_numpy())
            if len(ii):
                idx_ts[f"{t}|{s}"] = ii
    from mv2_common import model_variant
    mv = model_variant(a.arm, cx)
    eng = Brian2LIFEngine(cx, input_groups=groups, seed=0, model_variant=mv)
    eng.set_weight_scale(1.0)
    if a.arm != "R":
        res["model_arm"] = a.arm
        res["model_variant"] = mv.record()
        res["gap_pairs"] = [dict(pre=str(n["cell_type"].iloc[i]), pre_side=str(n["side"].iloc[i]),
                                 post=str(n["cell_type"].iloc[j]), post_side=str(n["side"].iloc[j])) for i, j in mv.gap_pairs]
    res["model_sha256"] = eng.model_sha256
    res["weights_sha256"] = eng.state_dict()["weights_sha256"]
    dur = N_ON * STEP_MS * 1e-3

    def rates(c):
        return {k: float(c[ii].sum() / (len(ii) * dur)) for k, ii in idx_ts.items()}

    def nondriven(drive):
        m = np.ones(cx.n_neurons, bool)
        for g in drive:
            m[groups[g]] = False
        return m

    def record(name, drive, s, on, net):
        nd = nondriven(drive)
        return dict(condition=name, seed=s, rate_hz=rates(on), network_rate_hz=float(on.sum() / (cx.n_neurons * dur)),
                    nondriven_rate_hz=float(on[nd].sum() / (nd.sum() * dur)),
                    group_rate_hz={g: float(on[groups[g]].sum() / (len(groups[g]) * dur)) for g in drive},
                    cut_last_step_hz=net[-1])

    def save():
        out.with_suffix(".partial.json").write_text(json.dumps(res, indent=1))

    # gate 1
    res["gate1"] = []
    for s in GATE1_SEEDS:
        _, end, _, _ = run(eng, {}, s, N_CUT, groups)
        res["gate1"].append(dict(seed=s, spikes=int(end.sum())))
    print("gate1 spikes", sum(r["spikes"] for r in res["gate1"]), f"({time.time() - t0:.0f} s)", flush=True)
    # R1
    res["r1"] = []
    for name, drive in R1.items():
        for s in RUN_SEEDS:
            on, _, net, _ = run(eng, drive, s, N_CUT, groups)
            res["r1"].append(record(name, drive, s, on, net))
        print("R1", name, "net %.3f Hz" % np.mean([r["network_rate_hz"] for r in res["r1"] if r["condition"] == name]),
              f"({time.time() - t0:.0f} s)", flush=True)
    save()
    # R2
    nd = nondriven(BASE)
    res["r2"] = dict(drive=BASE, runs=[])
    carriers = {}
    for s in RUN_SEEDS:
        on, end, net, per = run(eng, BASE, s, N_CUT_R2, groups, dict(idx_ts, _nondriven=np.flatnonzero(nd)))
        cut = end - on
        nz = np.flatnonzero(cut)
        for i in nz:
            c = carriers.setdefault(int(i), [0, 0])
            c[0] += 1
            c[1] += int(cut[i])
        last = [k for k, v in enumerate(net) if v > 0]
        nds = per.pop("_nondriven")
        res["r2"]["runs"].append(dict(seed=s, network_step_hz=net, nondriven_step_hz=nds, mn_step_hz=per,
                                      rate_hz=rates(on), network_rate_hz=float(on.sum() / (cx.n_neurons * dur)),
                                      cut_spikes=int(cut.sum()), n_cut_neurons=int(len(nz)),
                                      last_active_step=(last[-1] if last else None)))
        print("R2 seed", s, "cut spikes", int(cut.sum()), "neurons", len(nz), f"({time.time() - t0:.0f} s)", flush=True)
    cols = ("super_class", "cell_class", "sub_class", "cell_type", "side", "region")
    res["r2"]["carriers"] = [dict(index=i, whole_index=int(glob[i]), root_id=str(n["id"].iloc[i]),
                                  **{c: (str(n[c].iloc[i]) if c in n else None) for c in cols},
                                  n_seeds=v[0], cut_spikes=v[1]) for i, v in sorted(carriers.items(), key=lambda x: -x[1][1])]
    save()
    # R3
    res["r3"] = []
    for name, drive in R3.items():
        for s in RUN_SEEDS:
            on, _, net, _ = run(eng, drive, s, N_CUT, groups)
            res["r3"].append(record(name, drive, s, on, net))
        print("R3", name, "net %.3f Hz" % np.mean([r["network_rate_hz"] for r in res["r3"] if r["condition"] == name]),
              f"({time.time() - t0:.0f} s)", flush=True)
        save()
    # gate 2
    all_on = {g: 100 for g in groups}
    res["gate2_runs"] = []
    for s in GATE2_SEEDS:
        on, _, net, _ = run(eng, all_on, s, N_CUT, groups)
        res["gate2_runs"].append(dict(seed=s, last_cut_step_hz=net[-1], ignited=bool(net[-1] > 1.0),
                                      network_rate_hz=float(on.sum() / (cx.n_neurons * dur))))
    print("gate2 ignited", sum(r["ignited"] for r in res["gate2_runs"]), "/", len(GATE2_SEEDS), flush=True)
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
