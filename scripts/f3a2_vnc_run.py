"""C2 and C3 of prereg/07_f3a2_rate_persistence_steering.md on the BANC v3 graph (plus gates 1, 2, 3').

    env -u PYTHONPATH python scripts/f3a2_vnc_run.py --out runs/f3a2/f3a2_v3.json

C2: DNg02 both sides 100 Hz for 1 s, then a 1 s cut; per-step network and MN-type rates, carriers of the cut activity.
C3: baseline DNg02 both sides 100 Hz (label HAND) plus unilateral 100 Hz drive of one steering family; 13 conditions.
Reference arm unchanged (w_syn 0.275 mV, scale 1.0); 25 ms steps; seeds 0-9; gate 1 seeds 0-4, gate 2 seeds 200-249.
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
from f3_dn_crosswalk import family  # noqa: E402
import f3a_vnc_run as F  # noqa: E402

STEP_MS, N_ON, N_CUT_C2, N_CUT = 25.0, 40, 40, 8
RUN_SEEDS, GATE1_SEEDS, GATE2_SEEDS = tuple(range(10)), tuple(range(5)), tuple(range(200, 250))
STEER_FAMILIES = ("DNa02", "DNp31", "DNa04", "DNa10", "DNa15", "DNa16")
STEER_MN = ("b1 b2 b3 i1 i2 iii1 iii3 iii4 hg1 hg2 hg3 hg4 tp1 tp2 tpn ps1 ps2").split()
BASE = {"DNg02_L": 100, "DNg02_R": 100}
C3 = {"B": dict(BASE)}
for _f in STEER_FAMILIES:
    for _s in ("L", "R"):
        C3[f"{_f}_{_s}"] = dict(BASE, **{f"{_f}_{_s}": 100})


def groups_of(n):
    ct, sd, dn = n["cell_type"].astype(str), n["side"], n["super_class"] == "descending"
    fam = ct.map(family)
    sel = {"DNg02": dn & ct.str.startswith("DNg02")}
    for f in STEER_FAMILIES:
        sel[f] = dn & (fam == f)
    g = {}
    for k, m in sel.items():
        for s in ("left", "right"):
            g[f"{k}_{s[0].upper()}"] = np.flatnonzero((m & (sd == s)).to_numpy())
    return g


def run(eng, drive, seed, n_cut, groups, step_idx=None):
    """Counts at the end of the drive and at the end of the cut; per-step network rate; per-step counts of step_idx."""
    eng.reset(seed)
    eng.set_inputs({g: float(drive.get(g, 0.0)) for g in groups})
    n = eng.n_neurons
    net, per = [], {k: [] for k in (step_idx or {})}
    prev = eng.spike_counts().copy()
    on = None
    for k in range(N_ON + n_cut):
        if k == N_ON:
            eng.set_inputs({g: 0.0 for g in groups})
            on = prev.copy()
        eng.advance(STEP_MS)
        cur = eng.spike_counts().copy()
        d = cur - prev
        net.append(float(d.sum() / (n * STEP_MS * 1e-3)))
        for key, ii in (step_idx or {}).items():
            per[key].append(float(d[ii].sum() / (len(ii) * STEP_MS * 1e-3)))
        prev = cur
    return on.astype(np.int64), prev.astype(np.int64), net, per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cx = F.load()
    n = cx.neurons
    groups = groups_of(n)
    types = F.readout_types(n)
    if any(len(v) == 0 for v in groups.values()):
        raise SystemExit(f"empty DN group: {[k for k, v in groups.items() if len(v) == 0]}")
    sym = Q.type_symmetry(cx)
    res = dict(stage="prereg07", n_neurons=cx.n_neurons, pairs=int(len(cx.edges)), synapses=int(cx.edges["n_syn"].sum()),
               graph_meta={k: v for k, v in cx.meta.items() if k != "sign_rules"},
               dn_groups={k: dict(n=len(v), cell_types=sorted(set(n["cell_type"].astype(str).iloc[v]))) for k, v in groups.items()},
               baseline=BASE, baseline_label="HAND", c3_conditions=C3, steer_mn=STEER_MN,
               seeds=dict(run=list(RUN_SEEDS), gate1=list(GATE1_SEEDS), gate2=[GATE2_SEEDS[0], GATE2_SEEDS[-1]]),
               step_ms=STEP_MS, n_on=N_ON, n_cut_c2=N_CUT_C2, n_cut=N_CUT,
               types={t: dict(category=c, n_left=int((m & (n["side"] == "left")).sum()),
                              n_right=int((m & (n["side"] == "right")).sum())) for t, (c, m) in types.items()},
               symmetry={r["type"]: dict(n_left=int(r["n_left"]), n_right=int(r["n_right"]), log2_total=float(r["log2_total"]),
                                         log2_exc=float(r["log2_exc"]), passed=bool(r["pass"]))
                         for _, r in sym.iterrows() if r["type"] in types})
    idx_ts = {}
    for t, (_, msk) in types.items():
        for s in ("left", "right"):
            ii = np.flatnonzero((msk & (n["side"] == s)).to_numpy())
            if len(ii):
                idx_ts[f"{t}|{s}"] = ii
    eng = Brian2LIFEngine(cx, input_groups=groups, seed=0)
    eng.set_weight_scale(1.0)
    res["model_sha256"] = eng.model_sha256
    res["weights_sha256"] = eng.state_dict()["weights_sha256"]
    dur = N_ON * STEP_MS * 1e-3

    def rates(c):
        return {k: float(c[ii].sum() / (len(ii) * dur)) for k, ii in idx_ts.items()}

    def save():
        out.with_suffix(".partial.json").write_text(json.dumps(res, indent=1))

    # gate 1
    res["gate1"] = []
    for s in GATE1_SEEDS:
        on, end, _, _ = run(eng, {}, s, N_CUT, groups)
        res["gate1"].append(dict(seed=s, spikes=int(end.sum())))
    print("gate1 spikes", sum(r["spikes"] for r in res["gate1"]), f"({time.time() - t0:.0f} s)", flush=True)
    # C2
    res["c2"] = dict(drive=BASE, runs=[])
    carriers = {}
    for s in RUN_SEEDS:
        on, end, net, per = run(eng, BASE, s, N_CUT_C2, groups, idx_ts)
        cut = end - on
        nz = np.flatnonzero(cut)
        for i in nz:
            c = carriers.setdefault(int(i), [0, 0])
            c[0] += 1
            c[1] += int(cut[i])
        last = [k for k, v in enumerate(net) if v > 0]
        res["c2"]["runs"].append(dict(seed=s, network_step_hz=net, mn_step_hz=per, rate_hz=rates(on),
                                      network_rate_hz=float(on.sum() / (cx.n_neurons * dur)),
                                      cut_spikes=int(cut.sum()), n_cut_neurons=int(len(nz)),
                                      last_active_step=(last[-1] if last else None)))
        print("C2 seed", s, "cut spikes", int(cut.sum()), "neurons", len(nz), f"({time.time() - t0:.0f} s)", flush=True)
    cols = ("super_class", "cell_class", "sub_class", "cell_type", "side", "region")
    res["c2"]["carriers"] = [dict(index=i, root_id=str(n["id"].iloc[i]),
                                  **{c: (str(n[c].iloc[i]) if c in n else None) for c in cols},
                                  n_seeds=v[0], cut_spikes=v[1]) for i, v in sorted(carriers.items(), key=lambda x: -x[1][1])]
    save()
    # C3
    res["c3"] = []
    for name, drive in C3.items():
        driven = np.concatenate([groups[g] for g in drive])
        nd = np.ones(cx.n_neurons, bool)
        nd[driven] = False
        for s in RUN_SEEDS:
            on, end, net, _ = run(eng, drive, s, N_CUT, groups)
            res["c3"].append(dict(condition=name, seed=s, rate_hz=rates(on),
                                  network_rate_hz=float(on.sum() / (cx.n_neurons * dur)),
                                  nondriven_rate_hz=float(on[nd].sum() / (nd.sum() * dur)),
                                  group_rate_hz={g: float(on[groups[g]].sum() / (len(groups[g]) * dur)) for g in drive},
                                  cut_last_step_hz=net[-1]))
        print("C3", name, "net %.3f Hz" % np.mean([r["network_rate_hz"] for r in res["c3"] if r["condition"] == name]),
              f"({time.time() - t0:.0f} s)", flush=True)
        save()
    # gate 2: every group at 100 Hz
    all_on = {g: 100 for g in groups}
    res["gate2_runs"] = []
    for s in GATE2_SEEDS:
        on, end, net, _ = run(eng, all_on, s, N_CUT, groups)
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
