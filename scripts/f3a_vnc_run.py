"""F3a of prereg/06_banc_vnc_dn_to_mn.md: Poisson drive of DN types on the BANC v3 graph, readout at motor neurons.

    env -u PYTHONPATH python scripts/f3a_vnc_run.py --out runs/f3a/f3a_v3.json

Protocol (prereg section 3): 40 steps of 25 ms drive, then 8 steps cut; seeds 0-9. Gate 1 seeds 0-4 (D0), gate 2 seeds
200-249 (all DN groups at 100 Hz), reference arm unchanged (w_syn 0.275 mV, scale 1.0).
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
from connectome.adapters import banc  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb  # noqa: E402

B = ROOT / "banc"
STEP_MS, N_ON, N_CUT = 25.0, 40, 8
RUN_SEEDS, GATE1_SEEDS, GATE2_SEEDS = tuple(range(10)), tuple(range(5)), tuple(range(200, 250))
WING = ("DLM1-4 DLM5 DVM1a-c DVM2a-b DVM3a-b b1 b2 b3 i1 i2 iii1 iii3 iii4 hg1 hg2 hg3 hg4 tp1 tp2 tpn ps1 ps2 PSI").split()
OTHER_WING = ("MNxm01", "PSn_u")
DN_GROUPS = ("DNg02_L", "DNg02_R", "DNa02_L", "DNa02_R", "GF_L", "GF_R")
DRIVES = {
    "D0": {}, "G20": {"DNg02_L": 20, "DNg02_R": 20}, "G50": {"DNg02_L": 50, "DNg02_R": 50},
    "G100": {"DNg02_L": 100, "DNg02_R": 100}, "GL100": {"DNg02_L": 100}, "GR100": {"DNg02_R": 100},
    "AL100": {"DNa02_L": 100}, "AR100": {"DNa02_R": 100}, "F100": {"GF_L": 100, "GF_R": 100},
}


def load():
    return banc.load_connectome_full(B / "neurons-banc.csv.gz", B / "banc_888_edgelist_simple_v3.feather",
                                     B / "banc_888_meta.feather", B / "connections_princeton-banc.csv.gz",
                                     neuron_set="codex", hash_files=True, data_source="banc888_full_v3")


def dn_groups(n):
    ct, sd, sc = n["cell_type"].astype(str), n["side"], n["super_class"]
    dn = sc == "descending"
    sel = {"DNg02": dn & ct.str.startswith("DNg02"), "DNa02": dn & (ct == "DNa02"), "GF": dn & (ct == "DNp01")}
    g = {}
    for k, m in sel.items():
        for s in ("left", "right"):
            g[f"{k}_{s[0].upper()}"] = np.flatnonzero((m & (sd == s)).to_numpy())
    return g


def readout_types(n):
    """{label: (category, boolean mask)} per cell type, by registered name (prereg section 2)."""
    ct, mo = n["cell_type"].astype(str), n["super_class"] == "motor"
    out = {}
    for t in WING:
        out[t] = ("wing_steering" if (n["sub_class"] == "wing_steering_motor_neuron")[mo & (ct == t)].any() else "wing",
                  mo & (ct == t))
    for t in OTHER_WING:
        out[t] = ("other_wing", mo & (ct == t))
    for t in sorted(ct[mo & (n["cell_class"] == "haltere_motor_neuron")].unique()):
        out[t] = ("haltere", mo & (ct == t))
    for t in sorted(ct[mo & (n["cell_class"] == "neck_motor_neuron")].unique()):
        out[t] = ("neck", mo & (ct == t))
    out["TTMn"] = ("TTMn", mo & (ct == "TTMn"))
    return out


def one_run(eng, drive, seed, n_cut=N_CUT):
    eng.reset(seed)
    eng.set_inputs({g: float(drive.get(g, 0.0)) for g in DN_GROUPS})
    n = eng.n_neurons
    steps = []
    on = None
    for k in range(N_ON + n_cut):
        if k == N_ON:
            eng.set_inputs({g: 0.0 for g in DN_GROUPS})
            on = eng.spike_counts()
        b = eng.spike_counts().sum()
        eng.advance(STEP_MS)
        steps.append(float((eng.spike_counts().sum() - b) / (n * STEP_MS * 1e-3)))
    if on is None:
        on = eng.spike_counts()
    return on.astype(np.int64), steps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    cx = load()
    n = cx.neurons
    groups = dn_groups(n)
    types = readout_types(n)
    res = dict(stage="prereg06", n_neurons=cx.n_neurons, pairs=int(len(cx.edges)), synapses=int(cx.edges["n_syn"].sum()),
               graph_meta={k: v for k, v in cx.meta.items() if k != "sign_rules"},
               dn_groups={k: len(v) for k, v in groups.items()}, drives=DRIVES,
               seeds=dict(run=list(RUN_SEEDS), gate1=list(GATE1_SEEDS), gate2=[GATE2_SEEDS[0], GATE2_SEEDS[-1]]),
               types={t: dict(category=c, n=int(m.sum()),
                              n_left=int((m & (n["side"] == "left")).sum()), n_right=int((m & (n["side"] == "right")).sum()))
                      for t, (c, m) in types.items()})
    if any(res["types"][t]["n"] == 0 for t in WING):
        raise SystemExit("a registered wing type is missing in the Codex table")
    sym = Q.type_symmetry(cx)
    res["symmetry"] = {r["type"]: dict(n_left=int(r["n_left"]), n_right=int(r["n_right"]), in_left=int(r["in_left"]),
                                       in_right=int(r["in_right"]), log2_total=float(r["log2_total"]),
                                       log2_exc=float(r["log2_exc"]), passed=bool(r["pass"]))
                       for _, r in sym.iterrows() if r["type"] in types}
    # data only: direct synapses DN -> readout types (not a criterion)
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy()
    gidx = {k: np.concatenate([groups[f"{k}_L"], groups[f"{k}_R"]]) for k in ("DNg02", "DNa02", "GF")}
    res["direct_synapses"] = {}
    for k, idx in gidx.items():
        m_pre = np.isin(pre, idx)
        res["direct_synapses"][k] = {t: int(w[m_pre & np.isin(post, np.flatnonzero(msk.to_numpy()))].sum())
                                     for t, (_, msk) in types.items()}
    side = n["side"].to_numpy()
    idx_ts = {}
    for t, (_, msk) in types.items():
        for s in ("left", "right"):
            ii = np.flatnonzero((msk & (n["side"] == s)).to_numpy())
            if len(ii):
                idx_ts[f"{t}|{s}"] = ii
    driven_all = np.concatenate(list(groups.values()))

    eng = Brian2LIFEngine(cx, input_groups=groups, seed=0)
    eng.set_weight_scale(1.0)
    res["model_sha256"] = eng.model_sha256
    res["weights_sha256"] = eng.state_dict()["weights_sha256"]
    res["runs"] = []
    dur = N_ON * STEP_MS * 1e-3
    nd = np.ones(cx.n_neurons, bool)
    for name, drive in DRIVES.items():
        driven = np.concatenate([groups[g] for g in drive]) if drive else np.array([], int)
        nd_c = np.ones(cx.n_neurons, bool)
        nd_c[driven] = False
        for s in RUN_SEEDS:
            c, steps = one_run(eng, drive, s)
            res["runs"].append(dict(
                drive=name, seed=s, total_spikes=int(c.sum()),
                rate_hz={k: float(c[ii].sum() / (len(ii) * dur)) for k, ii in idx_ts.items()},
                network_rate_hz=float(c.sum() / (cx.n_neurons * dur)),
                nondriven_rate_hz=float(c[nd_c].sum() / (nd_c.sum() * dur)),
                driven_dn_rate_hz=(float(c[driven].sum() / (len(driven) * dur)) if len(driven) else None),
                cut_mean_hz=float(np.mean(steps[N_ON:])), cut_last_step_hz=float(steps[-1])))
        print(name, "net %.3f Hz" % np.mean([r["network_rate_hz"] for r in res["runs"] if r["drive"] == name]),
              f"({time.time() - t0:.0f} s)", flush=True)
        out.with_suffix(".partial.json").write_text(json.dumps(res, indent=1))
    all_on = {g: 100 for g in DN_GROUPS}
    res["gate2_runs"] = []
    for s in GATE2_SEEDS:
        c, steps = one_run(eng, all_on, s)
        res["gate2_runs"].append(dict(seed=s, last_cut_step_hz=steps[-1], ignited=bool(steps[-1] > 1.0),
                                      network_rate_hz=float(c.sum() / (cx.n_neurons * dur))))
    print("gate2 ignited", sum(r["ignited"] for r in res["gate2_runs"]), "/", len(GATE2_SEEDS), flush=True)
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
