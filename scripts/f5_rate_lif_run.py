"""F5 step 2a (prereg/15): the untrained rate VNC (engine/rate_torch.py) on the F3a-3 protocol, plus the prereg/14
gate 3 engine checks on a 2,000-neuron subgraph of the real VNC.

    env -u PYTHONPATH python scripts/f5_rate_lif_run.py --out runs/f5/rate_lif

Graph: the prereg/08 BANC v3 VNC subgraph (`scripts/f3a3_subgraph.py:vnc_subgraph`) with its prereg/01 signs; every DN
is a clamped input node. Conditions: the F3a-3 R1 drives and the 25 R3 conditions (`scripts/f3a3_vnc_run.py`), a
driven DN neuron clamped to rate / R_IN, 200 drive intervals + 40 cut intervals of 5 ms. The rate model is at
initialisation (gain 1, leak LAM_0), float32 and deterministic, so one run per condition.

Writes `record.npz` (per condition and interval: the mean state of every readout type and side, the network mean |s|)
and `record.json` (graph, engine, groups, symmetry, the engine checks, wall time, peak RAM). Refuses an existing --out.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import qa_report as Q  # noqa: E402
from engine import rate_torch as RT  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb  # noqa: E402
from f3a2_vnc_run import groups_of  # noqa: E402
from f3a3_subgraph import vnc_subgraph  # noqa: E402
import f3a3_vnc_run as F3  # noqa: E402
import f3a_vnc_run as F  # noqa: E402

N_DRIVE, N_CUT = 200, 40          # 5 ms intervals: 1 s drive (F3a-3), 200 ms cut (F3a-3 R1 / R3)
CHECK_N, CHECK_T_FWD, CHECK_T_GRAD, CHECK_BATCH = 2000, 300, 40, 2
CHECK_SEED = 15


def conditions():
    """The F3a-3 R1 drives and R3 conditions, in that order (R1 G100 and R3 B are the same drive; both are run)."""
    return [("R1", k, v) for k, v in F3.R1.items()] + [("R3", k, v) for k, v in F3.R3.items()]


def check_subgraph(g: RT.RateGraph, seeds) -> np.ndarray:
    """CHECK_N neurons: the seed neurons, then breadth-first downstream of them in index order until CHECK_N."""
    out = [[] for _ in range(g.n)]
    for a, b in zip(g.pre, g.post):
        out[a].append(b)
    keep, seen = [], np.zeros(g.n, bool)
    q = deque(int(s) for s in seeds)
    for s in seeds:
        seen[s] = True
    while q and len(keep) < CHECK_N:
        a = q.popleft()
        keep.append(a)
        for b in sorted(out[a]):
            if not seen[b]:
                seen[b] = True
                q.append(b)
    return np.sort(np.asarray(keep[:CHECK_N]))


def engine_checks(sub: RT.RateGraph) -> dict:
    """prereg/14 gate 3 forward and gradient checks on ``sub`` (gains and leaks drawn away from their initial values)."""
    rng = np.random.default_rng(CHECK_SEED)
    theta = rng.normal(size=sub.n_edges)
    phi = rng.normal(size=sub.n)
    res = dict(n_neurons=sub.n, n_edges=sub.n_edges, n_inputs=int(len(sub.inputs)), seed=CHECK_SEED)
    # forward, float32, 300 intervals
    x = rng.uniform(0, 1, (CHECK_T_FWD, CHECK_BATCH, len(sub.inputs))).astype(np.float32)
    net = RT.RateNet(sub)
    with torch.no_grad():
        net.theta.copy_(torch.from_numpy(theta))
        net.phi.copy_(torch.from_numpy(phi))
        st, _ = net.run(torch.from_numpy(x))
    ref = RT.reference_numpy(sub, net.gains().detach().numpy(), net.leaks().detach().numpy(), x)
    res["forward"] = dict(intervals=CHECK_T_FWD, batch=CHECK_BATCH, dtype="float32",
                          max_abs_diff=float(np.abs(st.numpy() - ref).max()), max_abs_state=float(np.abs(ref).max()),
                          threshold=1e-5)
    res["forward"]["passed"] = res["forward"]["max_abs_diff"] < 1e-5
    # gradient, float64
    x = torch.from_numpy(rng.uniform(0, 1, (CHECK_T_GRAD, CHECK_BATCH, len(sub.inputs))))
    coef = torch.from_numpy(rng.normal(size=(CHECK_T_GRAD, CHECK_BATCH, sub.n)))
    net = RT.RateNet(sub, dtype=torch.float64)
    with torch.no_grad():
        net.theta.copy_(torch.from_numpy(theta))
        net.phi.copy_(torch.from_numpy(phi))
    st, _ = net.run(x)
    (st * coef).sum().backward()
    th = torch.tensor(theta, requires_grad=True)
    ph = torch.tensor(phi, requires_grad=True)
    dense = RT.dense_reference_torch(sub, th, ph, x)
    (dense * coef).sum().backward()

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max())
    res["gradient"] = dict(intervals=CHECK_T_GRAD, batch=CHECK_BATCH, dtype="float64",
                           forward_max_abs_diff=float((st - dense).detach().abs().max()),
                           rel_err_theta=rel(net.theta.grad, th.grad), rel_err_phi=rel(net.phi.grad, ph.grad),
                           threshold=1e-6, definition="max |sparse - dense| / max |dense| over all parameters")
    res["gradient"]["passed"] = max(res["gradient"]["rel_err_theta"], res["gradient"]["rel_err_phi"]) < 1e-6
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    cx, sg = vnc_subgraph(F.load())
    n = cx.neurons
    dn = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    groups = {k: v for k, v in groups_of(n).items() if k.rsplit("_", 1)[0] in F3.GATE2_GROUPS}
    graph = RT.RateGraph.from_connectome(cx, inputs=dn)
    col = {int(i): c for c, i in enumerate(graph.inputs)}
    types = F.readout_types(n)
    keys, members = [], []
    for t, (_, msk) in types.items():
        for s in ("left", "right"):
            ii = np.flatnonzero((msk & (n["side"] == s)).to_numpy())
            if len(ii):
                keys.append(f"{t}|{s}")
                members.append(ii)
    sym = Q.type_symmetry(cx)
    symmetry = {r["type"]: dict(log2_total=float(r["log2_total"]), passed=bool(r["pass"]))
                for _, r in sym.iterrows() if r["type"] in types}
    conds = conditions()
    T = N_DRIVE + N_CUT
    x = np.zeros((T, len(conds), len(graph.inputs)), np.float32)
    for b, (_, _, drv) in enumerate(conds):
        for gname, rate in drv.items():
            for i in groups[gname]:
                x[:N_DRIVE, b, col[int(i)]] = rate / RT.R_IN
    net = RT.RateNet(graph)
    t1 = time.time()
    with torch.no_grad():
        st, _ = net.run(torch.from_numpy(x))
    run_wall = time.time() - t1
    st = st.numpy()                                       # (T, conditions, n)
    mn = np.stack([st[:, :, ii].mean(2) for ii in members], axis=2).transpose(1, 0, 2)   # (conditions, T, keys)
    net_abs = np.abs(st).mean(2).T                         # (conditions, T)
    nondriven = np.ones(graph.n, bool)
    nondriven[graph.inputs] = False
    net_abs_nondriven = np.abs(st[:, :, nondriven]).mean(2).T
    del st
    seeds = np.concatenate([groups[g] for g in sorted(groups)] + members)
    sub = graph.induced(check_subgraph(graph, seeds))
    checks = engine_checks(sub)
    np.savez_compressed(out / "record.npz", mn_state=mn.astype(np.float32), net_abs=net_abs.astype(np.float32),
                        net_abs_nondriven=net_abs_nondriven.astype(np.float32), keys=np.array(keys),
                        conditions=np.array([f"{p}:{k}" for p, k, _ in conds]))
    rec = dict(prereg="prereg/15_f5_rate_lif_comparison.md", arm="untrained", subgraph=sg["kept"],
               graph_meta={k: v for k, v in cx.meta.items() if k not in ("sign_rules", "global_index")},
               engine=net.record(), n_dn_inputs=int(len(dn)), n_intervals=dict(drive=N_DRIVE, cut=N_CUT),
               interval_ms=5, conditions=[dict(part=p, name=k, drive=v) for p, k, v in conds],
               dn_groups={k: dict(n=int(len(v)), cell_types=sorted(set(n["cell_type"].astype(str).iloc[v])))
                          for k, v in groups.items()},
               type_side_neurons={k: int(len(ii)) for k, ii in zip(keys, members)}, symmetry=symmetry,
               engine_checks=dict(check_subgraph_rule=f"seed neurons (8 DN groups + readout MNs), then breadth-first "
                                                      f"downstream until {CHECK_N}", **checks),
               threads=a.threads, run_wall_s=run_wall, total_wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb(),
               torch_version=torch.__version__)
    (out / "record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(run_wall_s=run_wall, checks=checks), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
