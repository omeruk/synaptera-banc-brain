"""F5 step 2a-2 (prereg/16): reachability of the 12.5 deg turn amplitude by the bounded rate VNC.

    env -u PYTHONPATH python scripts/f5_reach_run.py --part setup          # graphs, shuffle, inputs (once)
    env -u PYTHONPATH python scripts/f5_reach_run.py --part true_I1        # P0, P1, P2 on one graph and input set
    ... true_I2, shuffled_I1, shuffled_I2, linear
    env -u PYTHONPATH python scripts/f5_reach_run.py --part all            # every missing part in order

Output under runs/f5/reach/ (not committed). A finished part (<part>.json) is skipped; P2 resumes from its latest
checkpoint (every 25 steps, train/checkpoint.py).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from engine import rate_torch as RT  # noqa: E402
from train import checkpoint as CK  # noqa: E402
from train.graphs import degree_preserving_shuffle, shuffle_check  # noqa: E402

OUT = ROOT / "runs" / "f5" / "reach"
G1, A_MAX = 0.395, 12.5
LATERAL = {"b2": 1, "b3": -1, "i1": -1, "i2": -1, "hg1": -1, "hg2": -1, "hg3": -1, "hg4": -1}
SCALE = G1 * RT.R_IN                                  # 39.5 deg per unit of sum sigma (sL - sR)
I1_DRIVES = [(f, s, r) for r in (25, 100) for f in ("DNa04", "DNa15") for s in ("L", "R")]
I1_T, I1_WIN = 200, (100, 200)
I2_T, I2_WIN = 300, (120, 300)
I2_DIRS = (("f3b1", range(0, 10)), ("f3b3", range(10, 30)), ("mv2/t3_R", range(30, 40)))
SHUFFLE_SEED = 20261501
LR, STEPS, BATCH, EVAL_EVERY, CKPT_EVERY, ORDER_SEED = 0.05, 200, 8, 50, 25, 16
FILTER_MS = (10.0, 25.0, 50.0, 100.0)
PARTS = ("linear", "true_I1", "true_I2", "shuffled_I1", "shuffled_I2")


def setup():
    import f3a_vnc_run as F
    import f3b3_common as C
    from coupling.dn_bridge import DNBridge, family
    from f3a2_vnc_run import groups_of
    from f3a3_subgraph import vnc_subgraph
    path = OUT / "setup.npz"
    if path.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cx, _ = vnc_subgraph(F.load())
    n = cx.neurons
    dn = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    g = RT.RateGraph.from_connectome(cx, inputs=dn)
    col = {int(i): k for k, i in enumerate(g.inputs)}
    groups = groups_of(n)
    types = F.readout_types(n)
    ro, sig = [], []
    for t, s in LATERAL.items():
        for side in ("left", "right"):
            ii = np.flatnonzero((types[t][1] & (n["side"] == side)).to_numpy())
            if len(ii) != 1:
                raise SystemExit(f"{t} {side}: {len(ii)} neurons (one expected)")
            ro.append(int(ii[0]))
            sig.append(s if side == "left" else -s)          # dA = SCALE * sum sigma (sL - sR)
    base = np.zeros(len(g.inputs), np.float32)
    for k in ("DNg02_L", "DNg02_R"):
        base[[col[int(i)] for i in groups[k]]] = 1.0
    # I1
    x1 = np.repeat(base[None, None], I1_T, 0).repeat(len(I1_DRIVES), 1)
    c1 = np.zeros(len(I1_DRIVES), np.float32)
    for b, (f, s, r) in enumerate(I1_DRIVES):
        x1[:, b, [col[int(i)] for i in groups[f"{f}_{s}"]]] = r / RT.R_IN
        c1[b] = 1.0 if s == "R" else -1.0
    # I2
    banc_tab = pd.DataFrame(dict(cell_type=n["cell_type"].astype(str).to_numpy()[dn],
                                 side=n["side"].astype(str).to_numpy()[dn], index=dn))
    recs, shas, hashes = [], {}, set()
    for d, seeds in I2_DIRS:
        for s in seeds:
            for stim in ("rot_left", "rot_right"):
                p = ROOT / "runs" / d / "brain" / f"{stim}_s{s}.npz"
                z = np.load(p, allow_pickle=True)
                recs.append((stim, s, z))
                shas[f"{d}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
                hashes.add((str(z["model_sha256"]), str(z["weights_sha256"])))
    if len(hashes) != 1:
        raise SystemExit(f"records come from more than one brain model: {hashes}")
    z0 = recs[0][2]
    fw_tab = pd.DataFrame(dict(cell_type=z0["dn_cell_type"], side=z0["dn_side"]))
    bridge = DNBridge(fw_tab, banc_tab, C.bridge_config("ensemble"))
    tgt = np.array([col[int(i)] for i in bridge.target_index])
    x2 = np.repeat(base[None, None], I2_T, 0).repeat(len(recs), 1)
    c2 = np.zeros(len(recs), np.float32)
    units = np.zeros((I2_T, len(recs), len(bridge.units)), np.float32)
    for b, (stim, s, z) in enumerate(recs):
        if not (np.array_equal(z["dn_cell_type"], z0["dn_cell_type"]) and np.array_equal(z["dn_side"], z0["dn_side"])):
            raise SystemExit("DN tables differ between records")
        x2[:, b, tgt] = bridge.rates(z["dn_counts"]) / RT.R_IN
        units[:, b] = bridge.unit_rates(z["dn_counts"]) / RT.R_IN
        c2[b] = 1.0 if stim == "rot_right" else -1.0
    # shuffled graph
    t1 = time.time()
    post_s, info = degree_preserving_shuffle(g.pre, g.post, g.n, SHUFFLE_SEED)
    info["wall_s"] = time.time() - t1
    info["check"] = shuffle_check(g.pre, g.post, post_s, g.n, g.sign)
    np.savez_compressed(path, n=g.n, pre=g.pre, post=g.post, post_shuffled=post_s, n_syn=g.n_syn, sign=g.sign,
                        inputs=g.inputs, readout=np.array(ro), sigma=np.array(sig, np.float32),
                        x1=x1, c1=c1, x2=x2, c2=c2, units2=units,
                        trials2=np.array([f"{stim}_s{s}" for stim, s, _ in recs]))
    meta = dict(graph_sha256=g.sha256(), n_neurons=g.n, n_edges=g.n_edges, n_inputs=int(len(g.inputs)),
                readout=[f"{t}|{sd}" for t in LATERAL for sd in ("left", "right")], sigma=sig,
                i1_drives=[f"{f}_{s}_{r}" for f, s, r in I1_DRIVES], i2_records_sha256=shas,
                brain_model=sorted(hashes)[0], bridge=dict(n_units=len(bridge.units), n_targets=int(len(tgt)),
                                                           config=bridge.config.record()),
                shuffle=info, wall_s=time.time() - t0)
    (OUT / "setup.json").write_text(json.dumps(meta, indent=1))
    print("setup done", round(time.time() - t0), "s; shuffle check", info["check"]["passed"], flush=True)


def load_setup():
    return np.load(OUT / "setup.npz")


def make_net(z, graph_name):
    post = z["post"] if graph_name == "true" else z["post_shuffled"]
    g = RT.RateGraph.from_arrays(int(z["n"]), z["pre"], post, z["n_syn"], z["sign"], z["inputs"])
    return RT.RateNet(g)


def evaluate(net, x, c, ro, sig, win, batch=16):
    """Per trial: D (window mean of c dA), and the largest per-interval |dA|."""
    D, mx = [], []
    with torch.no_grad():
        for a in range(0, x.shape[1], batch):
            st, _ = net.run(torch.from_numpy(x[:, a:a + batch]), record=ro)
            dA = SCALE * (st * torch.from_numpy(sig)).sum(2).numpy()            # (T, b)
            D += (c[a:a + batch] * dA[win[0]:win[1]].mean(0)).tolist()
            mx += np.abs(dA).max(0).tolist()
    D = np.array(D)
    return dict(reach=float(D.mean()), min_D=float(D.min()), frac_ge_target=float((D >= A_MAX).mean()),
                frac_ge_half=float((D >= A_MAX / 2).mean()), max_abs_dA=float(max(mx)), D=D.tolist())


def run_part(part):
    out = OUT / f"{part}.json"
    if out.exists():
        print(part, "exists, skipped", flush=True)
        return
    z = load_setup()
    graph_name, inp = part.split("_")
    x, c = (z["x1"], z["c1"]) if inp == "I1" else (z["x2"], z["c2"])
    win = I1_WIN if inp == "I1" else I2_WIN
    ro, sig = z["readout"], z["sigma"]
    net = make_net(z, graph_name)
    res = dict(part=part, graph=graph_name, inputs=inp, n_trials=int(x.shape[1]), window=list(win),
               graph_sha256=net.graph.sha256(), lr=LR, steps=STEPS, batch=BATCH, threads=torch.get_num_threads())
    res["P0"] = evaluate(net, x, c, ro, sig, win)
    with torch.no_grad():
        theta0 = net.theta.detach().clone()
        net.theta.fill_(40.0)                               # sigmoid(40) == 1.0 in float32: every gain = G_MAX
        assert float(net.gains().min()) == RT.G_MAX
        res["P1"] = evaluate(net, x, c, ro, sig, win)
        net.theta.copy_(theta0)
    print(part, "P0", round(res["P0"]["reach"], 3), "P1", round(res["P1"]["reach"], 3), flush=True)
    # P2
    ck_dir = OUT / part
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(ORDER_SEED)
    n_tr = x.shape[1]
    if inp == "I2":
        order = np.concatenate([rng.permutation(n_tr) for _ in range(STEPS * BATCH // n_tr + 1)])
    start, evals, step_s, J_hist = 0, [], [], []
    last = CK.latest(ck_dir)
    if last is not None:
        ck = CK.load(last, net, opt)
        start, evals, step_s, J_hist = ck["step"], ck["extra"]["evals"], ck["extra"]["step_s"], ck["extra"]["J"]
        print(part, "resumed at step", start, flush=True)
    xt, ct = torch.from_numpy(x), torch.from_numpy(c)
    sg = torch.from_numpy(sig)
    for step in range(start, STEPS):
        t0 = time.time()
        idx = np.arange(n_tr) if inp == "I1" else order[step * BATCH:(step + 1) * BATCH]
        opt.zero_grad()
        st, _ = net.run(xt[:, idx], record=ro)
        dA = SCALE * (st * sg).sum(2)
        J = (ct[idx] * dA[win[0]:win[1]]).mean() / A_MAX
        (-J).backward()
        opt.step()
        step_s.append(time.time() - t0)
        J_hist.append(float(J))
        if (step + 1) % EVAL_EVERY == 0:
            e = evaluate(net, x, c, ro, sig, win)
            e.pop("D")
            evals.append(dict(step=step + 1, **e))
            print(part, "step", step + 1, "reach", round(e["reach"], 3), "J", round(float(J), 4),
                  f"{np.mean(step_s[-EVAL_EVERY:]):.1f} s/step", flush=True)
        if (step + 1) % CKPT_EVERY == 0:
            CK.save(ck_dir, step + 1, net, opt, extra=dict(evals=evals, step_s=step_s, J=J_hist))
    res["P2"] = evaluate(net, x, c, ro, sig, win)
    res["P2"]["evals"] = evals
    res["P2"]["J"] = J_hist
    j50 = [e["reach"] for e in evals if e["step"] in (STEPS - EVAL_EVERY, STEPS)]
    res["P2"]["not_converged"] = bool(len(j50) == 2 and j50[1] > 1.01 * j50[0])
    res["P2"]["gains"] = dict(min=float(net.gains().min()), max=float(net.gains().max()),
                              frac_above_1_9=float((net.gains() > 1.9).float().mean()),
                              frac_below_0_1=float((net.gains() < 0.1).float().mean()))
    res["P2"]["leaks"] = dict(min=float(net.leaks().min()), max=float(net.leaks().max()))
    res["step_s"] = step_s
    res["step_s_mean"] = float(np.mean(step_s))
    out.write_text(json.dumps(res))
    print(part, "done: P2 reach", round(res["P2"]["reach"], 3), flush=True)


def run_linear():
    out = OUT / "linear.json"
    if out.exists():
        print("linear exists, skipped", flush=True)
        return
    z = load_setup()
    u, c = z["units2"].astype(np.float64), z["c2"].astype(np.float64)       # (T, trials, units)
    feats = []
    for tau in FILTER_MS:
        a = np.exp(-5.0 / tau)
        y = np.zeros_like(u[0])
        f = np.empty_like(u)
        for k in range(len(u)):
            y = a * y + (1 - a) * u[k]
            f[k] = y
        feats.append(f)
    X = np.concatenate(feats + [np.ones(u.shape[:2] + (1,))], axis=2)[I2_WIN[0]:I2_WIN[1]]   # (W, trials, F)
    y = np.broadcast_to(c * A_MAX, X.shape[:2])
    w, *_ = np.linalg.lstsq(X.reshape(-1, X.shape[2]), y.reshape(-1), rcond=None)
    res = dict(part="linear", n_features=int(X.shape[2]), filters_ms=list(FILTER_MS), window=list(I2_WIN))
    for k in (1.0, 2.0, 10.0):
        dA = X @ (k * w)                                                        # (W, trials)
        D = c * dA.mean(0)
        res[f"scale_{k:g}"] = dict(reach=float(D.mean()), min_D=float(D.min()),
                                   frac_ge_target=float((D >= A_MAX).mean()),
                                   frac_ge_half=float((D >= A_MAX / 2).mean()), max_abs_dA=float(np.abs(dA).max()))
    out.write_text(json.dumps(res, indent=1))
    print("linear done", res["scale_1"], flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("setup", "all") + PARTS)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.part in ("setup", "all"):
        setup()
    for p in (PARTS if a.part == "all" else (a.part,) if a.part != "setup" else ()):
        if p == "linear":
            run_linear()
        else:
            run_part(p)
    if a.part == "all":
        (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
