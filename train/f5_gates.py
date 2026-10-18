"""F5 stage-A gates (prereg/14 section 9, with prereg/17 section 3 for gate 5 and prereg/18 for gate 3).

1. Information alignment (rule 9): cross-validated R2 > 0 of the trial-mean dA*, 5-fold by training data seed (fold =
   seed mod 5), ridge with alpha from 10^-3 .. 10^3 (7 values) chosen by an inner 5-fold CV of the same kind:
   - on the window-mean 224 bridge units;
   - on a fixed Gaussian projection (`default_rng(9)`, 2,048 dimensions) of the window-mean FlyVis output (the
     boundary rates of both eyes).
   Training data seeds only; the 6 training conditions. The window-mean dA* equals the trial-mean dA* times a constant
   (dA* is 0 in the lead-in), and ridge R2 does not depend on that scale.
2. Symmetry (rule 5): |log2(R/L)| < 0.5 of total and excitatory input synapses for the 112 bridge families (FlyWire and
   BANC, each on its own graph) and for the 8 readout types on the VNC subgraph.
3. Engine: on a 2,000-neuron induced subgraph of the VNC, sddmm backward against dense float64 autograd (max relative
   error < 1e-6) and forward against NumPy over 300 intervals (max absolute difference < 1e-5, float32); causality of
   the whole stage-A input path; resume bit-identity of the stage-A trainer.
4. Shuffle check of every training seed's shuffled VNC.
5. Records: every record matches the manifest, every input check passed, and the prereg/17 bit-identity check passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from engine import rate_torch as RT
from train import f5_stageA as A

ALPHAS = tuple(10.0 ** np.arange(-3, 4))
N_FOLDS = 5
PROJ_DIM, PROJ_SEED = 2048, 9
SUBGRAPH_N, SUBGRAPH_SEED = 2000, 3
SYM_THR = 0.5


# ── 1. information alignment ────────────────────────────────────────────────────────────────────────────────────────
def _ridge_fit(X, y, alpha):
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    n, p = Xc.shape
    if p <= n:
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ (y - my))
    else:                                            # dual form when there are more features than samples
        w = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(n), y - my)
    return w, my - mx @ w


def _folds(seeds, k=N_FOLDS):
    return np.asarray(seeds) % k


def ridge_cv_r2(X, y, seeds) -> dict:
    """Outer 5-fold CV by data seed; alpha per outer fold from an inner 5-fold CV by data seed (lowest MSE). R2 = 1 -
    SSE / SST over the pooled out-of-fold predictions."""
    X, y, seeds = np.asarray(X, np.float64), np.asarray(y, np.float64), np.asarray(seeds)
    fold = _folds(seeds)
    pred = np.zeros_like(y)
    chosen = []
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        s_tr = seeds[tr]
        inner = (s_tr // N_FOLDS) % N_FOLDS             # a different split of the outer-training seeds
        mse = []
        for a in ALPHAS:
            e = 0.0
            for g in range(N_FOLDS):
                itr, ite = inner != g, inner == g
                if not ite.any() or not itr.any():
                    continue
                w, b = _ridge_fit(X[tr][itr], y[tr][itr], a)
                e += float(((X[tr][ite] @ w + b - y[tr][ite]) ** 2).sum())
            mse.append(e)
        a = ALPHAS[int(np.argmin(mse))]
        chosen.append(float(a))
        w, b = _ridge_fit(X[tr], y[tr], a)
        pred[te] = X[te] @ w + b
    return dict(r2=A.r2(pred, y), alphas=chosen, n=int(len(y)), n_features=int(X.shape[1]),
                folds="data seed mod 5 (outer); (seed // 5) mod 5 (inner)", passed=bool(A.r2(pred, y) > 0))


def window_mean(a):
    """(T, N, F) -> (N, F) over the analysis window."""
    return np.asarray(a)[A.WINDOW[0]:A.WINDOW[1]].mean(0)


def projection(n_in, dim=PROJ_DIM, seed=PROJ_SEED):
    return (np.random.default_rng(seed).standard_normal((n_in, dim)) / np.sqrt(dim)).astype(np.float32)


# ── 2. symmetry ─────────────────────────────────────────────────────────────────────────────────────────────────────
def symmetry_gate(families, fw_rows: dict, banc_rows: dict, type_rows) -> dict:
    """``fw_rows`` / ``banc_rows``: scripts/f3_dn_crosswalk.per_family of each dataset; ``type_rows``:
    connectome.qa_report.type_symmetry of the VNC subgraph restricted to the 8 readout types."""
    fam = {f: dict(flywire=fw_rows.get(f, {}).get("gate"), banc=banc_rows.get(f, {}).get("gate"),
                   log2_total_flywire=fw_rows.get(f, {}).get("log2_total"),
                   log2_total_banc=banc_rows.get(f, {}).get("log2_total")) for f in families}
    types = {r["type"]: dict(log2_total=float(r["log2_total"]), log2_exc=float(r["log2_exc"]), passed=bool(r["pass"]))
             for r in type_rows}
    fail_fam = [f for f, v in fam.items() if not (v["flywire"] and v["banc"])]
    fail_types = [t for t, v in types.items() if not v["passed"]]
    return dict(threshold=SYM_THR, families=fam, readout_types=types, failing_families=fail_fam,
                failing_readout_types=fail_types, n_families=len(fam), n_readout_types=len(types),
                passed=bool(not fail_fam and not fail_types and len(types) == 8))


# ── 3. engine ───────────────────────────────────────────────────────────────────────────────────────────────────────
def gate_subgraph(setup: A.Setup, bridge, n=SUBGRAPH_N, seed=SUBGRAPH_SEED) -> RT.RateGraph:
    """Induced subgraph: the 16 readout MNs, the DNg02 baseline DNs, the bridge targets, then random VNC neurons."""
    g = setup.graph()
    keep = set(np.asarray(setup.readout).tolist())
    keep |= set(np.asarray(setup.inputs)[np.asarray(setup.base) != 0].tolist())
    if bridge is not None:
        keep |= set(bridge.target_index.tolist())
    rest = np.setdiff1d(np.arange(g.n), np.fromiter(keep, np.int64))
    rng = np.random.default_rng(seed)
    keep = np.union1d(np.fromiter(keep, np.int64), rng.choice(rest, size=n - len(keep), replace=False))
    return g.induced(keep)


def _randomise(net, seed=2):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.theta.copy_(torch.randn(net.theta.shape, generator=gen, dtype=net.theta.dtype))
        net.phi.copy_(torch.randn(net.phi.shape, generator=gen, dtype=net.phi.dtype))


def _rel_err(a, b):
    a, b = a.detach().numpy(), b.detach().numpy()
    return float(np.abs(a - b).max() / np.abs(b).max())


def engine_gradient_and_forward(sub: RT.RateGraph, T_grad=40, batch=2, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    x64 = rng.uniform(0, 1, (T_grad, batch, len(sub.inputs)))
    net = RT.RateNet(sub, dtype=torch.float64, edge_grad=A.EDGE_GRAD)
    _randomise(net)
    coef = torch.from_numpy(np.random.default_rng(3).normal(size=(T_grad, batch, sub.n)))
    x = torch.from_numpy(x64)
    st, _ = net.run(x)
    (st * coef).sum().backward()
    th = net.theta.detach().clone().requires_grad_(True)
    ph = net.phi.detach().clone().requires_grad_(True)
    ref = RT.dense_reference_torch(sub, th, ph, x)
    (ref * coef).sum().backward()
    g_theta, g_phi = _rel_err(net.theta.grad, th.grad), _rel_err(net.phi.grad, ph.grad)
    x32 = rng.uniform(0, 1, (A.T_TRIAL, batch, len(sub.inputs))).astype(np.float32)
    net32 = RT.RateNet(sub, edge_grad=A.EDGE_GRAD)
    _randomise(net32)
    with torch.no_grad():
        s32, _ = net32.run(torch.from_numpy(x32))
        ref32 = RT.reference_numpy(sub, net32.gains().numpy(), net32.leaks().numpy(), x32)
    fwd = float(np.abs(s32.numpy() - ref32).max())
    return dict(n_neurons=sub.n, n_edges=sub.n_edges, n_inputs=int(len(sub.inputs)), edge_grad=A.EDGE_GRAD,
                grad_intervals=T_grad, rel_err_theta=g_theta, rel_err_phi=g_phi, forward_intervals=A.T_TRIAL,
                forward_max_abs_diff=fwd, forward_max_abs_state=float(np.abs(ref32).max()),
                passed=bool(g_theta < 1e-6 and g_phi < 1e-6 and fwd < 1e-5))


def causality(model, bridge, counts, j, device="cpu") -> dict:
    """Perturb the DN counts of interval j (every FlyWire DN of the bridge +3 spikes) and find the first interval at
    which dA changes. The registered total latency from DN spikes to dA is the bridge latency (1 interval): the clamp of
    interval j + 1 is the first to see interval j, and a readout MN is at least one hop from a DN, which acts in the
    interval of its clamp (engine: clamp latency 0, hop latency 1). The body would get dA one interval later still."""
    c2 = counts.astype(np.int64).copy()
    used = np.unique(np.concatenate([u["fw_pos"] for u in bridge.units]))
    c2[j, used] += 3
    u0 = (bridge.unit_rates(counts) / RT.R_IN).astype(np.float32)[:, None]
    u1 = (bridge.unit_rates(c2) / RT.R_IN).astype(np.float32)[:, None]
    model = model.to(device)
    with torch.no_grad():
        d0 = model(torch.from_numpy(u0).to(device)).double().cpu().numpy()[:, 0]
        d1 = model(torch.from_numpy(u1).to(device)).double().cpu().numpy()[:, 0]
    diff = np.abs(d1 - d0)
    changed = np.flatnonzero(diff > 0)
    first = int(changed[0]) if len(changed) else None
    lat = A.LATENCIES["bridge_latency_intervals"]
    return dict(perturbed_interval=int(j), first_changed_interval=first, registered_latency_intervals=lat,
                command_latency_intervals=A.LATENCIES["command_latency_intervals"],
                max_abs_change_deg=float(diff.max()),
                passed=bool(first is not None and first >= j + lat and diff[:j + lat].max() == 0.0))


def resume_check(make, train, run_root, steps=4, device="cpu") -> dict:
    """Train ``steps`` steps uninterrupted, and steps / 2 + resume to ``steps`` in another directory; the parameters
    must be bit-identical. ``make()`` returns (model, record)."""
    k = steps // 2
    out = {}
    for name, stop in (("full", None), ("resumed", k)):
        d = Path(run_root) / name
        model, rec = make()
        cfg = A.RunConfig(arm="true", seed=0, lr=A.LR_GRID[-1], run_dir=d, steps=steps, eval_every=10**9,
                          ckpt_every=1, device=device, smoke=True)
        A.train_run(cfg, model, rec, train, None, log_fn=lambda *a: None, stop_after=stop)
        if stop is not None:
            torch.manual_seed(12345)                 # the resumed run must restore every RNG itself
            model, rec = make()
            A.train_run(cfg, model, rec, train, None, log_fn=lambda *a: None)
        out[name] = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
    same = all(torch.equal(out["full"][k_], out["resumed"][k_]) for k_ in out["full"])
    moved = any(not torch.equal(v, torch.zeros_like(v)) for k_, v in out["full"].items() if k_.endswith("theta"))
    return dict(steps=steps, interrupted_at=k, device=str(device), bit_identical=bool(same), parameters_moved=moved,
                passed=bool(same and moved))


# ── 4. shuffle ──────────────────────────────────────────────────────────────────────────────────────────────────────
def shuffle_gate(setup: A.Setup, seeds, cache_dir) -> dict:
    rows = {}
    for t in seeds:
        _, info = A.shuffled_post(setup, t, cache_dir)
        rows[f"t{t}"] = dict(seed=info["seed"], accepted=info["accepted"], **info["check"])
    return dict(seeds=list(seeds), checks=rows, passed=bool(all(r["passed"] for r in rows.values())))


# ── 5. records ──────────────────────────────────────────────────────────────────────────────────────────────────────
def records_gate(records: Path, step2c_json: Path, n_expected=2400) -> dict:
    """Hashes only (test records are hashed, never parsed), the stored input checks and the prereg/17 check."""
    records = Path(records)
    man = A.read_manifest(records / "manifest.sha256")
    plan = json.loads((records / "plan.json").read_text())["records"]
    files = sorted(p.relative_to(records).as_posix() for p in (records / "brain").glob("*.npz"))
    bad = [f for f, h in man.items() if A.sha256_file(records / f) != h]
    checks = json.loads((records / "input_checks.json").read_text())
    failed = [k for k, v in checks.items() if not v.get("passed")]
    g5 = json.loads(Path(step2c_json).read_text())["gate5"]
    machine = json.loads((records / "machine.json").read_text())
    res = dict(n_plan=len(plan), n_manifest=len(man), n_files=len(files),
               plan_equals_manifest=sorted(r["file"] for r in plan) == sorted(man),
               files_equal_manifest=files == sorted(man), hash_mismatches=bad, n_input_checks=len(checks),
               input_checks_failed=failed, prereg17_bit_identical=bool(g5["bit_identical"]),
               machine=dict(cpu=machine.get("cpu"), env_file_sha256=machine.get("env_file_sha256")))
    res["passed"] = bool(res["n_plan"] == n_expected and res["plan_equals_manifest"] and res["files_equal_manifest"]
                         and not bad and not failed and res["prereg17_bit_identical"])
    return res
