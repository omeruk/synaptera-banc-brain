"""F5 step 2b: wall-time-only benchmark of the registered training steps on this machine (prereg/14 section 10: data
seed 9999, not registered; no metric is read; the inputs are synthetic).

    env -u PYTHONPATH python scripts/f5_bench_run.py --part a_gpu      # stage A: VNC, batch 16, 300 intervals
    ... a_cpu | b_gpu | b_cpu | flyvis | all
    ... --part b_gpu --batch 2 --intervals 50 --ckpts 0      # a size probe: runs/f5/bench/b_gpu_probe_b2_t50_ck0.json

Output: runs/f5/bench/<part>.json (not committed; existing parts are skipped). Report: scripts/f5_bench_report.py.
One GPU job at a time (4-5 CUDA processes at once caused nvlddmkm driver faults on this machine with driver 596.36);
each CUDA process is capped at GPU_FRACTION of the GPU memory. Run every part through scripts/f5_bench_guard.py.
Graphs: the VNC of runs/f5/reach/setup.npz (prereg/08 subgraph, prereg/01 signs); the FlyWire v783 brain (F0 signs);
the prereg/11 `ensemble` bridge (engine/rate_hybrid.py). Deterministic algorithms are on, as in training.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from engine import rate_torch as RT  # noqa: E402

OUT = ROOT / "runs" / "f5" / "bench"
SETUP = ROOT / "runs" / "f5" / "reach" / "setup.npz"
SEED = 9999
A_BATCH, B_BATCH, T_TRIAL, T_FLIGHT = 16, 8, 300, 500
G1, A_MAX = 0.395, 12.5
SCALE = G1 * RT.R_IN
GPU_FRACTION = 0.7                             # per-process cap on the GPU memory
FILTERS = 4                                    # exponential filters of the mlp / linear input (prereg/14 section 5)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def cool_down(dev, below=70, max_wait=600):
    """Before a measured GPU config: wait until the GPU is below ``below`` C (the laptop GPU reaches 85 C under load;
    the guard kills at 87 C). The wait is not part of any timing."""
    if dev != "cuda":
        return
    t0 = time.time()
    while time.time() - t0 < max_wait:
        q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.strip()
        if q.isdigit() and int(q) < below:
            return
        time.sleep(5)


def machine():
    cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
    d = dict(machine=platform.machine(), cpu=cpu[0], cpu_threads=len(cpu), torch_threads=torch.get_num_threads(),
             torch=torch.__version__, python=platform.python_version(), deterministic=True,
             peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        d.update(gpu=torch.cuda.get_device_name(0), gpu_total_gib=total / 2**30, gpu_free_gib_at_start=free / 2**30,
                 cuda=torch.version.cuda, gpu_fraction_cap=GPU_FRACTION)
    return d


# ── stage A ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def vnc_parts():
    z = np.load(SETUP)
    g = RT.RateGraph.from_arrays(int(z["n"]), z["pre"], z["post"], z["n_syn"], z["sign"], z["inputs"])
    return g, z["readout"], z["sigma"].astype(np.float32)


def a_inputs(g, batch, T, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 0.3, (T, batch, len(g.inputs))).astype(np.float32)
    target = rng.choice([-A_MAX, 0.0, A_MAX], size=batch).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(np.repeat(target[None], T, 0))


def loss_of(states, sigma, target):
    dA = SCALE * (states * sigma).sum(2)
    return ((dA - target) / A_MAX).square().mean()


def train_steps(step_fn, dev, n_warm, n_meas):
    times = []
    cool_down(dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(n_warm + n_meas):
        sync(dev)
        t0 = time.perf_counter()
        step_fn()
        sync(dev)
        if i >= n_warm:
            times.append(time.perf_counter() - t0)
    out = dict(step_s=times, median_step_s=statistics.median(times), n_warm=n_warm, n_meas=n_meas)
    if dev == "cuda":
        out.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return out


def a_step_fn(net, opt, x, target, ro, sigma, ce):
    sig = torch.from_numpy(sigma).to(net.device)
    xd, td = x.to(net.device), target.to(net.device)

    def step():
        opt.zero_grad(set_to_none=True)
        st, _ = net.run(xd, record=ro, checkpoint_every=ce)
        loss = loss_of(st, sig, td)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss
    return step


def eval_time(run_fn, dev, n=3):
    times = []
    for _ in range(n):
        sync(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            run_fn()
        sync(dev)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def grads_of(net, step):
    """Loss and gradients of one forward + backward (no optimiser step), for the identity checks."""
    net.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss = step()
    return loss.detach().cpu(), net.theta.grad.detach().cpu().clone(), net.phi.grad.detach().cpu().clone()


def part_a(dev, n_warm, n_meas, extras):
    torch.use_deterministic_algorithms(True)
    g, ro, sigma = vnc_parts()
    x, target = a_inputs(g, A_BATCH, T_TRIAL)
    res = dict(stage="A", device=dev, graph_sha256=g.sha256(), n_neurons=g.n, n_edges=g.n_edges,
               n_trainable=g.n_trainable, batch=A_BATCH, intervals=T_TRIAL, configs={})
    for eg in RT.EDGE_GRADS:
        for ce in (0, 50):
            net = RT.RateNet(g, edge_grad=eg).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            key = f"{eg}_ckpt{ce}"
            try:
                res["configs"][key] = train_steps(a_step_fn(net, opt, x, target, ro, sigma, ce), dev, n_warm, n_meas)
            except torch.OutOfMemoryError as e:
                res["configs"][key] = dict(oom=True, error=str(e)[:200])
            del net, opt
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(dev, key, res["configs"][key].get("median_step_s", "oom"), flush=True)
    # validation / test forward (no grad, batch 240 = one validation set)
    net = RT.RateNet(g).to(dev)
    xv, _ = a_inputs(g, 240, T_TRIAL, seed=SEED + 1)
    xv = xv.to(dev)
    res["eval_s_per_trial"] = eval_time(lambda: net.run(xv, record=ro), dev, n=2) / 240
    if extras and dev == "cuda":
        res["deterministic_off"] = {eg: _det_off(g, x, target, ro, sigma, n_warm, n_meas, eg) for eg in RT.EDGE_GRADS}
        res["compile"] = {eg: _try_compile(g, x, target, ro, sigma, eg) for eg in RT.EDGE_GRADS}
        res["cuda_graph"] = {eg: _try_cuda_graph(g, x, target, ro, sigma, eg) for eg in RT.EDGE_GRADS}
        res["mlp"] = _mlp_step(g, dev, n_warm, n_meas)
    return res


def _det_off(g, x, target, ro, sigma, n_warm, n_meas, eg):
    torch.use_deterministic_algorithms(False)
    try:
        net = RT.RateNet(g, edge_grad=eg).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        return train_steps(a_step_fn(net, opt, x, target, ro, sigma, 0), "cuda", n_warm, n_meas)
    finally:
        torch.use_deterministic_algorithms(True)


def _fwd_bwd(net, x, target, ro, sigma, run=None):
    sig = torch.from_numpy(sigma).cuda()
    xd, td, rod = x.cuda(), target.cuda(), torch.as_tensor(np.asarray(ro), dtype=torch.int64).cuda()
    run = run or net.run

    def step():
        st, _ = run(xd, record=rod)
        loss = loss_of(st, sig, td)
        loss.backward()
        return loss
    return step


def _try_compile(g, x, target, ro, sigma, eg):
    """torch.compile of the forward; kept only if loss and gradients equal eager bit for bit."""
    out = dict()
    try:
        torch._dynamo.reset()
        net = RT.RateNet(g, edge_grad=eg).cuda()
        eager = grads_of(net, _fwd_bwd(net, x, target, ro, sigma))
        t0 = time.perf_counter()
        comp = torch.compile(net.run)
        c1 = grads_of(net, _fwd_bwd(net, x, target, ro, sigma, run=comp))
        out["compile_s"] = time.perf_counter() - t0
        out["identical"] = all(torch.equal(a, b) for a, b in zip(eager, c1))
        out["max_abs_grad_diff"] = float((eager[1] - c1[1]).abs().max())
        step = _fwd_bwd(net, x, target, ro, sigma, run=comp)
        ref = _fwd_bwd(net, x, target, ro, sigma)
        out["compiled"] = train_steps(lambda: (net.zero_grad(set_to_none=True), step()), "cuda", 1, 3)
        out["eager"] = train_steps(lambda: (net.zero_grad(set_to_none=True), ref()), "cuda", 1, 3)
        out["speedup"] = out["eager"]["median_step_s"] / out["compiled"]["median_step_s"]
        out["works"] = True
    except Exception as e:                                       # noqa: BLE001 (recorded, not hidden)
        out.update(works=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    return out


def _try_cuda_graph(g, x, target, ro, sigma, eg):
    """Capture forward + backward in one CUDA graph; kept only if loss and gradients equal eager bit for bit."""
    out = dict()
    try:
        net = RT.RateNet(g, edge_grad=eg).cuda()
        step = _fwd_bwd(net, x, target, ro, sigma)
        eager = grads_of(net, step)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                net.zero_grad(set_to_none=False)
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        net.zero_grad(set_to_none=False)
        with torch.cuda.graph(graph):
            loss = step()

        def replay():
            for p in net.parameters():
                p.grad.zero_()
            graph.replay()
            return loss
        replay()
        torch.cuda.synchronize()
        got = (loss.detach().cpu(), net.theta.grad.cpu().clone(), net.phi.grad.cpu().clone())
        out["identical"] = all(torch.equal(a, b) for a, b in zip(eager, got))
        out["max_abs_grad_diff"] = float((eager[1] - got[1]).abs().max())
        out["graph"] = train_steps(replay, "cuda", 1, 3)
        out["eager"] = train_steps(lambda: (net.zero_grad(set_to_none=True), step()), "cuda", 1, 3)
        out["speedup"] = out["eager"]["median_step_s"] / out["graph"]["median_step_s"]
        out["works"] = True
    except Exception as e:                                       # noqa: BLE001
        out.update(works=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    return out


def _mlp_step(g, dev, n_warm, n_meas):
    """The `mlp` arm of stage A at its registered size: input 225 units x 4 filters; h the smallest width with
    parameters >= the `true` VNC's trainable count; the filters are not timed (a cumulative sum per interval)."""
    n_in, n_out, target_params = 225 * FILTERS, 16, g.n_trainable
    h = 1
    while (n_in * h + h) + (h * h + h) + (h * n_out + n_out) < target_params:
        h += 1
    mlp = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(),
                              torch.nn.Linear(h, n_out)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    xin = torch.from_numpy(rng.uniform(0, 0.3, (T_TRIAL, A_BATCH, n_in)).astype(np.float32)).to(dev)
    sig = torch.ones(n_out, device=dev)
    tgt = torch.zeros(T_TRIAL, A_BATCH, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = loss_of(torch.tanh(mlp(xin)), sig, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
    r = train_steps(step, dev, n_warm, n_meas)
    r.update(width=h, n_params=sum(p.numel() for p in mlp.parameters()))
    return r


# ── stage B ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def hybrid():
    """Rate FlyWire brain + ensemble bridge + rate VNC (arm `true`), and the brain drive count."""
    import pandas as pd

    import f3a_vnc_run as F
    import f3b3_common as C
    from connectome.adapters import flywire783 as fw
    from coupling.dn_bridge import DNBridge
    from engine.rate_hybrid import RateHybrid
    from f3a2_vnc_run import groups_of
    from f3a3_subgraph import vnc_subgraph
    from sense import vision_boundary as VB
    t0 = time.time()
    cx = fw.load_connectome(hash_files=False)
    table = VB.load_table(ROOT / "data" / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    vg = VB.boundary_groups(cx, table)
    drives = np.concatenate([vg["vbnd_L"], vg["vbnd_R"]])
    gb = RT.RateGraph.from_connectome(cx, drives=drives)
    n = cx.neurons
    dn_b = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    fw_tab = pd.DataFrame(dict(cell_type=n["cell_type"].astype(str).to_numpy()[dn_b],
                               side=n["side"].astype(str).to_numpy()[dn_b]))
    vx, _ = vnc_subgraph(F.load())
    nv = vx.neurons
    dn_v = np.flatnonzero((nv["super_class"] == "descending").to_numpy())
    gv = RT.RateGraph.from_connectome(vx, inputs=dn_v)
    z = np.load(SETUP)
    if gv.sha256() != RT.RateGraph.from_arrays(int(z["n"]), z["pre"], z["post"], z["n_syn"], z["sign"],
                                               z["inputs"]).sha256():
        raise SystemExit("VNC graph differs from runs/f5/reach/setup.npz")
    banc_tab = pd.DataFrame(dict(cell_type=nv["cell_type"].astype(str).to_numpy()[dn_v],
                                 side=nv["side"].astype(str).to_numpy()[dn_v], index=dn_v))
    bridge = DNBridge(fw_tab, banc_tab, C.bridge_config("ensemble"))
    col = {int(i): k for k, i in enumerate(gv.inputs)}
    base = np.zeros(len(gv.inputs))
    groups = groups_of(nv)
    for k in ("DNg02_L", "DNg02_R"):
        base[[col[int(i)] for i in groups[k]]] = 1.0
    units = RateHybrid.units_from_bridge(bridge, dn_b)
    hy = RateHybrid(RT.RateNet(gb), RT.RateNet(gv), units, base, C.BRIDGE_LATENCY_INTERVALS, C.BRIDGE_WINDOW_INTERVALS)
    return hy, z["readout"], z["sigma"].astype(np.float32), time.time() - t0


def b_inputs(hy, batch, T, seed=SEED):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.4, (T, batch, len(hy.brain.drives))).astype(np.float32)
    target = rng.choice([-A_MAX, 0.0, A_MAX], size=batch).astype(np.float32)
    return torch.from_numpy(u), torch.from_numpy(np.repeat(target[None], T, 0))


def b_step_fn(hy, opt, u, target, ro, sigma, ce):
    dev = hy.vnc.device
    sig, ud, td = torch.from_numpy(sigma).to(dev), u.to(dev), target.to(dev)

    def step():
        opt.zero_grad(set_to_none=True)
        st, _, _ = hy.run(ud, record=ro, checkpoint_every=ce)
        loss = loss_of(st, sig, td)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hy.parameters(), 1.0)
        opt.step()
        return loss
    return step


def part_b(dev, n_warm, n_meas, cks, egs=RT.EDGE_GRADS, batch=B_BATCH, T=T_TRIAL, probe=False, c_shape=True):
    """The registered stage-B step (batch 8, 300 intervals), or a size probe (training steps only). c_shape: also the
    stage-C step (batch 8, a 500-interval flight)."""
    torch.use_deterministic_algorithms(True)
    hy, ro, sigma, build_s = hybrid()
    rec = hy.record()
    res = dict(stage="B", device=dev, build_s=build_s, batch=batch, intervals=T, probe=probe, record=rec,
               bridge_units=hy.n_units,
               n_trainable=rec["brain"]["n_trainable"] + rec["vnc"]["n_trainable"], configs={})
    hy.to(dev)
    u, target = b_inputs(hy, batch, T)
    res["c_shape"], res["c_shape_ckpt"] = {}, {}
    for eg in egs:
        hy.brain.edge_grad = hy.vnc.edge_grad = eg
        for ce in cks:
            key = f"{eg}_ckpt{ce}"
            opt = torch.optim.Adam(hy.parameters(), lr=1e-3)
            try:
                res["configs"][key] = train_steps(b_step_fn(hy, opt, u, target, ro, sigma, ce), dev, n_warm, n_meas)
            except torch.OutOfMemoryError as e:
                res["configs"][key] = dict(oom=True, error=str(e)[:200])
            del opt
            hy.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(dev, key, res["configs"][key].get("median_step_s", "oom"), flush=True)
        ok = [c for c in cks if "median_step_s" in res["configs"][f"{eg}_ckpt{c}"]]
        uc, tc = b_inputs(hy, B_BATCH, T_FLIGHT)
        for ce in (ok if c_shape and not probe else []):    # the first checkpoint setting that fits 500 intervals
            res["c_shape_ckpt"][eg] = ce
            opt = torch.optim.Adam(hy.parameters(), lr=1e-3)
            try:
                res["c_shape"][eg] = train_steps(b_step_fn(hy, opt, uc, tc, ro, sigma, ce), dev, 1, max(1, n_meas - 1))
            except torch.OutOfMemoryError as e:
                res["c_shape"][eg] = dict(oom=True, error=str(e)[:200])
            del opt
            hy.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(dev, eg, "c_shape ckpt", ce, res["c_shape"][eg].get("median_step_s", "oom"), flush=True)
            if "median_step_s" in res["c_shape"][eg]:
                break
    hy.brain.edge_grad = hy.vnc.edge_grad = "chunked"
    if probe:
        return res
    uv, _ = b_inputs(hy, B_BATCH, T_TRIAL, seed=SEED + 1)
    uv = uv.to(dev)
    res["eval_s_per_trial"] = eval_time(lambda: hy.run(uv, record=ro), dev, n=2) / B_BATCH
    for nb in (1, 160):
        ui, _ = b_inputs(hy, nb, 20, seed=SEED + 2)
        ui = ui.to(dev)
        res[f"forward_s_per_interval_batch{nb}"] = eval_time(lambda: hy.run(ui, record=ro), dev, n=2) / 20
    return res


# ── online vision (stage C rollouts) ────────────────────────────────────────────────────────────────────────────────
def part_flyvis(n_int):
    """CPU time per interval of the online visual path of a closed-loop flight: scene luminance -> retina encoding ->
    FlyVis (CPU) -> boundary rates, as the F3b render loop (rot_left, 60 deg/s)."""
    import mujoco
    from flygym.vision import Retina

    import f3b3_common as C
    from body.flygym_body import FlightBody
    from f3b1_brain import build
    from tasks import t0_stimuli as S
    _, _, eyes = build(log=lambda *a: None)
    b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs="tuck", enable_vision=True)
    mujoco.mj_forward(b.m, b.d)
    rays = S.camera_rays(b.m, b.d, b.root)
    retina = Retina()
    dt = C.INTERVAL_MS * 1e-3
    cond = C.STIMULI["rot_left"][0]
    eyes.reset()
    t_scene, t_eyes = [], []
    for k in range(n_int):
        t0 = time.perf_counter()
        frame = S.encode(S.luminance(cond, (k + 1) * dt, rays), retina)
        t1 = time.perf_counter()
        act, _ = eyes.step([frame])
        eyes.rates(act)
        t2 = time.perf_counter()
        t_scene.append(t1 - t0)
        t_eyes.append(t2 - t1)
    return dict(n_intervals=n_int, scene_s_per_interval=statistics.median(t_scene),
                flyvis_s_per_interval=statistics.median(t_eyes),
                total_s_per_interval=statistics.median(np.add(t_scene, t_eyes).tolist()))


PARTS = ("a_gpu", "a_cpu", "b_gpu", "b_cpu", "flyvis")


def run_part(p):
    if p == "a_gpu":
        return part_a("cuda", 2, 10, extras=True)
    if p == "a_cpu":
        return part_a("cpu", 1, 3, extras=False)
    if p == "b_gpu":
        return part_b("cuda", 1, 3, (0, 50, 25))
    if p == "b_cpu":
        return part_b("cpu", 1, 1, (50,), c_shape=False)
    if p == "flyvis":
        return part_flyvis(200)
    raise ValueError(p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=PARTS + ("all",))
    ap.add_argument("--batch", type=int, help="stage-B size probe: batch")
    ap.add_argument("--intervals", type=int, help="stage-B size probe: intervals")
    ap.add_argument("--ckpts", default="0", help="stage-B size probe: checkpoint_every values, comma-separated")
    ap.add_argument("--edge-grads", default="chunked", help="stage-B size probe: edge gradients, comma-separated")
    a = ap.parse_args(argv)
    probe = a.batch is not None or a.intervals is not None
    if probe and a.part not in ("b_gpu", "b_cpu"):
        ap.error("--batch / --intervals are for b_gpu and b_cpu")
    if a.part in ("a_gpu", "b_gpu", "all"):              # CPU parts never create a CUDA context
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (PARTS if a.part == "all" else (a.part,)):
        cks = tuple(int(c) for c in a.ckpts.split(","))
        name = f"{p}_probe_b{a.batch or B_BATCH}_t{a.intervals or T_TRIAL}_ck{'-'.join(map(str, cks))}" \
            if probe else p
        path = OUT / f"{name}.json"
        if path.exists():
            print(name, "exists; skipped", flush=True)
            continue
        t0 = time.time()
        if probe:
            res = part_b("cuda" if p == "b_gpu" else "cpu", 1, 2, cks, tuple(a.edge_grads.split(",")),
                         batch=a.batch or B_BATCH, T=a.intervals or T_TRIAL, probe=True)
        else:
            res = run_part(p)
        res.update(part=name, seed=SEED, machine=machine(), wall_s=time.time() - t0)
        path.write_text(json.dumps(res, indent=1))
        print(name, "done", round(time.time() - t0), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
