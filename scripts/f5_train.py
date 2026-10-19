"""F5 stage-A training runner (prereg/14, prereg/17, prereg/18; library: train/f5_stageA.py, train/f5_gates.py).

    env -u PYTHONPATH python scripts/f5_train.py setup                       # VNC graph, readout, DN tables (once)
    env -u PYTHONPATH python scripts/f5_train.py gates --part records        # ... shuffle engine causality resume
                                                                             # alignment_bridge flyvis_render
                                                                             # alignment_flyvis symmetry summary
    env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 0 --lr 1e-3     # one run (resumes)
    env -u PYTHONPATH python scripts/f5_train.py select-lr --arm true        # after the three seed-0 runs
    env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 3     # seeds 1-9: the selected rate
    env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 0 --replicate   # prereg/19 second run
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_train.py final-eval --arm true --seed 3
                                                    # the ONLY reader of test seeds; CPU (prereg/19)
    env -u PYTHONPATH python scripts/f5_train.py curves          # PNGs of every run, copied to the preview folder
    env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 0 --lr 1e-2 --smoke --steps 20

Every GPU command runs through scripts/f5_train_guard.py (one GPU job at a time; pause and resume on heat or low
memory). Output: runs/f5/A/ (not committed); smoke runs in runs/f5/A_smoke/, marked smoke in every file, evaluated on
training trials only, and never read by select-lr, final-eval or a report. A training run needs runs/f5/A/gates.json
with every gate passed; a finished run (DONE) or a test evaluation (test.json) is never overwritten (rule 13).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from engine import rate_torch as RT  # noqa: E402
from train import f5_gates as G  # noqa: E402
from train import f5_stageA as A  # noqa: E402

SETUP = A.OUT / "setup.npz"
GATES = A.OUT / "gates"
GRAPHS = A.OUT / "graphs"
STEP2C_JSON = ROOT / "docs" / "results" / "f5_stageA_records.json"
LATERAL = {"b2": 1, "b3": -1, "i1": -1, "i2": -1, "hg1": -1, "hg2": -1, "hg3": -1, "hg4": -1}
REACH_GRAPH_SHA256 = "b10789886fe5ede36674d416fd389daee3583c3deca711b8390b54cdfbdf77a0"   # runs/f5/reach, PR #25
GATE_PARTS = ("records", "shuffle", "engine", "causality", "resume", "alignment_bridge", "alignment_flyvis",
              "symmetry")
FLYVIS_WORKERS = 2
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")     # machine 2: learning curves for viewing in Windows
PREVIEW_TRIES, PREVIEW_WAIT_S = 3, 1.0                            # best-effort preview copy (copy_preview)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


# ── setup ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_setup():
    import f3a_vnc_run as F
    import f3b3_common as C
    from f3a2_vnc_run import groups_of
    from f3a3_subgraph import vnc_subgraph
    if SETUP.exists():
        print("setup exists", flush=True)
        return
    t0 = time.time()
    cx, _ = vnc_subgraph(F.load())
    n = cx.neurons
    dn = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    g = RT.RateGraph.from_connectome(cx, inputs=dn)
    col = {int(i): k for k, i in enumerate(g.inputs)}
    groups, types = groups_of(n), F.readout_types(n)
    ro, sig, names = [], [], []
    for t, s in LATERAL.items():
        for side in ("left", "right"):
            ii = np.flatnonzero((types[t][1] & (n["side"] == side)).to_numpy())
            if len(ii) != 1:
                raise SystemExit(f"{t} {side}: {len(ii)} neurons (one expected)")
            ro.append(int(ii[0]))
            sig.append(s if side == "left" else -s)            # dA = SCALE * sum sigma (s_L - s_R)
            names.append(f"{t}|{side}")
    base = np.zeros(len(g.inputs), np.float32)
    for k in ("DNg02_L", "DNg02_R"):
        base[[col[int(i)] for i in groups[k]]] = 1.0
    A.check_seed(0)
    z = np.load(A.RECORDS / "brain" / A.record_name("rot_left", 60, 0), allow_pickle=True)   # training seed 0
    setup = A.Setup(n=g.n, pre=g.pre, post=g.post, n_syn=g.n_syn, sign=g.sign, inputs=g.inputs,
                    readout=np.array(ro, np.int64), sigma=np.array(sig, np.float32), base=base,
                    banc_cell_type=n["cell_type"].astype(str).to_numpy()[dn].astype("U"),
                    banc_side=n["side"].astype(str).to_numpy()[dn].astype("U"), banc_index=dn.astype(np.int64),
                    fw_cell_type=np.asarray(z["dn_cell_type"]).astype("U"), fw_side=np.asarray(z["dn_side"]).astype("U"),
                    fw_index=np.asarray(z["dn_index"], np.int64), families=tuple(C.ENSEMBLE))
    bridge = A.make_bridge(setup, "true", 0)
    setup.meta = dict(graph_sha256=g.sha256(), reach_graph_sha256=REACH_GRAPH_SHA256,
                      graph_equals_reach=g.sha256() == REACH_GRAPH_SHA256, n_neurons=g.n, n_edges=g.n_edges,
                      n_inputs=int(len(g.inputs)), n_trainable=g.n_trainable, readout_names=names,
                      dng02_inputs=int((base != 0).sum()), bridge_units=len(bridge.units),
                      bridge_targets=int(len(bridge.target_index)), bridge_sha256=A.bridge_key(bridge),
                      unit_sides={s: int(sum(u["side"] == s for u in bridge.units)) for s in ("left", "right", "center")},
                      permutations_sha256=C.permutations_sha256(), fw_table_from=A.record_name("rot_left", 60, 0),
                      build_s=time.time() - t0)
    if not setup.meta["graph_equals_reach"]:
        raise SystemExit("the VNC graph differs from the PR #25 benchmark graph")
    setup.meta["sha256"] = setup.content_sha256()
    A.OUT.mkdir(parents=True, exist_ok=True)
    setup.save(SETUP)
    write_json(A.OUT / "setup.json", setup.meta)
    print(json.dumps(setup.meta), flush=True)


def load_setup():
    if not SETUP.exists():
        raise SystemExit(f"{SETUP} missing: run `setup` first")
    return A.Setup.load(SETUP)


# ── gates ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def flyvis_keys():
    """Render key of every training trial (training seeds x 6 training conditions)."""
    import f5_stageA_records as R
    return {(s, st, v): R.render_key(st, v or 0, s) for s in A.splits()["train"] for st, v in A.TRAIN_CONDITIONS}


def flyvis_worker(keys, out_dir):
    """Renders keys (tasks/t0_stimuli + FlyVis + boundary layer, as scripts/f3b1_brain.py) and keeps only the window
    mean of each eye (the renders themselves are not kept)."""
    import gc

    import f5_stageA_records as R
    import torch
    from f3b1_brain import build, render_setup, render_stimulus
    torch.set_num_threads(1)
    cx, _, eyes = build(log=lambda *a, **k: None)
    del cx
    gc.collect()
    rays, retina = render_setup()
    for key in keys:
        p = Path(out_dir) / f"{key}.npy"
        if p.exists():
            continue
        cond, tf, ph = R.key_params(key)
        L, Rr = render_stimulus(eyes, cond, rays, retina, tf_hz=tf, phase_deg=ph)
        wm = np.concatenate([L[A.WINDOW[0]:A.WINDOW[1]].mean(0), Rr[A.WINDOW[0]:A.WINDOW[1]].mean(0)]).astype(np.float32)
        tmp = p.with_name(p.stem + ".tmp.npy")
        np.save(tmp, wm)
        os.replace(tmp, p)
        print("rendered", key, flush=True)


def flyvis_render(workers):
    import f5_stageA_records as R
    out = GATES / "flyvis_wm"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(flyvis_keys().values()))
    todo = [k for k in keys if not (out / f"{k}.npy").exists()]
    print(f"{len(keys)} renders, {len(todo)} to make, {workers} workers", flush=True)
    env = dict(os.environ, **R.WORKER_ENV, MUJOCO_GL="egl")
    procs = []
    for w in range(workers):
        part = todo[w::workers]
        if part:
            procs.append(subprocess.Popen([sys.executable, __file__, "flyvis-worker", "--keys", ",".join(part)],
                                          env=env, cwd=ROOT))
            time.sleep(60)                                  # staggered start (peak RAM while building)
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"flyvis workers failed: {rc}")


def gate(part, a):
    setup = load_setup() if part not in ("records", "symmetry") else None
    out = GATES / f"{part}.json"
    t0 = time.time()
    if part == "records":
        res = G.records_gate(A.RECORDS, STEP2C_JSON)
    elif part == "shuffle":
        res = G.shuffle_gate(setup, A.TRAIN_SEEDS["shuffled"], GRAPHS)
    elif part == "engine":
        res = G.engine_gradient_and_forward(G.gate_subgraph(setup, A.make_bridge(setup, "true", 0)))
    elif part == "causality":
        res = {}
        A.check_seed(0)
        counts = A.open_record(A.RECORDS / "brain" / A.record_name("rot_right", 60, 0))["dn_counts"]
        for arm in ("true", "derange", "mlp", "linear"):
            br = A.make_bridge(setup, arm, 0)
            model, _ = A.make_model(arm, 0, setup, br, GRAPHS)
            res[arm] = G.causality(model, br, counts, j=150, device=a.device)
        res["passed"] = all(v["passed"] for v in res.values())
    elif part == "resume":
        br = A.make_bridge(setup, "true", 0)
        train = A.load_trials("train", setup, br)

        def make():
            model, info = A.make_model("true", 0, setup, br, GRAPHS)
            return model, dict(gate="resume", model=info)
        res = G.resume_check(make, train, GATES / "resume_runs", steps=4, device=a.device)
    elif part == "alignment_bridge":
        train = A.load_trials("train", setup, A.make_bridge(setup, "true", 0))
        res = G.ridge_cv_r2(G.window_mean(train.units), G.window_mean(train.target[..., None])[:, 0], train.seed)
    elif part == "alignment_flyvis":
        keys = flyvis_keys()
        wm_dir = GATES / "flyvis_wm"
        missing = [k for k in set(keys.values()) if not (wm_dir / f"{k}.npy").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} FlyVis window means missing: run `gates --part flyvis_render` first")
        rows = sorted(keys)
        X = np.stack([np.load(wm_dir / f"{keys[r]}.npy") for r in rows])
        P = G.projection(X.shape[1])
        y = np.array([A.target_value(st, v) for _, st, v in rows])
        res = G.ridge_cv_r2(X @ P, y, np.array([s for s, _, _ in rows]))
        res.update(n_renders=len(set(keys.values())), projection=dict(dim=G.PROJ_DIM, seed=G.PROJ_SEED))
    elif part == "symmetry":
        res = symmetry()
    else:
        raise SystemExit(part)
    res["wall_s"] = time.time() - t0
    write_json(out, res)
    print(part, "passed" if res["passed"] else "FAILED", flush=True)


def symmetry():
    import f3a_vnc_run as F
    import f3b3_common as C
    from connectome import qa_report as Q
    from connectome.adapters import flywire783 as fw
    from f3_dn_crosswalk import per_family
    from f3a3_subgraph import vnc_subgraph
    fcx = fw.load_connectome(hash_files=False)
    fw_rows = per_family(fcx)
    del fcx
    bcx = F.load()
    banc_rows = per_family(bcx)
    sub, _ = vnc_subgraph(bcx)
    del bcx
    sym = Q.type_symmetry(sub)
    rows = [r for _, r in sym.iterrows() if r["type"] in LATERAL]
    return G.symmetry_gate(C.ENSEMBLE, fw_rows, banc_rows, rows)


def gates_summary():
    res = {}
    for p in GATE_PARTS:
        f = GATES / f"{p}.json"
        res[p] = json.loads(f.read_text())["passed"] if f.exists() else None
    out = dict(stage=A.STAGE, parts=res, passed=bool(all(v is True for v in res.values())))
    write_json(A.OUT / "gates.json", out)
    print(json.dumps(out), flush=True)


# ── runs ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selected_lr(arm):
    p = A.OUT / arm / "lr_selected.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run the three seed-0 runs and `select-lr --arm {arm}` first")
    return json.loads(p.read_text())["lr"]


def run(a):
    if a.arm not in A.ARMS:
        raise SystemExit(f"unknown arm {a.arm}")
    if a.seed not in A.TRAIN_SEEDS[a.arm]:
        raise SystemExit(f"training seed {a.seed} is not registered for {a.arm}")
    if a.replicate and (a.smoke or a.arm != "true" or a.seed != 0):
        raise SystemExit("the prereg/19 replicate is the true arm at training seed 0 (not a smoke run)")
    if a.smoke:
        if a.arm == "untrained":
            raise SystemExit("no smoke run for untrained")
        lr = a.lr if a.lr is not None else A.LR_GRID[0]
        out_root = A.SMOKE_OUT
    else:
        g = A.OUT / "gates.json"
        if not g.exists() or not json.loads(g.read_text())["passed"]:
            raise SystemExit("the stage-A gates have not all passed (runs/f5/A/gates.json); no training run")
        if a.steps != A.STEPS or a.eval_every != A.EVAL_EVERY or a.ckpt_every != A.CKPT_EVERY:
            raise SystemExit("--steps / --eval-every / --ckpt-every differ from prereg/14 only in a smoke run")
        if a.arm == "untrained":
            lr = None
        elif a.replicate:
            lr = selected_lr(a.arm)
        elif a.seed == 0:
            if a.lr is None or a.lr not in A.LR_GRID:
                raise SystemExit(f"seed 0 runs the learning-rate grid: --lr one of {A.LR_GRID}")
            lr = a.lr
        else:
            lr = selected_lr(a.arm)
            if a.lr is not None and a.lr != lr:
                raise SystemExit(f"seeds 1-9 use the selected rate {lr:g}")
        out_root = A.OUT
    run_dir = A.run_dir_of(a.arm, a.seed, lr, out_root, replicate=a.replicate)
    if (run_dir / "DONE").exists():
        if not a.overwrite:
            print(f"{run_dir} is finished (DONE); nothing to do (rule 13: --overwrite to redo)", flush=True)
            return 0
        if (run_dir / "test.json").exists() or (run_dir / "test.lock").exists():
            raise SystemExit(f"{run_dir} has a test evaluation; it is never overwritten")
        shutil.rmtree(run_dir)
    setup = load_setup()
    bridge = A.make_bridge(setup, a.arm, a.seed)
    t0 = time.time()
    train = A.load_trials("train", setup, bridge)
    val = None if a.smoke else A.load_trials("val", setup, bridge)
    load_s = time.time() - t0
    model, info = A.make_model(a.arm, a.seed, setup, bridge, GRAPHS)
    cfg = A.RunConfig(arm=a.arm, seed=a.seed, lr=lr if lr is not None else 0.0, run_dir=run_dir, steps=a.steps,
                      eval_every=a.eval_every, ckpt_every=a.ckpt_every, device=a.device, smoke=a.smoke,
                      replicate=a.replicate)
    rec = A.run_record(cfg, setup, info, bridge, train, val)
    if a.smoke:
        rec["smoke_note"] = ("SMOKE RUN: timing and memory only; evaluated on training trials; never used for any "
                             "decision, selection, test or report")
    print(f"{run_dir}: {len(train)} training trials, {0 if val is None else len(val)} validation trials, loaded in "
          f"{load_s:.1f} s; {model.n_trainable():,} trainable parameters", flush=True)
    copy_to = a.copy_to if a.copy_to is not None else (PREVIEW if PREVIEW.parent.exists() else None)
    done = A.train_run(cfg, model, rec, train, val, log_fn=lambda m: print(m, flush=True),
                       on_eval=lambda: curves_of(run_dir, copy_to))
    done["data_load_s"] = load_s
    write_json(run_dir / "DONE", done)
    curves_of(run_dir, copy_to)
    print(json.dumps(done), flush=True)
    return 0


def select_lr(a):
    res = A.select_lr(a.arm)
    p = A.OUT / a.arm / "lr_selected.json"
    if p.exists() and json.loads(p.read_text())["lr"] != res["lr"]:
        raise SystemExit(f"{p} exists with another rate; it is never changed")
    write_json(p, res)
    print(json.dumps(res), flush=True)


# ── final evaluation (the only reader of the test seeds) ────────────────────────────────────────────────────────────
def final_eval(a):
    lr = None if a.arm == "untrained" else selected_lr(a.arm)
    run_dir = A.run_dir_of(a.arm, a.seed, lr, A.OUT)
    setup = load_setup()
    bridge = A.make_bridge(setup, a.arm, a.seed)
    model, _ = A.make_model(a.arm, a.seed, setup, bridge, GRAPHS)
    print(json.dumps(A.final_evaluate(run_dir, setup, bridge, model)), flush=True)


# ── learning curves ─────────────────────────────────────────────────────────────────────────────────────────────────
def curves_of(run_dir, copy_to=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    log = A.JsonLog(run_dir / "log.jsonl").read()
    cfg = json.loads((run_dir / "config.json").read_text())
    steps = [r for r in log if r.get("event") == "step"]
    evals = [r for r in log if r.get("event") == "eval"]
    if not evals:
        return None
    smoke = cfg.get("smoke", False)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    if steps:
        x = [r["trials_seen"] for r in steps]
        ax[0].plot(x, [r["loss"] for r in steps], lw=0.6, color="0.6", label="training batch")
    ax[0].plot([e["trials_seen"] for e in evals], [e["loss"] for e in evals], "o-", ms=3, label=evals[0]["set"])
    ax[0].set_ylabel("loss ((dA - dA*) / A_MAX)^2")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    for k, (key, lab) in enumerate((("sign_accuracy", "turn-sign accuracy"), ("r2", "R2 (window mean)"))):
        ax[k + 1].plot([e["trials_seen"] for e in evals], [e[key] for e in evals], "o-", ms=3)
        ax[k + 1].set_ylabel(lab)
    ax[1].axhline(A.SET_ACCURACY, color="r", lw=0.6, ls="--")
    if steps:
        ax[3].plot([r["trials_seen"] for r in steps], [r["step_s"] for r in steps], lw=0.6)
    ax[3].set_ylabel("step wall time (s)")
    for x_ in ax:
        x_.set_xlabel("training trials presented")
    c = cfg["config"]
    title = f"F5 stage A: {c['arm']}, training seed {c['seed']}, lr {c['lr']:g}"
    if smoke:
        title = "SMOKE RUN (timing only; evaluated on training trials; not a result) - " + title
    fig.suptitle(title, color="r" if smoke else "k")
    fig.tight_layout()
    png = run_dir / "curves.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    if copy_to:
        copy_preview(png, Path(copy_to) / f"{'SMOKE_' if smoke else ''}f5A_{c['arm']}_{run_dir.name}.png")
    return png


def copy_preview(png, dst, tries=PREVIEW_TRIES, wait_s=PREVIEW_WAIT_S):
    """Best-effort copy of a learning-curve PNG to the viewing folder. On the Windows mount (drvfs) a write fails with
    EINVAL while Windows holds the file open (e.g. an image viewer showing it), so a failed copy is retried and then
    skipped with a warning: the preview never stops a run (the PNG in the run directory is always written)."""
    dst = Path(dst)
    for k in range(tries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
            shutil.copyfile(png, tmp)
            os.replace(tmp, dst)
            return True
        except OSError as e:
            err = e
            if k + 1 < tries:
                time.sleep(wait_s)
    print(f"warning: preview copy to {dst} skipped after {tries} tries ({err})", flush=True)
    return False


def curves(a):
    root = A.SMOKE_OUT if a.smoke else A.OUT
    n = 0
    for log in sorted(root.glob("*/*/log.jsonl")):
        if curves_of(log.parent, a.copy_to):
            n += 1
    print(f"{n} learning-curve PNGs", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    g = sub.add_parser("gates")
    g.add_argument("--part", required=True, choices=GATE_PARTS + ("flyvis_render", "summary"))
    g.add_argument("--device", default="cpu")
    g.add_argument("--workers", type=int, default=FLYVIS_WORKERS)
    w = sub.add_parser("flyvis-worker")
    w.add_argument("--keys", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--steps", type=int, default=A.STEPS)
    r.add_argument("--eval-every", type=int, default=A.EVAL_EVERY)
    r.add_argument("--ckpt-every", type=int, default=A.CKPT_EVERY)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--replicate", action="store_true", help="prereg/19: second run of true / seed 0 / selected rate")
    r.add_argument("--copy-to", default=None, help=f"copy the learning-curve PNG here (default {PREVIEW} if present)")
    s = sub.add_parser("select-lr")
    s.add_argument("--arm", required=True)
    f = sub.add_parser("final-eval")
    f.add_argument("--arm", required=True)
    f.add_argument("--seed", type=int, required=True)
    c = sub.add_parser("curves")
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--copy-to", default=str(PREVIEW) if PREVIEW.parent.exists() else None)
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        build_setup()
    elif a.cmd == "gates":
        if a.part == "flyvis_render":
            flyvis_render(a.workers)
        elif a.part == "summary":
            gates_summary()
        else:
            gate(a.part, a)
    elif a.cmd == "flyvis-worker":
        flyvis_worker(a.keys.split(","), GATES / "flyvis_wm")
    elif a.cmd == "run":
        return run(a)
    elif a.cmd == "select-lr":
        select_lr(a)
    elif a.cmd == "final-eval":
        final_eval(a)
    elif a.cmd == "curves":
        curves(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
