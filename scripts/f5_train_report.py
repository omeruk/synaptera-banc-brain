"""F5 step 3: report of the stage-A training runner (design numbers, smoke-run timing, pause / resume, determinism).

    env -u PYTHONPATH python scripts/f5_train_report.py          # writes docs/results/f5_stageA_runner.{json,md}

Inputs (not committed): runs/f5/A/setup.{npz,json} (scripts/f5_train.py setup), the smoke runs in
runs/f5/A_smoke/true/ (`t0_lr0.001_uninterrupted`: 20 steps without a pause; `t0_lr0.001`: the same run paused by
runs/f5/PAUSE and resumed by scripts/f5_train_guard.py), runs/f5/A_smoke/spmm_determinism.json
(scripts/diag/f5_spmm_determinism.py) and runs/f5/bench/a_gpu.json (PR #25: the mlp step). The smoke runs are timing
and memory only: their evaluations used training trials, and no metric of theirs is reported or used. Refuses to
overwrite without --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train import f5_stageA as A  # noqa: E402

SMOKE = A.SMOKE_OUT / "true"
RUN_FULL, RUN_PAUSED = SMOKE / "t0_lr0.001_uninterrupted", SMOKE / "t0_lr0.001"
DET = A.SMOKE_OUT / "spmm_determinism.json"
BENCH_A = ROOT / "runs" / "f5" / "bench" / "a_gpu.json"
OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_runner.json"
OUT_MD = ROOT / "docs" / "results" / "f5_stageA_runner.md"
DAY = 86400.0
# registered run counts of stage A (prereg/14 sections 5-6): seed 0 of a trained arm runs the 3-rate grid
GRAPH_RUNS = dict(true=12, shuffled=12, derange=12, bridge_off=7)
ANN_RUNS = dict(mlp=12, linear=12)
N_VAL, N_TEST = 240, 400


def _log(run):
    return A.JsonLog(run / "log.jsonl").read()


def _params(run, step):
    import torch
    return torch.load(run / f"step_{step:08d}.pt", weights_only=False, map_location="cpu")["model"]


def build():
    import torch
    setup = A.Setup.load(A.OUT / "setup.npz")
    meta = json.loads((A.OUT / "setup.json").read_text())
    nf = meta["bridge_units"] * len(A.FILTER_MS)
    h = A.mlp_width(nf, meta["n_trainable"])
    derange_units = {f"t{t}": len(A.make_bridge(setup, "derange", t).units) for t in A.TRAIN_SEEDS["derange"]}
    design = dict(graph_sha256=meta["graph_sha256"], graph_equals_pr25=meta["graph_equals_reach"],
                  n_neurons=meta["n_neurons"], n_edges=meta["n_edges"], n_inputs=meta["n_inputs"],
                  n_trainable_graph=meta["n_trainable"], dng02_inputs=meta["dng02_inputs"],
                  bridge_units=meta["bridge_units"], bridge_unit_sides=meta["unit_sides"],
                  bridge_targets=meta["bridge_targets"], derange_units=derange_units,
                  mlp_features=nf, mlp_width=h, mlp_params=A.mlp_param_count(nf, h), linear_params=nf * 16 + 16,
                  n_train_trials=len(A.splits()["train"]) * len(A.TRAIN_CONDITIONS),
                  n_val_trials=len(A.splits()["val"]) * len(A.TRAIN_CONDITIONS),
                  batch=A.BATCH, steps=A.STEPS, lr_grid=list(A.LR_GRID), eval_every=A.EVAL_EVERY,
                  ckpt_every=A.CKPT_EVERY, edge_grad=A.EDGE_GRAD, checkpoint_every=A.CHECKPOINT_EVERY,
                  split_sha256=A.split_sha256())
    done = json.loads((RUN_FULL / "DONE").read_text())
    cfg = json.loads((RUN_FULL / "config.json").read_text())
    lf = _log(RUN_FULL)
    ev = [r for r in lf if r["event"] == "eval"]
    st = [r for r in lf if r["event"] == "step"]
    smoke = dict(run=str(RUN_FULL.relative_to(ROOT)), smoke=cfg["smoke"], arm=cfg["config"]["arm"],
                 training_seed=cfg["config"]["seed"], lr=cfg["config"]["lr"], steps=done["steps"],
                 device=cfg["config"]["device"], eval_set=done["eval_set"], data_load_s=done["data_load_s"],
                 median_step_s=done["median_step_s"], first_step_s=st[0]["step_s"], max_step_s=max(r["step_s"] for r in st),
                 eval_s_240=float(np.median([e["eval_s"] for e in ev])), n_evals=len(ev),
                 peak_rss_gib=done["peak_rss_gib"], gpu_peak_allocated_gib=done["gpu_peak_allocated_gib"],
                 gpu_peak_reserved_gib=done["gpu_peak_reserved_gib"])
    lp = _log(RUN_PAUSED)
    resume = [r for r in lp if r["event"] == "resume"]
    pause = dict(run=str(RUN_PAUSED.relative_to(ROOT)), resumes=[dict(step=r["step"], checkpoint=r["from_checkpoint"])
                                                                    for r in resume],
                 steps_logged_contiguous=[r["step"] for r in lp if r["event"] == "step"] == list(range(1, done["steps"] + 1)),
                 median_step_s=json.loads((RUN_PAUSED / "DONE").read_text())["median_step_s"], diff_vs_uninterrupted={})
    for step in (10, 20):
        a, b = _params(RUN_FULL, step), _params(RUN_PAUSED, step)
        pause["diff_vs_uninterrupted"][f"step_{step}"] = {
            k: dict(bit_identical=bool(torch.equal(a[k], b[k])), max_abs_diff=float((a[k] - b[k]).abs().max()),
                    max_abs_param=float(a[k].abs().max())) for k in ("net.theta", "net.phi")}
    det = json.loads(DET.read_text())
    determinism = {dev: {k: v for k, v in det[dev]["products"].items()} for dev in ("cpu", "cuda")}
    mlp_step = json.loads(BENCH_A.read_text())["mlp"]["median_step_s"]
    per_run = A.STEPS * smoke["median_step_s"] + (A.STEPS // A.EVAL_EVERY + 1) * smoke["eval_s_240"]
    test_s = N_TEST / N_VAL * smoke["eval_s_240"]
    graph_s = sum(GRAPH_RUNS.values()) * (per_run + test_s) + smoke["eval_s_240"] + test_s      # + untrained
    ann_s = sum(ANN_RUNS.values()) * A.STEPS * mlp_step
    projection = dict(note="informational; prereg/18 section 4 recomputes the section-10 projection before the first "
                           "stage-A training run", graph_runs=GRAPH_RUNS, ann_runs=ANN_RUNS,
                      graph_run_s=per_run, mlp_step_s_pr25=mlp_step, days=(graph_s + ann_s) / DAY)
    return dict(step="F5 step 3", design=design, smoke=smoke, pause_resume=pause, determinism=determinism,
                projection_stage_A=projection)


def summary(d):
    s, p = d["smoke"], d["projection_stage_A"]
    return (f"Smoke run (true arm, training seed 0, {s['steps']} steps on the GPU, training trials only, not a result): "
            f"{s['median_step_s']:.3f} s per step (median), {s['eval_s_240']:.2f} s per 240-trial evaluation, "
            f"{s['gpu_peak_allocated_gib']:.2f} GiB peak GPU memory, {s['peak_rss_gib']:.2f} GiB peak RAM; serial "
            f"stage-A projection {p['days']:.2f} days.")


def render(d):
    D, s, p, pr = d["design"], d["smoke"], d["pause_resume"], d["projection_stage_A"]
    det = d["determinism"]
    L = ["# F5 step 3: stage-A training runner", "",
         "Generated by `scripts/f5_train_report.py`; checked by `scripts/verify_report.py`. **No training run has "
         "started.** The smoke runs below measure time and memory only. Their evaluations used training trials, and "
         "none of their metrics is reported or used for any decision.", "",
         f"- Summary: {summary(d)}", "",
         "## 1. Design (prereg/14, prereg/17, prereg/18)", "",
         f"- VNC graph: {D['n_neurons']:,} neurons, {D['n_edges']:,} edges, {D['n_inputs']:,} clamped BANC DNs "
         f"({D['dng02_inputs']} DNg02 at 1, `HAND`); {D['n_trainable_graph']:,} trainable parameters. Identical to the "
         f"PR #25 benchmark graph: {D['graph_equals_pr25']}.",
         f"- Bridge (prereg/11 `ensemble`, unchanged): {D['bridge_units']} units ({D['bridge_unit_sides']['left']} left, "
         f"{D['bridge_unit_sides']['right']} right), {D['bridge_targets']} BANC targets. `derange` bridges: "
         f"{sorted(set(D['derange_units'].values()))} units. prereg/14 section 2 writes 224 units for the ensemble; "
         "prereg/10 and prereg/11 register 225 (221 by exact type, 4 by family; DNg18 has an extra left unit by type), "
         "and each derangement has 224.",
         f"- `mlp`: {D['mlp_features']} filtered inputs, width {D['mlp_width']:,}, {D['mlp_params']:,} parameters; "
         f"`linear`: {D['linear_params']:,} parameters.",
         f"- Data: {D['n_train_trials']} training and {D['n_val_trials']} validation trials; split file SHA-256 "
         f"`{D['split_sha256']}`.",
         f"- Training: batch {D['batch']}, {D['steps']:,} steps, learning-rate grid {D['lr_grid']}, validation every "
         f"{D['eval_every']} steps, checkpoints every {D['ckpt_every']} steps; edge gradient `{D['edge_grad']}`, "
         f"gradient checkpointing {D['checkpoint_every']} (none).", "",
         "## 2. Smoke run", "",
         "| quantity | value |", "|---|---|",
         f"| run | `{s['run']}` ({s['arm']}, training seed {s['training_seed']}, lr {s['lr']:g}, {s['device']}) |",
         f"| steps | {s['steps']} |",
         f"| data load (960 training records, hash checks, bridge) | {s['data_load_s']:.1f} s |",
         f"| step, median | {s['median_step_s']:.3f} s |",
         f"| step, first / largest | {s['first_step_s']:.3f} / {s['max_step_s']:.3f} s |",
         f"| evaluation of 240 trials, median of {s['n_evals']} | {s['eval_s_240']:.2f} s |",
         f"| peak GPU memory, allocated / reserved | {s['gpu_peak_allocated_gib']:.2f} / {s['gpu_peak_reserved_gib']:.2f} GiB |",
         f"| peak RAM of the process | {s['peak_rss_gib']:.2f} GiB |", "",
         f"- Serial stage-A projection from this step time ({pr['note']}): {sum(pr['graph_runs'].values())} graph-arm "
         f"runs of {pr['graph_run_s'] / 3600:.2f} h each, plus {sum(pr['ann_runs'].values())} `mlp` / `linear` runs at "
         f"{pr['mlp_step_s_pr25'] * 1e3:.1f} ms per step (PR #25): {pr['days']:.2f} days.", "",
         "## 3. Pause and resume", "",
         f"- `{p['run']}`: the same smoke run, paused by `runs/f5/PAUSE` through `scripts/f5_train_guard.py`. Resumed "
         + ", ".join(f"at step {r['step']} from `{r['checkpoint']}`" for r in p["resumes"])
         + f"; logged steps contiguous: {p['steps_logged_contiguous']}; median step {p['median_step_s']:.3f} s.",
         "", "| step | parameter | bit-identical to the uninterrupted run | largest difference | largest value |",
         "|---|---|---|---|---|"]
    for step, row in p["diff_vs_uninterrupted"].items():
        for k, v in row.items():
            L.append(f"| {step.split('_')[1]} | `{k}` | {v['bit_identical']} | {v['max_abs_diff']:.2e} | "
                     f"{v['max_abs_param']:.3e} |")
    L += ["", "- Step 10 precedes the pause, so its difference is the run-to-run difference of two uninterrupted GPU "
          "runs (section 4); the resume adds no difference of a larger size.", "",
          "## 4. Determinism of the sparse products (diagnostic, synthetic inputs)", "",
          "| device | product | batch | bit-identical over 5 calls | largest difference | ms per product |",
          "|---|---|---|---|---|---|"]
    for dev in ("cpu", "cuda"):
        for k, v in det[dev].items():
            name, b = k.rsplit("_b", 1)
            ms = f"{v['ms_per_product']:.2f}" if "ms_per_product" in v else "-"
            L.append(f"| {dev} | `{name}` | {b} | {v['identical']} | {v['max_diff']:.2e} | {ms} |")
    L += ["", "- `W_s`: the CSR product of the forward pass; `Wt_s`: the transposed product of the state gradient; "
          "`sddmm`: the edge gradient (prereg/18); `W_s_segment`: a deterministic alternative (not used).",
          "- With the registered products, a resumed CPU run is bit-identical to an uninterrupted one (unit tests), and "
          "GPU runs agree only to float32 rounding. Every run records this.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=1) + "\n")
    OUT_MD.write_text(render(d))
    print(summary(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
