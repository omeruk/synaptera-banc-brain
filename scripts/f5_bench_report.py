"""F5 step 2b: benchmark report and serial wall-time projection of stages A, B and C on this machine (prereg/14
section 10).

    env -u PYTHONPATH python scripts/f5_bench_report.py --capture-incident --old-driver 596.36   # once, after the freeze
    env -u PYTHONPATH python scripts/f5_bench_report.py          # writes docs/results/f5_benchmark.{json,md}

Inputs (not committed): runs/f5/bench/{a_gpu,a_cpu,b_gpu,b_cpu,flyvis}.json and the stage-B size probes
(scripts/f5_bench_run.py), runs/f5/bench/inputs/prereg17_check_run{1,2}.log (the prereg/17 brain-record check: render,
engine-build and record times) and runs/f5/bench/incident_2026-10-17.json (the driver faults of the first attempt).
The projection counts the registered runs of prereg/14 sections 5-7 and multiplies them by the measured medians; it is
serial (one job at a time), as section 10 asks and as this machine now requires. Refuses to overwrite without
--overwrite.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "runs" / "f5" / "bench"
INCIDENT = BENCH / "incident_2026-10-17.json"
LOGS = [BENCH / "inputs" / f"prereg17_check_run{i}.log" for i in (1, 2)]
OUT_JSON = ROOT / "docs" / "results" / "f5_benchmark.json"
OUT_MD = ROOT / "docs" / "results" / "f5_benchmark.md"
PARTS = ("a_gpu", "a_cpu", "b_gpu", "b_cpu", "flyvis")
EDGE_GRADS = ("chunked", "sddmm")
DAY = 86400.0

# prereg/14 sections 5-7 and 10
STEPS = dict(A=2000, B=1000, C_bc=1000, C_dagger=5 * 500)
EVAL_EVERY, N_VAL, N_TEST = 50, 240, 400
T_TRIAL, T_FLIGHT = 300, 500
DAGGER_FLIGHTS, VAL_FLIGHTS, BC_TEACHER_FLIGHTS = 5 * 160, 6 * 40, 960
LR_GRID = 3
FALLBACK_DAYS = 60
N_STAGE_A_RECORDS = 2400                     # prereg/14 section 7 + prereg/17 (all on machine 2)
N_RENDERS = 10 + 200 * 10                    # (condition, phase): seeds 0-39 phase 0 x 10; seeds 40-239 x 10
SEEDS = dict(registered=dict(true=10, shuffled=10, derange=10, bridge_off=5, mlp=10, linear=10),
             fallback_1=dict(true=10, shuffled=10, derange=7, bridge_off=5, mlp=7, linear=7),
             fallback_2=dict(true=7, shuffled=7, derange=7, bridge_off=5, mlp=7, linear=7))
GRAPH_ARMS = ("true", "shuffled", "derange", "bridge_off")


def runs_of(n_seeds):
    """The LR grid at seed 0 (3 runs; the best is the seed-0 replicate) plus seeds 1 .. n-1."""
    return LR_GRID + n_seeds - 1


def load(p):
    return json.loads((BENCH / f"{p}.json").read_text())


def capture_incident(old_driver):
    """The nvlddmkm events of the first benchmark attempt (2026-10-17, multi-process stage-A test) from the Windows
    System log, and the last lines of its log."""
    if INCIDENT.exists():
        raise SystemExit(f"{INCIDENT} exists")
    ps = ("$e=Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; "
          "StartTime=(Get-Date '2026-10-17 01:00'); EndTime=(Get-Date '2026-10-17 09:20')} -Oldest; "
          "$e.Count; $e[0].TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); $e[-1].TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); "
          "($e | ForEach-Object { $_.Properties[1].Value } | Sort-Object -Unique) -join '|'; "
          "$u=Get-WinEvent -FilterHashtable @{LogName='System'; Id=6008; StartTime=(Get-Date '2026-10-17 09:00'); "
          "EndTime=(Get-Date '2026-10-17 10:00')}; ($u | Select -First 1).Message")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         timeout=120).stdout.strip().splitlines()
    log = (BENCH / "bench.log").read_text().splitlines()
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    rec = dict(captured=datetime.now().isoformat(timespec="seconds"),
               driver_at_incident=old_driver,
               driver_at_incident_source="nvidia-smi after the reboot, before the update",
               driver_now=drv, nvlddmkm_events=int(out[0]), first_event=out[1], last_event=out[2],
               messages=out[3].split("|"), unexpected_shutdown=out[4].strip(),
               bench_log_tail=log[-6:], a_gpu_finished=next(ln for ln in log if ln.startswith("a_gpu done")),
               parallel_passed=[ln for ln in log if ln.startswith("chunked k=") and "ok=True" in ln])
    INCIDENT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def datagen_times():
    """Render (per stimulus), engine build and record times of the two prereg/17 check runs."""
    rend, rec, build = [], [], []
    for p in LOGS:
        t = p.read_text()
        rend += [int(x) for x in re.findall(r"^render \S+\s+\(.*\(([0-9]+) s\)$", t, re.M)]
        build += [int(x) for x in re.findall(r"^engine .* \(([0-9]+) s\)$", t, re.M)]
        rec += [int(x) for x in re.findall(r"^\S+ s[0-9]+: DN spikes .* \(([0-9]+) s, free", t, re.M)]
    return dict(render_s=rend, build_s=build, record_s=rec, render_mean_s=sum(rend) / len(rend),
                build_mean_s=sum(build) / len(build), record_mean_s=sum(rec) / len(rec))


def fastest(configs, eg):
    """The fastest checkpoint setting that fit in memory for an edge gradient: (key, entry)."""
    ok = {k: v for k, v in configs.items() if k.startswith(eg + "_") and "median_step_s" in v}
    k = min(ok, key=lambda k: ok[k]["median_step_s"])
    return k, ok[k]


def stage_costs(a, b, fv, eg):
    """Seconds per run of each arm type, per stage, for one edge gradient (registered: chunked)."""
    _, sa = fastest(a["configs"], eg)
    _, sb = fastest(b["configs"], eg)
    ea, eb = a["eval_s_per_trial"], b["eval_s_per_trial"]
    mlp_a = a["mlp"]["median_step_s"]
    mlp_scale = b["n_trainable"] / a["n_trainable"]            # the mlp width follows the trainable count of `true`
    n_eval = lambda steps: steps // EVAL_EVERY                 # noqa: E731
    flight_interval = fv["total_s_per_interval"] + b["forward_s_per_interval_batch1"]
    flights = DAGGER_FLIGHTS + VAL_FLIGHTS + N_TEST
    c_step = b["c_shape"][eg]["median_step_s"]
    return dict(
        A=dict(graph=STEPS["A"] * sa["median_step_s"] + (n_eval(STEPS["A"]) * N_VAL + N_TEST) * ea,
               ann=STEPS["A"] * mlp_a, untrained=N_TEST * ea, shared=0.0),
        B=dict(graph=STEPS["B"] * sb["median_step_s"] + (n_eval(STEPS["B"]) * N_VAL + N_TEST) * eb,
               ann=STEPS["B"] * mlp_a * mlp_scale, untrained=N_TEST * eb, shared=0.0),
        C=dict(graph=(STEPS["C_bc"] + STEPS["C_dagger"]) * c_step + flights * T_FLIGHT * flight_interval,
               ann=(STEPS["C_bc"] + STEPS["C_dagger"]) * mlp_a * mlp_scale * T_FLIGHT / T_TRIAL
               + flights * T_FLIGHT * fv["total_s_per_interval"],
               untrained=N_TEST * T_FLIGHT * flight_interval,
               shared=BC_TEACHER_FLIGHTS * T_FLIGHT * fv["total_s_per_interval"]),
        flight_s=T_FLIGHT * flight_interval, flights_per_run=flights)


def projection(costs):
    out = {}
    for regime, seeds in SEEDS.items():
        r = {}
        for st in ("A", "B", "C"):
            c = costs[st]
            n_graph = sum(runs_of(seeds[a]) for a in GRAPH_ARMS)
            n_ann = runs_of(seeds["mlp"]) + runs_of(seeds["linear"])
            total = n_graph * c["graph"] + n_ann * c["ann"] + c["untrained"] + c["shared"]
            r[st] = dict(graph_runs=n_graph, ann_runs=n_ann, days=total / DAY,
                         over_fallback_limit=total / DAY > FALLBACK_DAYS)
        out[regime] = r
    return out


def build():
    a, b, fv = load("a_gpu"), load("b_gpu"), load("flyvis")
    ac, bc = load("a_cpu"), load("b_cpu")
    probes = [json.loads(p.read_text()) for p in BENCH.glob("b_gpu_probe_*.json")]
    probes = {v["part"]: v for v in sorted(probes, key=lambda v: (v["batch"] * v["intervals"], v["intervals"]))}
    inc = json.loads(INCIDENT.read_text())
    dg = datagen_times()
    costs = {eg: stage_costs(a, b, fv, eg) for eg in EDGE_GRADS}
    proj = {eg: projection(costs[eg]) for eg in EDGE_GRADS}
    cap = b["machine"]["gpu_total_gib"] * b["machine"]["gpu_fraction_cap"]

    def step_row(d, key):
        c = d["configs"][key]
        return dict(step_s=c.get("median_step_s"), peak_gib=c.get("peak_reserved_gib"), oom=c.get("oom", False))

    stage_a = {k: step_row(a, k) for k in a["configs"]}
    stage_b = {k: step_row(b, k) for k in b["configs"]}
    free = b["machine"]["gpu_free_gib_at_start"]
    a_peak = a["configs"]["chunked_ckpt0"]["peak_reserved_gib"]
    rec_s = dg["record_mean_s"] * N_STAGE_A_RECORDS
    rend_s = dg["render_mean_s"] * N_RENDERS
    return dict(
        prereg="prereg/14_f5_training.md section 10: wall-time-only benchmark, data seed 9999, synthetic inputs",
        machine=b["machine"],
        policy=dict(one_gpu_job_at_a_time=True, gpu_fraction_cap=b["machine"]["gpu_fraction_cap"],
                    gpu_cap_gib=cap, ram_cap="MemoryMax=14G, no swap (systemd scope)", kill_temp_c=87),
        incident=inc,
        graphs=dict(vnc_neurons=a["n_neurons"], vnc_edges=a["n_edges"], vnc_trainable=a["n_trainable"],
                    hybrid_trainable=b["n_trainable"], bridge_units=b["bridge_units"],
                    brain_neurons=b["record"]["brain"]["n_neurons"], brain_edges=b["record"]["brain"]["n_edges"]),
        stage_a=dict(batch=a["batch"], intervals=a["intervals"], gpu=stage_a,
                     cpu={k: step_row(ac, k) for k in ac["configs"]},
                     eval_s_per_trial=dict(gpu=a["eval_s_per_trial"], cpu=ac["eval_s_per_trial"]),
                     mlp=dict(width=a["mlp"]["width"], params=a["mlp"]["n_params"], step_s=a["mlp"]["median_step_s"]),
                     deterministic_off={eg: a["deterministic_off"][eg]["median_step_s"] for eg in EDGE_GRADS},
                     checkpointing_needed=a_peak > cap,
                     side_by_side_by_memory=math.floor(free / a_peak)),
        stage_b=dict(batch=b["batch"], intervals=b["intervals"], build_s=b["build_s"], gpu=stage_b,
                     cpu={k: step_row(bc, k) for k in bc["configs"]},
                     eval_s_per_trial=dict(gpu=b["eval_s_per_trial"], cpu=bc["eval_s_per_trial"]),
                     checkpointing_needed=not any(v["step_s"] for k, v in stage_b.items() if k.endswith("_ckpt0")),
                     c_shape={eg: dict(ckpt=b["c_shape_ckpt"][eg], step_s=b["c_shape"][eg].get("median_step_s"),
                                       peak_gib=b["c_shape"][eg].get("peak_reserved_gib"))
                              for eg in EDGE_GRADS},
                     forward_s_per_interval=dict(batch1=b["forward_s_per_interval_batch1"],
                                                 batch160=b["forward_s_per_interval_batch160"]),
                     probes={k: dict(batch=v["batch"], intervals=v["intervals"],
                                     configs={c: step_row(v, c) for c in v["configs"]},
                                     peak_rss_gib=v["machine"]["peak_rss_gib"]) for k, v in probes.items()}),
        speedups={eg: dict(compile=a["compile"][eg]["speedup"], compile_identical=a["compile"][eg]["identical"],
                           compile_max_grad_diff=a["compile"][eg]["max_abs_grad_diff"],
                           cuda_graph=a["cuda_graph"][eg]["speedup"],
                           cuda_graph_identical=a["cuda_graph"][eg]["identical"],
                           cuda_graph_max_grad_diff=a["cuda_graph"][eg]["max_abs_grad_diff"]) for eg in EDGE_GRADS},
        gpu_over_cpu=dict(
            A={eg: fastest(ac["configs"], eg)[1]["median_step_s"] / fastest(a["configs"], eg)[1]["median_step_s"]
               for eg in EDGE_GRADS},
            B={eg: bc["configs"][f"{eg}_ckpt50"]["median_step_s"] / b["configs"][f"{eg}_ckpt50"]["median_step_s"]
               for eg in EDGE_GRADS}),
        flyvis=dict(scene_s_per_interval=fv["scene_s_per_interval"], flyvis_s_per_interval=fv["flyvis_s_per_interval"],
                    total_s_per_interval=fv["total_s_per_interval"], n_intervals=fv["n_intervals"]),
        datagen=dict(times=dg, n_records=N_STAGE_A_RECORDS, n_renders=N_RENDERS,
                     records_days=rec_s / DAY, renders_days=rend_s / DAY,
                     stage_a_days=(rec_s + rend_s + dg["build_mean_s"]) / DAY),
        costs_s=costs, projection=proj, fallback_limit_days=FALLBACK_DAYS)


def f(x, d=2):
    return f"{x:,.{d}f}"


def summary(o):
    p = o["projection"]["chunked"]["registered"]
    s = o["projection"]["sddmm"]["registered"]
    return (f"Serial projection on this machine with the registered chunked edge gradient: stage A {f(p['A']['days'], 1)} "
            f"days, B {f(p['B']['days'], 1)} days, C {f(p['C']['days'], 1)} days (sddmm, opt-in: "
            f"{f(s['A']['days'], 1)}, {f(s['B']['days'], 1)}, {f(s['C']['days'], 1)}); stage-A records "
            f"{f(o['datagen']['stage_a_days'], 1)} days.")


def render(o):
    m, inc, A, B = o["machine"], o["incident"], o["stage_a"], o["stage_b"]
    L = ["# F5 step 2b: GPU support and benchmark", ""]
    L.append("Generated by `scripts/f5_bench_report.py` from `runs/f5/bench/` (not committed). "
             f"{o['prereg']}; no metric was read and no registered data or training run was made. Evaluation "
             "forwards are timed at batch 240 (A) and 8 (B), so the stage-B evaluation cost is an upper bound.")
    L.append("")
    L.append(f"- Machine: {m['machine']}, {m['cpu']} ({m['cpu_threads']} threads, torch {m['torch_threads']}), "
             f"{m['gpu']} ({f(m['gpu_total_gib'])} GiB); torch {m['torch']}, CUDA {m['cuda']}, Python {m['python']}. "
             "Deterministic algorithms on in every timed step.")
    L.append(f"- Summary: {summary(o)}")
    L.append("")
    L.append("## 1. Safety policy and the first attempt")
    L.append("")
    L.append(f"- The first attempt ran k = 1, 2, ... stage-A training processes side by side on the GPU (NVIDIA driver "
             f"{inc['driver_at_incident']}). The single-process part before it (`{inc['a_gpu_finished']}`) logged no "
             f"driver event. The driver logged {inc['nvlddmkm_events']:,} nvlddmkm events "
             f"({', '.join(inc['messages'])}) from {inc['first_event']}, in the first rounds of the side-by-side test, "
             f"to {inc['last_event']}. The rounds k = 1-{len(inc['parallel_passed'])} still finished; the round with "
             f"k = {len(inc['parallel_passed']) + 1} processes hung and the machine had to be powered off. The "
             "side-by-side timings were lost; the `a_gpu` results were complete and are used here.")
    pol = o["policy"]
    L.append(f"- Policy since then (driver {inc['driver_now']}): one GPU job at a time, no side-by-side GPU processes; "
             f"each CUDA process capped at {pol['gpu_fraction_cap']} of the GPU memory ({f(pol['gpu_cap_gib'])} GiB); "
             f"every job in a {pol['ram_cap']}; pre-checks and a watchdog (`scripts/f5_bench_guard.py`) that kills "
             f"the job at a GPU temperature of {pol['kill_temp_c']} C, low free RAM or a new nvlddmkm event.")
    L.append("")
    L.append("## 2. Stage A (VNC, batch %d, %d intervals)" % (A["batch"], A["intervals"]))
    L.append("")
    g = o["graphs"]
    L.append(f"VNC {g['vnc_neurons']:,} neurons, {g['vnc_edges']:,} edges, {g['vnc_trainable']:,} trainable parameters.")
    L.append("")
    L.append("| edge gradient | checkpoint every | GPU s/step | GPU peak GiB | CPU s/step |")
    L.append("|---|---|---|---|---|")
    for k, v in A["gpu"].items():
        eg, ck = k.split("_ckpt")
        c = A["cpu"].get(k, {})
        L.append(f"| {eg} | {ck or '-'} | {f(v['step_s'], 3)} | {f(v['peak_gib'])} | "
                 f"{f(c['step_s'], 1) if c.get('step_s') else '-'} |")
    L.append("")
    L.append(f"- Evaluation forward: {f(1e3 * A['eval_s_per_trial']['gpu'], 2)} ms per trial on the GPU, "
             f"{f(1e3 * A['eval_s_per_trial']['cpu'], 2)} ms on the CPU (batch 240).")
    L.append(f"- GPU over CPU (the fastest setting on each): chunked {f(o['gpu_over_cpu']['A']['chunked'], 1)}x, sddmm "
             f"{f(o['gpu_over_cpu']['A']['sddmm'], 1)}x.")
    L.append(f"- Gradient checkpointing needed: {'yes' if A['checkpointing_needed'] else 'no'} (peak without it below "
             f"the {f(pol['gpu_cap_gib'])} GiB cap).")
    L.append(f"- Side by side: by memory alone {A['side_by_side_by_memory']} stage-A runs would fit, but side-by-side GPU "
             "processes caused the driver faults of section 1, so stage-A runs go one at a time.")
    L.append(f"- `mlp` arm (width {A['mlp']['width']:,}, {A['mlp']['params']:,} parameters): "
             f"{f(1e3 * A['mlp']['step_s'], 1)} ms per step.")
    L.append(f"- Deterministic algorithms off (for information only; not used): chunked "
             f"{f(A['deterministic_off']['chunked'], 3)} s, sddmm {f(A['deterministic_off']['sddmm'], 3)} s per step.")
    L.append("")
    L.append("## 3. torch.compile and CUDA graphs")
    L.append("")
    L.append("| edge gradient | torch.compile speed-up | identical | max abs grad diff | CUDA graph speed-up | identical "
             "| max abs grad diff |")
    L.append("|---|---|---|---|---|---|---|")
    for eg, s in o["speedups"].items():
        L.append(f"| {eg} | {f(s['compile'])} | {s['compile_identical']} | {s['compile_max_grad_diff']:.1e} | "
                 f"{f(s['cuda_graph'])} | {s['cuda_graph_identical']} | {s['cuda_graph_max_grad_diff']:.1e} |")
    L.append("")
    L.append("Neither gives results identical to the eager step, so neither is used.")
    L.append("")
    L.append("## 4. Stage B (brain + bridge + VNC, batch %d, %d intervals)" % (B["batch"], B["intervals"]))
    L.append("")
    L.append(f"FlyWire brain {g['brain_neurons']:,} neurons, {g['brain_edges']:,} edges; `ensemble` bridge "
             f"{g['bridge_units']} units; {g['hybrid_trainable']:,} trainable parameters; graph build "
             f"{f(B['build_s'], 0)} s. Synthetic boundary-layer drive.")
    L.append("")
    L.append("Size probes first (chunked):")
    L.append("")
    L.append("| batch | intervals | checkpoint every | GPU s/step | GPU peak GiB | peak RSS GiB |")
    L.append("|---|---|---|---|---|---|")
    for v in B["probes"].values():
        for k, c in v["configs"].items():
            L.append(f"| {v['batch']} | {v['intervals']} | {k.split('_ckpt')[1]} | {f(c['step_s'], 3)} | "
                     f"{f(c['peak_gib'])} | {f(v['peak_rss_gib'])} |")
    L.append("")
    L.append("Registered size:")
    L.append("")
    L.append("| edge gradient | checkpoint every | GPU s/step | GPU peak GiB | CPU s/step |")
    L.append("|---|---|---|---|---|")
    for k, v in B["gpu"].items():
        eg, ck = k.split("_ckpt")
        c = B["cpu"].get(k, {})
        L.append(f"| {eg} | {ck} | {f(v['step_s'], 2) if v['step_s'] else 'out of memory'} | "
                 f"{f(v['peak_gib']) if v['peak_gib'] else '-'} | {f(c['step_s'], 1) if c.get('step_s') else '-'} |")
    L.append("")
    c0, c50 = B["gpu"]["chunked_ckpt0"], B["gpu"]["chunked_ckpt50"]
    L.append(f"- Gradient checkpointing needed: {'yes' if B['checkpointing_needed'] else 'no'}. Without it the chunked "
             f"step peaks at {f(c0['peak_gib'])} GiB of the {f(pol['gpu_cap_gib'])} GiB cap; checkpointing every 50 "
             f"intervals lowers this to {f(c50['peak_gib'])} GiB for {f(100 * (c50['step_s'] / c0['step_s'] - 1), 0)} % "
             "more time per step. The stage-C step (500 intervals) needs it.")
    L.append(f"- GPU over CPU (checkpoint every 50): chunked {f(o['gpu_over_cpu']['B']['chunked'], 1)}x, sddmm "
             f"{f(o['gpu_over_cpu']['B']['sddmm'], 1)}x.")
    L.append(f"- Evaluation forward: {f(1e3 * B['eval_s_per_trial']['gpu'], 1)} ms per trial on the GPU, "
             f"{f(1e3 * B['eval_s_per_trial']['cpu'], 1)} ms on the CPU (batch 8).")
    L.append(f"- Stage-C step (batch 8, one 500-interval flight): " + "; ".join(
        f"{eg} {f(c['step_s'], 2)} s (checkpoint every {c['ckpt']}, peak {f(c['peak_gib'])} GiB)"
        for eg, c in B["c_shape"].items()) + ".")
    L.append(f"- Closed-loop forward per interval: {f(1e3 * B['forward_s_per_interval']['batch1'], 2)} ms at batch 1, "
             f"{f(1e3 * B['forward_s_per_interval']['batch160'], 2)} ms at batch 160.")
    L.append("")
    fv = o["flyvis"]
    L.append("## 5. Online vision (stage C) and data generation (CPU)")
    L.append("")
    L.append(f"- Scene + retina encoding {f(1e3 * fv['scene_s_per_interval'], 1)} ms, FlyVis + boundary "
             f"{f(1e3 * fv['flyvis_s_per_interval'], 1)} ms, together {f(1e3 * fv['total_s_per_interval'], 1)} ms per "
             f"interval (median of {fv['n_intervals']}; FlyVis on the CPU, as pinned).")
    dg = o["datagen"]
    t = dg["times"]
    L.append(f"- Stage-A brain records (prereg/17 check logs): render {f(t['render_mean_s'], 1)} s per stimulus, "
             f"engine build {f(t['build_mean_s'], 0)} s per process, {f(t['record_mean_s'], 1)} s per record. "
             f"{dg['n_records']:,} records and {dg['n_renders']:,} renders: {f(dg['records_days'], 2)} + "
             f"{f(dg['renders_days'], 2)} days serial = {f(dg['stage_a_days'], 2)} days.")
    L.append(f"- The stage-B inputs are the same {dg['n_renders']:,} FlyVis sequences; kept from the record generation "
             f"they cost nothing more, rendered again {f(dg['renders_days'], 2)} days.")
    L.append("")
    L.append("## 6. Serial wall-time projection (prereg/14 section 10)")
    L.append("")
    L.append("Runs per stage: per arm the 3-rate grid at seed 0 plus the remaining seeds (`true`, `shuffled`, "
             "`derange`, `mlp`, `linear` 12 runs each at 10 seeds, 9 at 7; `bridge_off` 7), plus one `untrained` "
             "test evaluation. A and B: steps, validation every 50 steps on 240 trials, and the 400-trial test. "
             "C: BC and DAgger steps at the stage-C step time, and per run "
             f"{o['costs_s']['chunked']['flights_per_run']:,} learner flights (DAgger, validation, test) at "
             f"{f(o['costs_s']['chunked']['flight_s'], 1)} s per flight (scene + FlyVis + batch-1 forward); the 960 BC "
             "teacher flights once. The `mlp` stage-B/C step is the stage-A one scaled by the trainable count; "
             "`linear` is costed as `mlp`.")
    L.append("")
    L.append("| edge gradient | seeds | stage A days | stage B days | stage C days | over 60 days |")
    L.append("|---|---|---|---|---|---|")
    for eg in EDGE_GRADS:
        for reg, r in o["projection"][eg].items():
            over = [s for s in ("A", "B", "C") if r[s]["over_fallback_limit"]]
            L.append(f"| {eg} | {reg.replace('_', ' ')} | {f(r['A']['days'], 1)} | {f(r['B']['days'], 1)} | "
                     f"{f(r['C']['days'], 1)} | {', '.join(over) if over else 'none'} |")
    L.append("")
    L.append("- `chunked` is the registered implementation. `sddmm` gives the same gradients within float32 tolerance "
             "but not bit for bit; using it needs an addendum before any training run (the owner's decision).")
    L.append("- The projection decides the fallback only before each stage's first training run (section 10).")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--capture-incident", action="store_true")
    ap.add_argument("--old-driver")
    a = ap.parse_args(argv)
    if a.capture_incident:
        capture_incident(a.old_driver)
        return 0
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    o = build()
    OUT_JSON.write_text(json.dumps(o, indent=1) + "\n")
    OUT_MD.write_text(render(o))
    print(summary(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
