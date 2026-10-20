"""prereg/20: GPU-hour estimate of the exploratory stage-A follow-up from the measured stage-A step times.

    env -u PYTHONPATH python scripts/f5_prereg20_estimate.py      # writes docs/results/f5_prereg20_estimate.json

Inputs (not committed): the finished stage-A runs of `true` and `shuffled` at lr 0.01, training seeds 0-9
(runs/f5/A/<arm>/t<t>_lr0.01/{DONE,log.jsonl}; the prereg/19 replicate is not used). Only wall times are read: the
median step time, the validation evaluation time and the data load time, all on machine 2 (RTX 4070 Laptop). No
metric is read. The estimate is in machine-2 GPU hours; other hardware is faster or slower by an unmeasured factor.
Refuses to overwrite without --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train import f5_stageA as A  # noqa: E402

OUT = ROOT / "docs" / "results" / "f5_prereg20_estimate.json"
ARMS, SEEDS, MEASURED_LR = ("true", "shuffled"), range(5), 0.01
# exploratory configurations of prereg/20 section 2: (name, lr, steps)
CONFIGS = (("i_lr0.03_2000_steps", 0.03, 2000), ("ii_lr0.01_6000_steps", 0.01, 6000))


def measured():
    step, ev, load = [], [], []
    for a in ARMS:
        for t in A.TRAIN_SEEDS[a]:
            run = A.run_dir_of(a, t, MEASURED_LR, A.OUT)
            done = json.loads((run / "DONE").read_text())
            step.append(done["median_step_s"])
            load.append(done["data_load_s"])
            ev += [r["eval_s"] for r in A.JsonLog(run / "log.jsonl").read() if r.get("event") == "eval"]
    return dict(n_runs=len(step), median_step_s=float(np.median(step)), median_eval_s=float(np.median(ev)),
                median_data_load_s=float(np.median(load)))


def build():
    m = measured()
    cfgs = {}
    for name, lr, steps in CONFIGS:
        n_eval = steps // A.EVAL_EVERY + 1
        run_s = steps * m["median_step_s"] + n_eval * m["median_eval_s"] + m["median_data_load_s"]
        n_runs = len(ARMS) * len(SEEDS)
        cfgs[name] = dict(lr=lr, steps=steps, n_evaluations=n_eval, n_runs=n_runs, hours_per_run=run_s / 3600,
                          gpu_hours=n_runs * run_s / 3600)
    return dict(note=("machine-2 GPU hours (RTX 4070 Laptop) from measured stage-A wall times; no metric read; heat "
                      "pauses, the CPU test evaluation and setup are not included"),
                measured_from=dict(arms=list(ARMS), lr=MEASURED_LR, seeds="0-9", **m),
                exploratory_arms=list(ARMS), exploratory_seeds=list(SEEDS), configurations=cfgs,
                total_gpu_hours=sum(c["gpu_hours"] for c in cfgs.values()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if OUT.exists() and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    d = build()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps(d, indent=1), flush=True)


if __name__ == "__main__":
    main()
