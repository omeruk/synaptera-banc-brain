"""F5 stage A: the validation learning curves of every finished run, as a committed JSON (validation split only).

    env -u PYTHONPATH python scripts/f5_stageA_curves.py [--overwrite]      # docs/results/f5_stageA_curves.json

Reads, read-only, exactly the runs scripts/f5_stageA_validation.py reads, with its helpers and checks: selected_lrs
(runs/f5/A/<arm>/lr_selected.json), run_dirs (the registered runs at the selected rate and the untrained run), the grid
runs A.run_dir_of(arm, 0, lr) for lr in A.LR_GRID, and the prereg/19 replicate pair; per run config.json, DONE and the
"eval" events of log.jsonl. run_row stops on a smoke run or a run whose eval_set is not "validation", and evals_of on
any evaluation that is not on the validation split. No record, test seed or test.json is opened.

Per run (each run directory once): arm, training seed, learning rate, replicate flag, selected-rate flag, steps,
final_step and selected_step from DONE, batch and the step budget (config_steps) from config.json, and every
evaluation in order, step 0 included (step, training trials presented = step x batch, validation loss, R2, turn-sign
accuracy; a step evaluated twice after a resume keeps its last evaluation, as the validation report does), plus the
SHA-256 of every source file. Refuses to overwrite without --overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_validation as V  # noqa: E402
from train import f5_stageA as A  # noqa: E402

OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_curves.json"
SOURCES = ("config.json", "log.jsonl", "DONE")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_entry(arm, t, lr, run, selected_rate):
    V.run_row(run)                                   # the validation report's checks: no smoke run, validation only
    cfg, done = json.loads((run / "config.json").read_text()), json.loads((run / "DONE").read_text())
    if (done["arm"], done["training_seed"], done["lr"]) != (arm, t, lr):
        raise SystemExit(f"{run}: DONE names another run")
    batch = cfg["config"]["batch"]
    ev = {e["step"]: e for e in V.evals_of(run)}
    steps = sorted(ev)
    return dict(run=f"{arm}/{run.name}", arm=arm, training_seed=t, lr=lr, replicate=bool(done["replicate"]),
                selected_rate=selected_rate, batch=batch, steps=done["steps"], config_steps=cfg["config"]["steps"],
                final_step=done["final_step"],
                selected_step=done["selected_step"],
                evaluations=dict(step=steps, trials=[s * batch for s in steps], loss=[ev[s]["loss"] for s in steps],
                                 r2=[ev[s]["r2"] for s in steps],
                                 sign_accuracy=[ev[s]["sign_accuracy"] for s in steps]),
                sha256={f: _sha(run / f) for f in SOURCES})


def build():
    sel = V.selected_lrs()
    entries, seen = [], set()

    def add(arm, t, lr, run, selected_rate):
        if run not in seen:
            seen.add(run)
            entries.append(run_entry(arm, t, lr, run, selected_rate))
    for arm, t, run in V.run_dirs(sel):                                # registered runs + the untrained run
        lr = None if arm == "untrained" else sel[arm]["lr"]
        add(arm, t, lr, run, True)
    for arm in V.TRAINED:                                              # the seed-0 grid (selected rate already in)
        for lr in A.LR_GRID:
            add(arm, 0, lr, A.run_dir_of(arm, 0, lr, A.OUT), lr == sel[arm]["lr"])
    lr = sel["true"]["lr"]                                             # the prereg/19 replicate pair
    rep = A.run_dir_of("true", 0, lr, A.OUT, replicate=True)
    add("true", 0, lr, rep, True)
    if not entries[-1]["replicate"]:
        raise SystemExit(f"{rep} is not marked replicate")
    return dict(stage=A.STAGE, split="validation",
                note=("validation evaluations written during training (data seeds 160-199, training speeds); no "
                      "record, no test seed and no test.json was opened"),
                batch=A.BATCH, steps=A.STEPS, eval_every=A.EVAL_EVERY, set_accuracy=A.SET_ACCURACY,
                lr_grid=list(A.LR_GRID), selected_lr={a: sel[a]["lr"] for a in V.TRAINED},
                lr_selected_sha256={a: _sha(A.OUT / a / "lr_selected.json") for a in V.TRAINED},
                n_runs=len(entries), runs=entries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if OUT_JSON.exists() and not a.overwrite:
        raise SystemExit(f"{OUT_JSON} exists; pass --overwrite to replace it")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=1) + "\n")
    print(f"{OUT_JSON}: {d['n_runs']} runs, {sum(len(r['evaluations']['step']) for r in d['runs'])} evaluations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
