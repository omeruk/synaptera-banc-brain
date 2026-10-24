"""F5 stage A: pre-flight before the test evaluation. Reads no test data and opens no record.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_stageA_preflight.py      # exit 1 if any run fails

For every registered stage-A run (the list of `scripts/f5_stageA_report.py`): DONE is a finished, non-smoke,
non-replicate validation run; config.json carries the current split hash; best.pt exists and is the selected
checkpoint; no test evaluation has run (test.json, test.lock); and the arm's model builds as in `final-eval` and
loads best.pt on the CPU (strict state dict), with one forward pass on a zero input. While it runs, the record
readers of train/f5_stageA.py raise, so a test (or any) record cannot be opened. Prints one line per run.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_report as R  # noqa: E402
from train import f5_stageA as A  # noqa: E402

FORWARD_T = 3                        # intervals of the zero-input forward pass


@contextlib.contextmanager
def no_data():
    """Every record reader of train/f5_stageA.py raises inside this block."""
    def refuse(*a, **k):
        raise A.SealedSplitError("the pre-flight reads no record")
    saved = A.open_record, A.load_trials
    A.open_record = A.load_trials = refuse
    try:
        yield
    finally:
        A.open_record, A.load_trials = saved


def check_run(arm, t, lr, run, setup, graphs, split_path=None) -> dict:
    """Problems of one run (empty list: ready for final-eval)."""
    run = Path(run)
    bad = []
    if (run / "test.json").exists() or (run / "test.lock").exists():
        bad.append("the test evaluation has already run (test.json / test.lock)")
    for f in ("DONE", "config.json", "best.pt"):
        if not (run / f).exists():
            bad.append(f"{f} missing")
    if bad:
        return dict(arm=arm, seed=t, run=str(run), ok=False, problems=bad)
    done, cfg = json.loads((run / "DONE").read_text()), json.loads((run / "config.json").read_text())
    if done.get("smoke") or cfg.get("smoke") or done["eval_set"] != "validation":
        bad.append("not a validation run")
    if done.get("replicate"):
        bad.append("the prereg/19 replicate")
    if (done["arm"], done["training_seed"], done["lr"]) != (arm, t, lr):
        bad.append(f"DONE is {done['arm']} / {done['training_seed']} / {done['lr']}")
    if done["final_step"] != done["steps"]:
        bad.append(f"final step {done['final_step']} != {done['steps']}")
    if cfg["split_sha256"] != A.split_sha256(split_path or A.SPLIT_FILE):
        bad.append("config.json has another split hash")
    ck = torch.load(run / "best.pt", weights_only=False, map_location="cpu")
    if ck["step"] != done["selected_step"]:
        bad.append(f"best.pt is step {ck['step']}, DONE selected step {done['selected_step']}")
    try:
        bridge = A.make_bridge(setup, arm, t)
        model, _ = A.make_model(arm, t, setup, bridge, graphs)
        model.load_state_dict(ck["model"])
        model = model.to("cpu").eval()
        with torch.no_grad():
            out = model(torch.zeros(FORWARD_T, 1, model.n_units))
        if tuple(out.shape) != (FORWARD_T, 1) or not bool(torch.isfinite(out).all()):
            bad.append(f"forward pass on zeros: shape {tuple(out.shape)}, finite {bool(torch.isfinite(out).all())}")
    except Exception as e:                                    # reported per run; the other runs are still checked
        bad.append(f"model: {type(e).__name__}: {e}")
    return dict(arm=arm, seed=t, run=str(run), ok=not bad, problems=bad, selected_step=done["selected_step"],
                n_params=int(sum(np.prod(v.shape) for v in ck["model"].values())))


def preflight(setup, graphs, runs=None, registered=None, split_path=None, log=print) -> list:
    reg = registered if registered is not None else R.registered(runs)
    out = []
    with no_data():
        for arm, t, lr, run in reg:
            r = check_run(arm, t, lr, run, setup, graphs, split_path)
            log(f"{'ok  ' if r['ok'] else 'FAIL'} {arm} t{t}: " + ("; ".join(r["problems"]) or
                                                                    f"best.pt step {r['selected_step']}"))
            out.append(r)
    return out


def main(argv=None):
    import f5_train as T
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    res = preflight(T.load_setup(), T.GRAPHS)
    n_bad = sum(not r["ok"] for r in res)
    print(f"{len(res)} registered runs, {len(res) - n_bad} ready, {n_bad} failed", flush=True)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
