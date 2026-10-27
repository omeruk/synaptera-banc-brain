"""F5 stage A: post-hoc, exploratory analyses (prereg/21). Never part of a stage-A verdict.

Part 1 holds the reader only: the validation data seeds 160-199 at the held-out drum speeds (30 and 90 deg/s), which
``train.f5_stageA.load_trials`` refuses outside the final evaluation. This reader does not use ``final_evaluation()``
and refuses to run inside it. It accepts only those seeds, the stimuli ``rot_left`` / ``rot_right`` and those speeds;
every request is checked in full before any file is opened, and a test seed (200-239) raises ``SealedSplitError``.
Each record is checked against ``runs/f5/stageA/manifest.sha256``, whose own SHA-256 is pinned (prereg/21 section 1).
The 60 and 120 deg/s validation trials are read through ``load_trials`` itself.

Part 2, analysis A (prereg/21 sections 2 and 3), in two commands:

- ``predict`` (one background job; CPU, deterministic, ``A.TEST_THREADS`` threads): first the window of every run at
  the training speeds on the validation seeds and the consistency check against the logged validation evaluation of
  the selected step; only if every run passes, the held-out-speed validation trials of every run. The dA of every
  trial is saved under ``runs/f5/A_posthoc/pred/``; an existing prediction file is never rewritten (a run whose files
  exist is skipped after its ``best.pt`` hash is checked). Each run's start, end and result go to the job log.
- ``report-A``: section A of ``docs/results/f5_stageA_posthoc.{json,md}`` from the saved predictions (replication,
  the one primary post-hoc test, speed tuning). Refuses to overwrite either file without ``--overwrite``.

Part 3, analysis B (prereg/21 section 4) and the figure (section 5):

- ``report``: sections A and B of ``docs/results/f5_stageA_posthoc.{json,md}``; B is the structure measures S1-S18 of
  the `true` VNC graph and the 10 cached `shuffled` graphs (no training, no record), the rank of `true` and the
  Spearman correlations of the shuffled-graph values with the deficit vectors D1-D3. ``--overwrite`` to replace.
- ``figure``: panels a-d from the committed JSON only, to ``runs/f5/A_posthoc/`` and copied (never overwriting) to
  the Windows preview folder ``posthoc/``, every file prefixed ``posthoc_``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import rate_torch as RT  # noqa: E402
from train import f5_stageA as A  # noqa: E402

LABEL = "post-hoc, exploratory"
PLAN = ROOT / "prereg" / "21_f5_stageA_posthoc_plan.md"
SEEDS = tuple(range(160, 200))                       # the validation data seeds of configs/f5_splits.json
SPEEDS = (A.HELD_OUT["secondary"], A.HELD_OUT["primary"])
STIMULI = ("rot_left", "rot_right")
GROUPS = {k: A.TEST_GROUPS[k] for k in ("held_out_primary_90", "held_out_secondary_30")}
MANIFEST_SHA256 = "33ddb2423ba40f9ed3a189f5afb979981ad0606a8cae06935c9386bc78c26476"   # prereg/21 section 1


class PosthocRefusal(ValueError):
    """A request outside the post-hoc reader's seeds, stimuli or speeds."""


def check_split(split_path=A.SPLIT_FILE) -> None:
    """The split file must still list exactly SEEDS as its validation seeds."""
    if tuple(A.splits(split_path)["val"]) != SEEDS:
        raise PosthocRefusal(f"{split_path}: the validation seeds are not 160-199")


def check_request(stimulus, speed, seed, split_path=A.SPLIT_FILE) -> None:
    """Refuses everything but a validation seed 160-199, rot_left / rot_right, 30 or 90 deg/s."""
    if A.in_final_evaluation():
        raise A.SealedSplitError("the post-hoc reader never runs inside final_evaluation()")
    seed = int(seed)
    if seed in A.splits(split_path)["test"]:
        raise A.SealedSplitError(f"data seed {seed} is a sealed test seed; the post-hoc reader never reads it")
    if seed not in SEEDS or seed not in A.splits(split_path)["val"]:
        raise PosthocRefusal(f"data seed {seed} is not a validation seed 160-199")
    if stimulus not in STIMULI:
        raise PosthocRefusal(f"stimulus {stimulus!r}: only {STIMULI} are read")
    if isinstance(speed, bool) or not isinstance(speed, (int, np.integer)) or int(speed) not in SPEEDS:
        raise PosthocRefusal(f"speed {speed!r}: only {SPEEDS} deg/s are read")


def request_specs(conditions, seeds=SEEDS, split_path=A.SPLIT_FILE) -> list[dict]:
    """Every (seed, stimulus, speed) of the request, all checked before any file is opened."""
    check_split(split_path)
    seeds = [int(s) for s in seeds]
    conditions = [(st, v) for st, v in conditions]
    if len(set(seeds)) != len(seeds) or len(set(conditions)) != len(conditions):
        raise PosthocRefusal("duplicate seed or condition in the request")
    if not seeds or not conditions:
        raise PosthocRefusal("empty request")
    specs = [dict(stimulus=st, speed=v, seed=s) for s in seeds for st, v in conditions]
    for sp in specs:
        check_request(sp["stimulus"], sp["speed"], sp["seed"], split_path)
    for sp in specs:
        sp["speed"] = int(sp["speed"])
        sp["file"] = f"brain/{A.record_name(sp['stimulus'], sp['speed'], sp['seed'])}"
    return specs


def read_heldout(setup, bridge, conditions, seeds=SEEDS, records=A.RECORDS, split_path=A.SPLIT_FILE, manifest=None,
                 manifest_sha256=MANIFEST_SHA256) -> A.Trials:
    """The validation trials of ``conditions`` (held-out speeds only) through one bridge, as ``A.load_trials`` builds
    them: unit rate / R_IN per interval, the teacher dA*, and the SHA-256 of each record (checked against the manifest).
    """
    specs = request_specs(conditions, seeds, split_path)
    records = Path(records)
    man_path = Path(manifest or records / "manifest.sha256")
    if manifest_sha256 is not None and A.sha256_file(man_path) != manifest_sha256:
        raise ValueError(f"{man_path}: SHA-256 differs from the pinned manifest (prereg/21 section 1)")
    man = A.read_manifest(man_path)
    U = 0 if bridge is None else len(bridge.units)
    units = np.zeros((A.T_TRIAL, len(specs), U), np.float32)
    target = np.zeros((A.T_TRIAL, len(specs)), np.float32)
    shas = []
    for b, sp in enumerate(specs):
        path = records / sp["file"]
        if A.seed_of_file(path) != sp["seed"]:
            raise ValueError(f"{sp['file']}: seed differs from the request")
        h = A.sha256_file(path)
        if man.get(sp["file"]) != h:
            raise ValueError(f"{sp['file']}: SHA-256 differs from the manifest")
        z = A.open_record(path, split_path)
        if not np.array_equal(z["dn_index"], setup.fw_index):
            raise ValueError(f"{sp['file']}: DN columns differ from the setup")
        if str(z["stimulus"]) != sp["stimulus"] or int(z["seed"]) != sp["seed"]:
            raise ValueError(f"{sp['file']}: stimulus or seed differs from its name")
        if bridge is not None:
            units[:, b] = bridge.unit_rates(z["dn_counts"]) / RT.R_IN
        target[:, b] = A.target_series(sp["stimulus"], sp["speed"])
        shas.append(h)
    return A.Trials(units, target, [s["stimulus"] for s in specs], np.array([s["speed"] for s in specs]),
                    np.array([s["seed"] for s in specs]), [s["file"] for s in specs], shas)


def read_group(name, setup, bridge, **kw) -> A.Trials:
    """One held-out group of ``A.TEST_GROUPS`` (the same conditions as final-eval) on the validation seeds."""
    if name not in GROUPS:
        raise PosthocRefusal(f"group {name!r}: only {tuple(GROUPS)}")
    return read_heldout(setup, bridge, GROUPS[name], **kw)


# ── part 2: analysis A (prereg/21 sections 2 and 3) ────────────────────────────────────────────────────────────────
OUT = A.OUT.parent / "A_posthoc"
PRED = OUT / "pred"
JOB_LOG = OUT / "analysis_A.log"
CONSISTENCY = OUT / "consistency.json"
GRAPHS = A.OUT / "graphs"
REPORT_JSON = ROOT / "docs" / "results" / "f5_stageA_report.json"
OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_posthoc.json"
OUT_MD = ROOT / "docs" / "results" / "f5_stageA_posthoc.md"
ARMS = ("true", "shuffled", "derange", "mlp", "linear")           # 10 seeds each; `untrained` (seed 0) for tuning only
TRAINING_SEEDS = tuple(range(10))
TOL_R2 = TOL_LOSS = 1e-4                                         # prereg/21 section 2
BOOT_SEED, N_BOOT, CI, ALPHA = 21, 10_000, 0.95, 0.05             # prereg/21 section 3.2
TUNING_SPEEDS = (30, 60, 90, 120)
HELD = {"g30": "held_out_secondary_30", "g90": "held_out_primary_90"}
REPLICATED = "replicated on the validation seeds (post-hoc)"
NOT_REPLICATED = "not replicated on the validation seeds (post-hoc)"


def run_list():
    return [(a, t) for a in ARMS for t in TRAINING_SEEDS] + [("untrained", 0)]


def selected_lr(arm):
    if arm == "untrained":
        return None
    return json.loads((A.OUT / arm / "lr_selected.json").read_text())["lr"]


def logged_eval(run_dir, step) -> dict:
    """The validation evaluation logged at ``step`` (the last one, if a resumed run logged it twice)."""
    rows = [json.loads(ln) for ln in (Path(run_dir) / "log.jsonl").read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("event") == "eval" and r.get("set") == "validation" and r.get("step") == step]
    if not rows:
        raise SystemExit(f"{run_dir}: no validation evaluation logged at step {step}")
    return rows[-1]


def compare(mine: dict, logged: dict) -> dict:
    """prereg/21 section 2: |delta R2| <= 1e-4, |delta loss| <= 1e-4, equal sign accuracy, same trial counts."""
    dr2, dloss = abs(mine["r2"] - logged["r2"]), abs(mine["loss"] - logged["loss"])
    ok = (dr2 <= TOL_R2 and dloss <= TOL_LOSS and mine["sign_accuracy"] == logged["sign_accuracy"]
          and mine["n_trials"] == logged["n_trials"] and mine["n_rotation"] == logged["n_rotation"])
    return dict(r2=mine["r2"], logged_r2=logged["r2"], abs_delta_r2=dr2, loss=mine["loss"], logged_loss=logged["loss"],
                abs_delta_loss=dloss, sign_accuracy=mine["sign_accuracy"], logged_sign_accuracy=logged["sign_accuracy"],
                n_trials=mine["n_trials"], passed=bool(ok))


def _records_sha(data) -> str:
    return hashlib.sha256("".join(data.sha256).encode()).hexdigest()


def _save(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _trial_arrays(prefix, data, dA):
    return {f"{prefix}dA": np.asarray(dA, np.float32), f"{prefix}stimulus": np.array(data.stimulus),
            f"{prefix}speed": data.speed, f"{prefix}seed": data.seed, f"{prefix}files": np.array(data.files),
            f"{prefix}sha256": np.array(data.sha256)}


def trials_of(z, prefix) -> A.Trials:
    """The saved trials of one prediction file as ``A.Trials`` (no units; the teacher rebuilt from the conditions)."""
    stim, speed = [str(s) for s in z[f"{prefix}stimulus"]], z[f"{prefix}speed"]
    target = np.stack([A.target_series(st, int(v) or None) for st, v in zip(stim, speed)], 1)
    return A.Trials(np.zeros((A.T_TRIAL, len(stim), 0), np.float32), target, stim, speed, z[f"{prefix}seed"],
                    [str(f) for f in z[f"{prefix}files"]], [str(h) for h in z[f"{prefix}sha256"]])


class JobLog:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, msg, **kw):
        line = json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), msg=msg, **kw), default=float)
        with self.path.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def predict(argv_log=None):
    """prereg/21 section 2, then 3.1: the training-speed pass and consistency check of every run, then (only if all
    pass) the held-out-speed pass. Never inside final_evaluation(); never a test seed (the reader and load_trials)."""
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_train as FT
    if A.in_final_evaluation():
        raise A.SealedSplitError("never inside final_evaluation()")
    torch.set_num_threads(A.TEST_THREADS)
    A.deterministic("cpu")
    PRED.mkdir(parents=True, exist_ok=True)
    log = JobLog(argv_log or JOB_LOG)
    setup = FT.load_setup()
    log("job start", phase="all", threads=torch.get_num_threads(), runs=len(run_list()))
    bridges, data_cache = {}, {}

    def bridge_of(arm, t):
        key = ("derange", t) if arm == "derange" else ("true", 0)
        if key not in bridges:
            bridges[key] = A.make_bridge(setup, key[0], key[1])
        return key, bridges[key]

    def model_of(arm, t, bridge):
        lr = selected_lr(arm)
        run_dir = A.run_dir_of(arm, t, lr, A.OUT)
        done = json.loads((run_dir / "DONE").read_text())
        if done.get("smoke") or done.get("replicate") or done["eval_set"] != "validation":
            raise SystemExit(f"{run_dir}: not a registered run")
        model, _ = A.make_model(arm, t, setup, bridge, GRAPHS)
        ck = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
        if ck["step"] != done["selected_step"]:
            raise SystemExit(f"{run_dir}: best.pt step {ck['step']} differs from the selected step")
        model.load_state_dict(ck["model"])
        return model.to("cpu"), run_dir, done, lr, A.sha256_file(run_dir / "best.pt")

    # phase 1: training speeds (validation seeds through load_trials) and the consistency check
    rows = {}
    for arm, t in run_list():
        path = PRED / f"{arm}_t{t}_train_speeds.npz"
        key, bridge = bridge_of(arm, t)
        run_dir = A.run_dir_of(arm, t, selected_lr(arm), A.OUT)
        t0 = time.time()
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["best_sha256"]) != A.sha256_file(run_dir / "best.pt"):
                raise SystemExit(f"{path}: made from another best.pt; not rewritten")
            data, dA = trials_of(z, ""), z["dA"].astype(np.float64)
            log("run skipped (prediction exists)", phase="consistency", arm=arm, seed=t)
        else:
            log("run start", phase="consistency", arm=arm, seed=t)
            if key not in data_cache:
                data_cache.clear()                                   # one bridge's trials in memory at a time
                data_cache[key] = A.load_trials("val", setup, bridge, A.TRAIN_CONDITIONS)
            data = data_cache[key]
            model, run_dir, done, lr, best_sha = model_of(arm, t, bridge)
            dA = A.predict(model, data.units, "cpu")
            _save(path, best_sha256=np.str_(best_sha), selected_step=np.int64(done["selected_step"]),
                  **_trial_arrays("", data, dA))
        m = A.metrics(dA, data)
        if arm == "untrained":
            row = dict(metrics=m, checked=False, passed=True)
        else:
            done = json.loads((run_dir / "DONE").read_text())
            row = dict(**compare(m, logged_eval(run_dir, done["selected_step"])), checked=True,
                       selected_step=done["selected_step"])
        rows[f"{arm}/t{t}"] = row
        log("run end", phase="consistency", arm=arm, seed=t, wall_s=round(time.time() - t0, 1),
            result={k: row[k] for k in ("passed", "abs_delta_r2", "abs_delta_loss") if k in row})
    checked = [r for r in rows.values() if r["checked"]]
    res = dict(label=LABEL, rule="prereg/21 section 2", tol_r2=TOL_R2, tol_loss=TOL_LOSS, n_checked=len(checked),
               n_passed=sum(r["passed"] for r in checked), passed=all(r["passed"] for r in checked),
               max_abs_delta_r2=max(r["abs_delta_r2"] for r in checked),
               max_abs_delta_loss=max(r["abs_delta_loss"] for r in checked), runs=rows)
    if CONSISTENCY.exists() and json.loads(CONSISTENCY.read_text()) != json.loads(json.dumps(res, default=float)):
        raise SystemExit(f"{CONSISTENCY} exists with other content; not rewritten")
    CONSISTENCY.write_text(json.dumps(res, indent=1, default=float))
    log("consistency check", passed=res["passed"], n_passed=res["n_passed"], n_checked=res["n_checked"],
        max_abs_delta_r2=res["max_abs_delta_r2"], max_abs_delta_loss=res["max_abs_delta_loss"])
    if not res["passed"]:
        log("job stop: the consistency check failed; no held-out-speed record was read")
        return 3

    # phase 2: held-out speeds (validation seeds through the post-hoc reader)
    data_cache.clear()
    for arm, t in run_list():
        path = PRED / f"{arm}_t{t}_held_out.npz"
        key, bridge = bridge_of(arm, t)
        run_dir = A.run_dir_of(arm, t, selected_lr(arm), A.OUT)
        if path.exists():
            if str(np.load(path)["best_sha256"]) != A.sha256_file(run_dir / "best.pt"):
                raise SystemExit(f"{path}: made from another best.pt; not rewritten")
            log("run skipped (prediction exists)", phase="held_out", arm=arm, seed=t)
            continue
        t0 = time.time()
        log("run start", phase="held_out", arm=arm, seed=t)
        if key not in data_cache:
            data_cache.clear()
            data_cache[key] = {g: read_group(name, setup, bridge) for g, name in HELD.items()}
        model, run_dir, done, lr, best_sha = model_of(arm, t, bridge)
        arrays, result = {}, {}
        for g, data in data_cache[key].items():
            dA = A.predict(model, data.units, "cpu")
            arrays.update(_trial_arrays(f"{g}_", data, dA))
            m = A.metrics(dA, data)
            result[g] = dict(r2=m["r2"], sign_accuracy=m["sign_accuracy"], n_trials=m["n_trials"])
        _save(path, best_sha256=np.str_(best_sha), selected_step=np.int64(done["selected_step"]), **arrays)
        log("run end", phase="held_out", arm=arm, seed=t, wall_s=round(time.time() - t0, 1), result=result)
    log("job end")
    return 0


# ── report, section A ───────────────────────────────────────────────────────────────────────────────────────────────
def window_mean(dA) -> np.ndarray:
    return np.asarray(dA, np.float64)[A.WINDOW[0]:A.WINDOW[1]].mean(0)


def tuning(wm, stimulus, speed) -> dict:
    """prereg/21 section 3.3 on the rotation trials: response per condition, OLS gain and offset against dA*, and the
    gain ratio G(v) = mean of sign(dA*) dA over the trials at v, divided by |dA*(v)| = K_P v."""
    wm, speed = np.asarray(wm, np.float64), np.asarray(speed, np.int64)
    tgt = np.array([A.target_value(s, int(v)) for s, v in zip(stimulus, speed)])
    g, b = np.polyfit(tgt, wm, 1)
    resp = {f"{st}_v{v:03d}": float(wm[(np.array(stimulus) == st) & (speed == v)].mean())
            for v in TUNING_SPEEDS for st in STIMULI}
    G = {str(v): float((np.sign(tgt[speed == v]) * wm[speed == v]).mean() / (A.K_P * v)) for v in TUNING_SPEEDS}
    n = {str(v): int((speed == v).sum()) for v in TUNING_SPEEDS}
    return dict(gain=float(g), offset=float(b), fit_r2=A.r2(g * tgt + b, wm), G=G, response=resp, n_trials=int(len(wm)),
                n_per_speed=n)


def _ms(x) -> dict:
    x = np.asarray(x, np.float64)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None, n=int(len(x)))


def _sha(p) -> str:
    return A.sha256_file(p)


def build_A(pred=PRED, consistency=CONSISTENCY, report_json=REPORT_JSON) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_stageA_report as R
    cons = json.loads(Path(consistency).read_text())
    if not cons["passed"]:
        raise SystemExit("the consistency check failed; section A is not built")
    rep = json.loads(Path(report_json).read_text())
    runs, records = {}, {}
    for arm, t in run_list():
        z1 = np.load(Path(pred) / f"{arm}_t{t}_train_speeds.npz", allow_pickle=False)
        z2 = np.load(Path(pred) / f"{arm}_t{t}_held_out.npz", allow_pickle=False)
        if str(z1["best_sha256"]) != str(z2["best_sha256"]):
            raise SystemExit(f"{arm} t{t}: the two prediction files come from different checkpoints")
        tr = trials_of(z1, "")
        row = dict(best_sha256=str(z1["best_sha256"]), selected_step=int(z1["selected_step"]),
                   val_training_speeds=A.metrics(z1["dA"], tr))
        wms, stims, speeds = [], [], []
        rot = tr.rotation()
        wms.append(window_mean(z1["dA"])[rot])
        stims += [s for s, r in zip(tr.stimulus, rot) if r]
        speeds.append(tr.speed[rot])
        for g in HELD:
            d = trials_of(z2, f"{g}_")
            m = A.metrics(z2[f"{g}_dA"], d)
            row[g] = dict(r2=m["r2"], sign_accuracy=m["sign_accuracy"], n_trials=m["n_trials"],
                          n_rotation=m["n_rotation"])
            records.setdefault(g, _records_sha(d))
            if records[g] != _records_sha(d):
                raise SystemExit(f"{arm} t{t}: other {g} records")
            wms.append(window_mean(z2[f"{g}_dA"]))
            stims += d.stimulus
            speeds.append(d.speed)
        row["tuning"] = tuning(np.concatenate(wms), stims, np.concatenate(speeds))
        if row["tuning"]["n_trials"] != 8 * len(SEEDS):
            raise SystemExit(f"{arm} t{t}: {row['tuning']['n_trials']} rotation trials, 320 expected")
        test = rep["runs"][arm][f"t{t}"]
        if test["best_sha256"] != row["best_sha256"]:
            raise SystemExit(f"{arm} t{t}: best.pt differs from the one the test report evaluated")
        row["test"] = {k: test[k] for k in ("r2", "sign_accuracy", "g30_r2", "g30_sign_accuracy", "g90_r2",
                                            "g90_sign_accuracy")}
        runs.setdefault(arm, {})[f"t{t}"] = row

    def per_seed(arm, g, k):
        return [runs[arm][f"t{t}"][g][k] for t in TRAINING_SEEDS]

    d30 = np.subtract(per_seed("true", "g30", "r2"), per_seed("shuffled", "g30", "r2"))
    sf = R.sign_flip(d30)
    ci = R.bootstrap_ci(d30, seed=BOOT_SEED, n_boot=N_BOOT, level=CI)
    replicated = bool(sf["mean_d"] > 0 and sf["p_greater"] < ALPHA)
    primary = dict(metric="R2 at 30 deg/s, validation seeds 160-199", d=d30.tolist(), test=sf, effect=ci,
                   alpha=ALPHA, h1="mean d > 0 (the direction of the test result)",
                   label=REPLICATED if replicated else NOT_REPLICATED,
                   test_split_d=[runs["true"][f"t{t}"]["test"]["g30_r2"] - runs["shuffled"][f"t{t}"]["test"]["g30_r2"]
                                 for t in TRAINING_SEEDS])
    comparisons = []
    for other in ARMS[1:]:
        for g in HELD:
            for k in ("r2", "sign_accuracy"):
                d = np.subtract(per_seed("true", g, k), per_seed(other, g, k))
                comparisons.append(dict(comparison=f"{k}(true) - {k}({other}) at {g[1:]} deg/s", other=other,
                                        group=g, metric=k, d=d.tolist(),
                                        **R.bootstrap_ci(d, seed=BOOT_SEED, n_boot=N_BOOT, level=CI)))
    summary = {}
    for arm in ARMS + ("untrained",):
        seeds = TRAINING_SEEDS if arm != "untrained" else (0,)
        rr = [runs[arm][f"t{t}"] for t in seeds]
        s = {f"val_{g}_{k}": _ms([r[g][k] for r in rr]) for g in HELD for k in ("r2", "sign_accuracy")}
        s.update({f"test_{g}_{k}": _ms([r["test"][f"{g}_{k}"] for r in rr]) for g in HELD
                  for k in ("r2", "sign_accuracy")})
        s["val_training_speeds_r2"] = _ms([r["val_training_speeds"]["r2"] for r in rr])
        s["gain"] = _ms([r["tuning"]["gain"] for r in rr])
        s["offset"] = _ms([r["tuning"]["offset"] for r in rr])
        s["fit_r2"] = _ms([r["tuning"]["fit_r2"] for r in rr])
        s["G"] = {str(v): _ms([r["tuning"]["G"][str(v)] for r in rr]) for v in TUNING_SPEEDS}
        summary[arm] = s
    return dict(
        label=LABEL, plan="prereg/21_f5_stageA_posthoc_plan.md", plan_sha256=_sha(PLAN),
        inputs=dict(split_sha256=A.split_sha256(), manifest_sha256=_sha(A.RECORDS / "manifest.sha256"),
                    setup_sha256=_sha(A.OUT / "setup.npz"), report_json_sha256=_sha(report_json),
                    held_out_records_sha256=records, val_seeds=[SEEDS[0], SEEDS[-1]]),
        consistency={k: cons[k] for k in ("rule", "tol_r2", "tol_loss", "n_checked", "n_passed", "passed",
                                          "max_abs_delta_r2", "max_abs_delta_loss")},
        A=dict(note="validation data seeds 160-199; held-out speeds read by the post-hoc reader, training speeds by "
                    "load_trials; test-split values from docs/results/f5_stageA_report.json (no test record opened)",
               primary=primary, comparisons=comparisons, summary=summary, runs=runs),
        B=None)


# ── md ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _f(x, k=4):
    return "n/a" if x is None else f"{x:.{k}f}"


def _msf(s, k=4):
    return _f(s["mean"], k) if s["sd"] is None else f"{s['mean']:.{k}f} ± {s['sd']:.{k}f}"


def plan_limitations(path=PLAN) -> list:
    sec = Path(path).read_text().split("## 7. ", 1)[1].split("\n## ", 1)[0]
    return [" ".join(x.split()) for x in re.findall(r"^- (.+?)(?=^- |\Z)", sec, re.S | re.M)]


def render(d) -> str:
    a, c = d["A"], d["consistency"]
    p, e, sf = a["primary"], a["primary"]["effect"], a["primary"]["test"]
    S = a["summary"]
    L = [f"# F5 stage A: post-hoc analyses ({LABEL})", "",
         f"**{LABEL.capitalize()}.** Written after the stage-A test; the analyses are fixed in "
         f"`{d['plan']}` (SHA-256 `{d['plan_sha256'][:12]}`). Nothing here enters or changes a stage-A verdict. "
         "Generated by `scripts/f5_stageA_posthoc.py`; checked by `scripts/verify_report.py` "
         "(`check_f5_stageA_posthoc`).", "",
         f"Primary post-hoc test: mean d = R2(true) - R2(shuffled) at 30 deg/s on the validation seeds "
         f"{_f(sf['mean_d'])} (95 % bootstrap interval {_f(e['lo'])} to {_f(e['hi'])}, {sf['n']} seeds); one-sided "
         f"exact p {_f(sf['p_greater'])} ({sf['count_greater']}/{sf['n_patterns']}). Label: **{p['label']}**.", "",
         "## A. Speed generalisation on the validation seeds", "",
         f"Validation data seeds {d['inputs']['val_seeds'][0]}-{d['inputs']['val_seeds'][1]}. The held-out speeds "
         "(30, 90 deg/s) were read for the first time by the post-hoc reader; the training speeds (60, 120 deg/s) "
         "through `load_trials`. Test-split values are read from `docs/results/f5_stageA_report.json`; no test record "
         "was opened.", "",
         "### A.0 Consistency check (prereg/21 section 2)", "",
         f"Validation metrics at the training speeds recomputed on the CPU against the evaluation logged at the "
         f"selected step: {c['n_passed']} of {c['n_checked']} runs pass (largest |delta R2| "
         f"{c['max_abs_delta_r2']:.2e}, largest |delta loss| {c['max_abs_delta_loss']:.2e}; tolerance "
         f"{c['tol_r2']:.0e} each, equal sign accuracy). The held-out-speed records were read only after this check "
         "passed.", "",
         "### A.1 Primary post-hoc test (prereg/21 section 3.2)", "",
         "d_t = R2_30(true, t) - R2_30(shuffled, t), validation seeds; the test-split d from the report for comparison.",
         "", "| seed | R2 true | R2 shuffled | d | test-split d |", "|---:|---:|---:|---:|---:|"]
    for t in TRAINING_SEEDS:
        rt, rs = a["runs"]["true"][f"t{t}"]["g30"]["r2"], a["runs"]["shuffled"][f"t{t}"]["g30"]["r2"]
        L.append(f"| {t} | {_f(rt)} | {_f(rs)} | {_f(p['d'][t])} | {_f(p['test_split_d'][t])} |")
    L += ["", f"- One-sided exact sign-flip test (all {sf['n_patterns']} sign patterns, H1: {p['h1']}): "
              f"p = {_f(sf['p_greater'])} ({sf['count_greater']}/{sf['n_patterns']}); two-sided p = "
              f"{_f(sf['p_two_sided'])} ({sf['count_two_sided']}/{sf['n_patterns']}).",
          f"- Effect size: mean d {_f(e['mean'])}, 95 % percentile bootstrap interval {_f(e['lo'])} to {_f(e['hi'])} "
          f"({e['n_boot']:,} resamples of the seeds, `default_rng({e['rng_seed']})`).",
          f"- Label: **{p['label']}** (mean d > 0 and one-sided p < {p['alpha']}).", "",
          "### A.2 Replication per arm (descriptive)", "",
          "Mean ± SD over the training seeds; validation (val) next to the test split (test). 80 rotation trials per "
          "speed on each split.", "",
          "| arm | val R2 30 | test R2 30 | val R2 90 | test R2 90 | val sign 30 | test sign 30 | val sign 90 | "
          "test sign 90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = S[arm]
        L.append(f"| {arm} | {_msf(s['val_g30_r2'])} | {_msf(s['test_g30_r2'])} | {_msf(s['val_g90_r2'])} | "
                 f"{_msf(s['test_g90_r2'])} | {_msf(s['val_g30_sign_accuracy'], 3)} | "
                 f"{_msf(s['test_g30_sign_accuracy'], 3)} | {_msf(s['val_g90_sign_accuracy'], 3)} | "
                 f"{_msf(s['test_g90_sign_accuracy'], 3)} |")
    L += ["", "Paired differences `true` minus each other arm on the validation seeds (95 % percentile bootstrap "
              f"intervals, {N_BOOT:,} resamples, a fresh `default_rng({BOOT_SEED})` each; no p-value):", "",
          "| comparison | mean d | 95 % interval |", "|---|---:|---:|"]
    for cmp in a["comparisons"]:
        L.append(f"| {cmp['comparison']} | {_f(cmp['mean'])} | {_f(cmp['lo'])} to {_f(cmp['hi'])} |")
    L += ["", "### A.3 Speed tuning (prereg/21 section 3.3, descriptive)", "",
          "Per run, on the 320 validation rotation trials (30, 60, 90, 120 deg/s, both directions): OLS fit "
          "dA = g dA* + b of the window-mean dA on the teacher (b in deg), and the gain ratio G(v) = mean "
          "sign(dA*) dA / (K_P v) per speed (G = 1 is the teacher). Mean ± SD over the training seeds; `untrained`: "
          "one run.", "",
          "| arm | g | b (deg) | fit R2 | G(30) | G(60) | G(90) | G(120) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS + ("untrained",):
        s = S[arm]
        L.append(f"| {arm} | {_msf(s['gain'], 3)} | {_msf(s['offset'], 3)} | {_msf(s['fit_r2'], 3)} | "
                 + " | ".join(_msf(s["G"][str(v)], 3) for v in TUNING_SPEEDS) + " |")
    L += [""] + (render_B(d["B"]) if d.get("B") else ["## B. Structure", "", "Not yet computed (part 3).", ""])
    L += ["## Limitations (prereg/21 section 7)", ""] + [f"- {x}" for x in plan_limitations()] + [""]
    return "\n".join(L)


# ── part 3: analysis B, structure (prereg/21 section 4) ───────────────────────────────────────────────────────────
TYPES = ("b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4")
READOUT_SIGN = {m: (1 if m == "b2" else -1) for m in TYPES}         # prereg/21 section 4.3, S18
MEASURES = {
    "S1": "mean shortest-path length over reachable pairs (hops)",
    "S2": "fraction of pairs with shortest path <= 2 hops",
    "S3": "number of 1-hop walks",
    "S4": "number of 2-hop walks",
    "S5": "number of 3-hop walks",
    "S6": "summed 1-hop weight, sum of W1 (synapses)",
    "S7": "summed 2-hop weight, sum of W2 (synapses^2)",
    "S8": "ipsilateral fraction of P",
    "S9": "excitatory fraction of P",
    **{f"S{10 + k}": f"signed lateral influence index L_{m}" for k, m in enumerate(TYPES)},
    "S18": "readout-aligned lateral index, sum of r_m L_m",
}
FRACTIONS = ("S1", "S2", "S8", "S9")                                # printed with 4 decimals; the rest are integers
DEFICITS = {"D1": "test split, training speeds (report JSON primary d)",
            "D2": "test split, 30 deg/s (report JSON g30_r2)",
            "D3": "validation split, 30 deg/s (section A.1)"}
RHO_CRIT = 0.648                                                   # prereg/21 section 4.4: two-sided 5 %, n = 10
LENGTH_BINS = ("1", "2", "3", "4", ">=5", "unreachable")


def graph_ends(setup, bridge) -> dict:
    """prereg/21 section 4.1: the bridge-target DNs with their BANC side, the 16 readout MNs with type and side."""
    src = np.asarray(bridge.target_index, np.int64)
    side_of = dict(zip(np.asarray(setup.banc_index).tolist(), np.asarray(setup.banc_side).astype(str).tolist()))
    names = setup.meta["readout_names"]
    if names != [f"{m}|{s}" for m in TYPES for s in ("left", "right")]:
        raise SystemExit(f"readout names {names} differ from the 8 types x left / right of prereg/21")
    if not np.array_equal(np.sign(setup.sigma), [READOUT_SIGN[m] * (1 if s == "left" else -1)
                                                  for m in TYPES for s in ("left", "right")]):
        raise SystemExit("the readout signs of the setup differ from r_m of prereg/21 section 4.3")
    return dict(src=src, src_side=np.array([side_of[int(i)] for i in src]), tgt=np.asarray(setup.readout, np.int64),
                tgt_type=np.array([x.split("|")[0] for x in names]), tgt_side=np.array([x.split("|")[1] for x in names]))


def structure(n, pre, post, n_syn, sign, ends) -> dict:
    """S1-S18 of one graph (prereg/21 sections 4.2 and 4.3); exact integer arithmetic except the ratios."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    src, tgt = ends["src"], ends["tgt"]
    pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
    w, s = np.asarray(n_syn, np.int64), np.asarray(sign, np.int64)
    if np.bincount(post, minlength=n)[src].any():
        raise SystemExit("a source DN has an incoming edge")

    def mat(v):
        return sp.csr_matrix((v, (pre, post)), shape=(n, n), dtype=np.int64)

    N, Ns, B = mat(w), mat(s * w), mat(np.ones_like(w))
    Np, Nn = mat(np.where(s > 0, w, 0)), mat(np.where(s < 0, w, 0))
    Ns_, Np_, Nn_, B_ = (M[src] for M in (Ns, Np, Nn, B))
    W1 = N[src][:, tgt].toarray()
    W2 = N[src] @ N[:, tgt].toarray()
    Q = Ns_[:, tgt].toarray() + Ns_ @ Ns[:, tgt].toarray()
    P = W1 + W2
    pos = int(Np_[:, tgt].sum()) + int((Np_ @ Np[:, tgt].toarray()).sum() + (Nn_ @ Nn[:, tgt].toarray()).sum())
    BM = B[:, tgt].toarray()
    dist = shortest_path(B.T.tocsr(), directed=True, unweighted=True, indices=tgt)[:, src].T     # (DN, MN) hops
    reach = np.isfinite(dist)
    hops = dist[reach].astype(np.int64)
    lat = np.isin(ends["src_side"], ("left", "right"))[:, None] & np.isin(ends["tgt_side"], ("left", "right"))[None]
    ipsi = ends["src_side"][:, None] == ends["tgt_side"][None, :]

    def q(src_side, m, tgt_side):
        return int(Q[np.ix_(ends["src_side"] == src_side, (ends["tgt_type"] == m) & (ends["tgt_side"] == tgt_side))].sum())

    L = {m: (q("left", m, "left") - q("right", m, "left")) - (q("left", m, "right") - q("right", m, "right"))
         for m in TYPES}
    v = dict(S1=float(hops.mean()) if len(hops) else None, S2=float((dist <= 2).sum() / dist.size),
             S3=int(BM[src].sum()), S4=int((B_ @ BM).sum()), S5=int((B_ @ (B @ BM)).sum()),
             S6=int(W1.sum()), S7=int(W2.sum()), S8=float(P[lat & ipsi].sum() / P[lat].sum()),
             S9=float(pos / P.sum()), **{f"S{10 + k}": L[m] for k, m in enumerate(TYPES)},
             S18=int(sum(READOUT_SIGN[m] * L[m] for m in TYPES)))
    lengths = dict(zip(LENGTH_BINS, [int((hops == k).sum()) for k in (1, 2, 3, 4)] + [int((hops >= 5).sum()),
                                                                                      int((~reach).sum())]))
    return dict(values=v, lengths=lengths, n_pairs=int(dist.size), n_sources=int(len(src)), n_targets=int(len(tgt)))


def avg_rank_desc(x) -> np.ndarray:
    """Ranks, 1 = largest, ties the average rank."""
    from scipy.stats import rankdata
    return rankdata(-np.asarray(x, np.float64), method="average")


def spearman(x, y):
    """Spearman rho with average ranks (Pearson of the ranks); None if either vector is constant."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def deficits(A_sec, rep) -> dict:
    """prereg/21 section 4.4: D1 and D2 from the committed test report JSON (no test record), D3 from section A."""
    runs = rep["runs"]
    d1 = list(rep["primary"]["d"])
    if d1 != [runs["true"][f"t{t}"]["r2"] - runs["shuffled"][f"t{t}"]["r2"] for t in TRAINING_SEEDS]:
        raise SystemExit("the report's primary d is not R2(true) - R2(shuffled) per seed")
    d2 = [runs["true"][f"t{t}"]["g30_r2"] - runs["shuffled"][f"t{t}"]["g30_r2"] for t in TRAINING_SEEDS]
    return dict(D1=d1, D2=d2, D3=list(A_sec["primary"]["d"]))


def build_B(A_sec, report_json=REPORT_JSON) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_train as FT
    setup = FT.load_setup()
    ends = graph_ends(setup, A.make_bridge(setup, "true", 0))
    graphs, shas = {"true": structure(setup.n, setup.pre, setup.post, setup.n_syn, setup.sign, ends)}, {}
    gsha = setup.graph().sha256()
    for t in TRAINING_SEEDS:
        path = GRAPHS / f"shuffled_t{t}.npz"
        z = np.load(path, allow_pickle=False)
        if int(z["seed"]) != A.SHUFFLE_SEED0 + t or str(z["graph_sha256"]) != gsha:
            raise SystemExit(f"{path}: not the cached shuffle of seed {t} of the true graph; never rebuilt here")
        post, info = A.shuffled_post(setup, t, GRAPHS)
        if not info["check"]["passed"]:
            raise SystemExit(f"{path}: the shuffle check failed")
        shas[f"t{t}"] = _sha(path)
        graphs[f"shuffled_t{t}"] = structure(setup.n, setup.pre, post, setup.n_syn, setup.sign, ends)
    D = deficits(A_sec, json.loads(Path(report_json).read_text()))
    sh = [f"shuffled_t{t}" for t in TRAINING_SEEDS]
    measures, corr = {}, {}
    for k in MEASURES:
        tv, xs = graphs["true"]["values"][k], [graphs[g]["values"][k] for g in sh]
        x = np.asarray(xs, np.float64)
        sd = float(x.std(ddof=1))
        measures[k] = dict(description=MEASURES[k], true=tv, shuffled=xs, shuffled_mean=float(x.mean()),
                           shuffled_sd=sd, shuffled_min=xs[int(np.argmin(x))], shuffled_max=xs[int(np.argmax(x))],
                           rank_true=float(avg_rank_desc([tv] + xs)[0]), n_graphs=1 + len(xs),
                           outside_shuffled_range=bool(tv > x.max() or tv < x.min()),
                           z_true=(tv - float(x.mean())) / sd if sd > 0 else None,
                           z_shuffled=((x - x.mean()) / sd).tolist() if sd > 0 else None)
        corr[k] = {dk: spearman(x, dv) if np.ptp(x) > 0 else None for dk, dv in D.items()}
    rhos = [(k, dk, r) for k in MEASURES for dk, r in corr[k].items() if r is not None]
    best = None
    for k, dk, r in rhos:                                           # first largest |rho|: S order, then D order
        if best is None or abs(r) > abs(best[2]):
            best = (k, dk, r)
    n_corr = len(MEASURES) * len(DEFICITS)
    return dict(
        note="no training and no record: the true VNC graph of runs/f5/A/setup.npz and the 10 cached shuffled graphs",
        inputs=dict(setup_sha256=_sha(A.OUT / "setup.npz"), graph_sha256=gsha, shuffled_graphs_sha256=shas),
        sources=dict(n=int(len(ends["src"])), left=int((ends["src_side"] == "left").sum()),
                     right=int((ends["src_side"] == "right").sum()), definition="bridge.target_index of the true bridge"),
        targets=dict(n=int(len(ends["tgt"])), names=setup.meta["readout_names"], readout_sign=READOUT_SIGN),
        graphs=graphs, measures=measures, deficits=D, deficit_definitions=DEFICITS,
        correlations=dict(rho=corr, n=n_corr, n_computed=len(rhos), rho_crit=RHO_CRIT,
                          expected_by_chance=n_corr * ALPHA,
                          n_at_or_above_crit=sum(abs(r) >= RHO_CRIT for _, _, r in rhos),
                          strongest=dict(measure=best[0], deficit=best[1], rho=best[2]),
                          correction="none (descriptive)"))


def build() -> dict:
    d = build_A()
    d["B"] = build_B(d["A"])
    return d


def _mv(k, x):
    if x is None:
        return "n/a"
    return f"{x:.4f}" if k in FRACTIONS else f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


def summary_B(b) -> str:
    out = [k for k, m in b["measures"].items() if m["outside_shuffled_range"]]
    c, s = b["correlations"], b["correlations"]["strongest"]
    return (f"`true` lies outside the range of the 10 shuffled graphs on {len(out)} of {len(b['measures'])} "
            f"measures ({', '.join(out) if out else 'none'}). Of {c['n']} uncorrected structure-deficit "
            f"correlations, {c['n_at_or_above_crit']} reach |rho| >= {c['rho_crit']} (about "
            f"{c['expected_by_chance']:.1f} expected by chance); the largest is {s['measure']} with {s['deficit']}, "
            f"rho = {s['rho']:.3f}.")


def render_B(b) -> list:
    src, c = b["sources"], b["correlations"]
    L = ["## B. Structure (prereg/21 section 4, descriptive)", "",
         f"No training and no record: the `true` VNC graph and the 10 cached `shuffled` graphs. Sources: the "
         f"{src['n']} bridge-target DNs ({src['left']} left, {src['right']} right); targets: the "
         f"{b['targets']['n']} readout MNs; {b['graphs']['true']['n_pairs']:,} (DN, MN) pairs. Rank 1 = largest of "
         f"{next(iter(b['measures'].values()))['n_graphs']} graphs (ties: average rank); z = (value - shuffled "
         "mean) / shuffled SD. No test.", "", summary_B(b), "",
         "### B.1 Measures, `true` against the shuffles", "",
         "| id | measure | true | shuffled mean ± SD | shuffled range | rank of true | z of true | outside range |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for k, m in b["measures"].items():
        z = "n/a" if m["z_true"] is None else f"{m['z_true']:.2f}"
        sd = _mv(k, m["shuffled_sd"]) if k in FRACTIONS else f"{m['shuffled_sd']:,.1f}"
        mean = _mv(k, m["shuffled_mean"]) if k in FRACTIONS else f"{m['shuffled_mean']:,.1f}"
        L.append(f"| {k} | {m['description']} | {_mv(k, m['true'])} | {mean} ± {sd} | {_mv(k, m['shuffled_min'])} "
                 f"to {_mv(k, m['shuffled_max'])} | {m['rank_true']:g} | {z} | {'yes' if m['outside_shuffled_range'] else 'no'} |")
    L += ["", "Shortest-path length distribution (number of (DN, MN) pairs; shuffled: mean over the 10 graphs):", "",
          "| graph | " + " | ".join(LENGTH_BINS) + " |", "|---|" + "---:|" * len(LENGTH_BINS)]
    sh = [g for g in b["graphs"] if g != "true"]
    L.append("| true | " + " | ".join(f"{b['graphs']['true']['lengths'][x]:,}" for x in LENGTH_BINS) + " |")
    L.append("| shuffled | " + " | ".join(f"{np.mean([b['graphs'][g]['lengths'][x] for g in sh]):,.1f}"
                                          for x in LENGTH_BINS) + " |")
    L += ["", "### B.2 Structure against the per-seed deficit (prereg/21 section 4.4)", "",
          "Deficit vectors d_t = R2(true, t) - R2(shuffled, t) per training seed: "
          + "; ".join(f"{k}: {v}" for k, v in b["deficit_definitions"].items())
          + ". D1 and D2 are read from the committed report JSON; no test record was opened. Spearman rho of the "
            "shuffled-graph value x_t of training seed t with d_t, n = 10.", "",
          f"**{c['n']} correlations, no correction.** About {c['expected_by_chance']:.1f} are expected at |rho| >= "
          f"{c['rho_crit']} (the two-sided 5 % critical value at n = 10) when nothing is there; "
          f"{c['n_at_or_above_crit']} reach it here. Every value below is descriptive.", "",
          "| id | rho D1 | rho D2 | rho D3 |", "|---|---:|---:|---:|"]
    for k, r in c["rho"].items():
        L.append(f"| {k} | " + " | ".join("n/a" if r[dk] is None else f"{r[dk]:.3f}" for dk in DEFICITS) + " |")
    s = c["strongest"]
    L += ["", f"Largest |rho| (selected as the largest of {c['n']} uncorrected correlations): {s['measure']} "
              f"({b['measures'][s['measure']]['description']}) with {s['deficit']}, rho = {s['rho']:.3f}.", ""]
    return L


def report(overwrite=False):
    for p in (OUT_JSON, OUT_MD):
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=1, default=float) + "\n")
    OUT_MD.write_text(render(json.loads(OUT_JSON.read_text())))
    print(OUT_MD.read_text())
    return 0


# ── figure (prereg/21 section 5), from the committed JSON only ─────────────────────────────────────────────────────
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/posthoc")    # machine 2: viewing in Windows
FIG_FILES = ("posthoc_stageA_figure.pdf", "posthoc_stageA_figure.png", "posthoc_stageA_figure_caption.txt")
MM = 1 / 25.4
FONT_PT = 7
ARM_COLOR = {"true": "#2a78d6", "shuffled": "#eb6834", "derange": "#1baf7a", "mlp": "#8a5cd6", "linear": "#b8860b",
             "untrained": "#8c8a84"}
INK, INK2, FAINT = "#000000", "#555555", "#bdbdbd"
SHORT = dict(S1="mean path length", S2="pairs <= 2 hops", S3="1-hop walks", S4="2-hop walks", S5="3-hop walks",
             S6="sum W1", S7="sum W2", S8="ipsilateral fraction", S9="excitatory fraction",
             **{f"S{10 + k}": f"L {m}" for k, m in enumerate(TYPES)}, S18="sum r_m L_m")


def font_setup():
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    if not any(f.name == "Arial" for f in fm.fontManager.ttflist):
        for f in ("arial.ttf", "arialbd.ttf"):                       # the Windows copy of Arial on this machine
            if (Path("/mnt/c/Windows/Fonts") / f).exists():
                fm.fontManager.addfont(str(Path("/mnt/c/Windows/Fonts") / f))
    if not any(f.name == "Arial" for f in fm.fontManager.ttflist):
        raise SystemExit("Arial is not available")
    mpl.rcParams.update({"font.family": "Arial", "font.size": FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out",
                         "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.labelsize": FONT_PT, "xtick.labelsize": FONT_PT - 1, "ytick.labelsize": FONT_PT - 1,
                         "legend.fontsize": FONT_PT - 1, "axes.titlesize": FONT_PT})


def caption(d) -> str:
    a, b = d["A"], d["B"]
    p, e, sf = a["primary"], a["primary"]["effect"], a["primary"]["test"]
    c, s = b["correlations"], b["correlations"]["strongest"]
    out = [k for k, m in b["measures"].items() if m["outside_shuffled_range"]]
    return (
        f"Post-hoc, exploratory analyses of F5 stage A (prereg/21; {LABEL}). Not part of a stage-A verdict.\n\n"
        "a, Speed tuning on the validation data seeds 160-199: gain ratio G(v) = mean sign(dA*) dA / (K_P v) of the "
        "window-mean output against drum speed; lines, means over the 10 training seeds; dots, single seeds; "
        "untrained, one run. Dashed line, the teacher (G = 1). 60 and 120 deg/s were training speeds; 30 and 90 deg/s "
        "(shaded) were held out.\n"
        f"b, R2 at 30 deg/s on the validation seeds, true against shuffled per training seed (lines pair the seeds). "
        f"Primary post-hoc test: mean d = {e['mean']:.3f} (95 % bootstrap interval {e['lo']:.3f} to {e['hi']:.3f}), "
        f"one-sided exact sign-flip p = {sf['p_greater']:.4f} ({sf['count_greater']}/{sf['n_patterns']}); "
        f"{p['label']}.\n"
        f"c, Structure of the true VNC graph against the 10 shuffled graphs: per measure S1-S18, z = (value - "
        f"shuffled mean) / shuffled SD (grey, shuffled graphs; coloured, true; filled, true outside the shuffled "
        f"range). Symmetric-log axis, linear within |z| <= 2. True lies outside the shuffled range on {len(out)} of "
        f"{len(b['measures'])} measures. Rank and z are descriptive; there is one true graph.\n"
        f"d, The strongest structure-deficit correlation, selected as the largest |rho| of {c['n']} uncorrected "
        f"Spearman correlations (18 measures x 3 deficit vectors; about {c['expected_by_chance']:.1f} expected at "
        f"|rho| >= {c['rho_crit']} by chance, {c['n_at_or_above_crit']} observed): {s['measure']} "
        f"({b['measures'][s['measure']]['description']}) of the shuffled graph of each training seed against "
        f"{s['deficit']} ({b['deficit_definitions'][s['deficit']]}); rho = {s['rho']:.3f}, n = 10; numbers, the "
        "training seeds.\n\n"
        "Test-split values come only from docs/results/f5_stageA_report.json; no test record was opened.\n")


def draw(d):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    font_setup()
    a, b = d["A"], d["B"]
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    top = GridSpec(1, 2, figure=fig, width_ratios=(1.55, 1), left=0.075, right=0.985, bottom=0.62, top=0.965,
                   wspace=0.3)
    bot = GridSpec(1, 2, figure=fig, width_ratios=(1.25, 1), left=0.165, right=0.985, bottom=0.07, top=0.52,
                   wspace=0.32)
    axa, axb, axc, axd = fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(bot[0]), fig.add_subplot(bot[1])
    seeds = [f"t{t}" for t in TRAINING_SEEDS]

    # a: speed tuning
    arms = ARMS + ("untrained",)
    for v in (30, 90):
        axa.axvspan(v - 9, v + 9, color="#efefef", lw=0, zorder=0)
    axa.axhline(1, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for i, arm in enumerate(arms):
        off = (i - (len(arms) - 1) / 2) * 1.6
        runs = [a["runs"][arm][s] for s in (seeds if arm != "untrained" else ["t0"])]
        x = np.array(TUNING_SPEEDS, float) + off
        G = np.array([[r["tuning"]["G"][str(v)] for v in TUNING_SPEEDS] for r in runs])
        col = ARM_COLOR[arm]
        if len(G) > 1:
            axa.scatter(np.repeat(x, len(G)), G.T.ravel(), s=2.5, color=col, alpha=0.45, lw=0, zorder=2)
        axa.plot(x, G.mean(0), color=col, lw=1.0, marker="o", ms=2.6, zorder=3, label=arm)
    axa.set_xticks(TUNING_SPEEDS, [f"{v}\n{'held out' if v in (30, 90) else 'training'}" for v in TUNING_SPEEDS])
    axa.set_xlim(18, 132)
    axa.set_xlabel("drum speed (deg/s)")
    axa.set_ylabel("gain ratio G(v)")
    axa.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.4, columnspacing=1.0, borderaxespad=0.2)

    # b: 30 deg/s per seed
    rt = [a["runs"]["true"][s]["g30"]["r2"] for s in seeds]
    rs = [a["runs"]["shuffled"][s]["g30"]["r2"] for s in seeds]
    for u, w in zip(rs, rt):
        axb.plot([0, 1], [u, w], color=FAINT, lw=0.6, zorder=1)
    axb.scatter(np.zeros(10), rs, s=9, color=ARM_COLOR["shuffled"], lw=0, zorder=2)
    axb.scatter(np.ones(10), rt, s=9, color=ARM_COLOR["true"], lw=0, zorder=2)
    axb.set_xticks([0, 1], ["shuffled", "true"])
    axb.set_xlim(-0.45, 1.45)
    axb.set_ylim(0, 0.75)
    axb.set_ylabel("validation R2 at 30 deg/s")
    p, e, sf = a["primary"], a["primary"]["effect"], a["primary"]["test"]
    axb.text(0.5, 0.04, f"mean d = {e['mean']:.3f}\n[{e['lo']:.3f}, {e['hi']:.3f}]\none-sided p = "
                        f"{sf['p_greater']:.4f}", transform=axb.transAxes, ha="center", va="bottom",
             fontsize=FONT_PT - 1)

    # c: structure, z of true against the shuffles
    ks = list(b["measures"])
    for r, k in enumerate(ks):
        m = b["measures"][k]
        y = len(ks) - 1 - r
        if m["z_true"] is None:
            continue
        axc.scatter(m["z_shuffled"], np.full(len(m["z_shuffled"]), y), s=4, color=FAINT, lw=0, zorder=2)
        out = m["outside_shuffled_range"]
        axc.scatter([m["z_true"]], [y], s=12, zorder=3, color=ARM_COLOR["true"] if out else "white",
                    edgecolor=ARM_COLOR["true"], lw=0.7)
    axc.axvline(0, color=INK2, lw=0.5, zorder=1)
    axc.set_xscale("symlog", linthresh=2, linscale=0.8)
    lim = max(abs(b["measures"][k]["z_true"]) for k in ks if b["measures"][k]["z_true"] is not None)
    axc.set_xlim(-lim * 1.6, lim * 1.6)
    axc.set_xticks([-100, -10, -2, 0, 2, 10, 100], ["-100", "-10", "-2", "0", "2", "10", "100"])
    axc.set_yticks(range(len(ks)), [f"{k}  {SHORT[k]}" for k in reversed(ks)])
    axc.tick_params(axis="y", length=0)
    axc.set_ylim(-0.7, len(ks) - 0.3)
    axc.set_xlabel("z = (value - shuffled mean) / shuffled SD")
    axc.spines["left"].set_visible(False)

    # d: the strongest structure-deficit correlation
    s = b["correlations"]["strongest"]
    x = b["measures"][s["measure"]]["shuffled"]
    y = b["deficits"][s["deficit"]]
    axd.scatter(x, y, s=10, color=ARM_COLOR["shuffled"], lw=0, zorder=2)
    for t, (u, w) in enumerate(zip(x, y)):
        axd.annotate(str(t), (u, w), xytext=(2.5, 1.5), textcoords="offset points", fontsize=FONT_PT - 1.5,
                     color=INK2)
    axd.set_xlabel(f"{s['measure']} of the shuffled graph ({b['measures'][s['measure']]['description'].split(' index ')[-1]})"
                   if s["measure"] in [f"S{10 + k}" for k in range(8)] else f"{s['measure']} of the shuffled graph")
    axd.set_ylabel(f"{s['deficit']}: R2(true) - R2(shuffled)")
    axd.text(0.03, 0.97, f"Spearman rho = {s['rho']:.3f}, n = 10\nlargest |rho| of {b['correlations']['n']} "
                         "(uncorrected)", transform=axd.transAxes, ha="left", va="top", fontsize=FONT_PT - 1)
    for ax, lab, x0 in ((axa, "a", 0.008), (axb, "b", None), (axc, "c", 0.008), (axd, "d", None)):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.075 if x0 is None else x0, pos.y1 + 0.012, lab, ha="left", va="bottom",
                 fontsize=FONT_PT + 1, fontweight="bold")
    fig.text(0.985, 0.995, LABEL, ha="right", va="top", fontsize=FONT_PT - 1, color=INK2)
    return fig


def _copy_new(src: Path, dst: Path):
    """Copy to a file that must not exist yet (never overwrites); one retry if Windows holds the file."""
    data = src.read_bytes()
    for attempt in (0, 1):
        try:
            with open(dst, "xb") as f:
                f.write(data)
            return
        except OSError as e:
            if e.errno != 22 or attempt:
                raise SystemExit(f"{dst}: {e}. If a viewer has the file open, close it and run `figure` again.")
            time.sleep(2)


def figure(overwrite=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads(OUT_JSON.read_text())
    if not d.get("B"):
        raise SystemExit(f"{OUT_JSON} has no section B; run `report` first")
    local = [OUT / f for f in FIG_FILES]
    for p in local:
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    existing = [PREVIEW / f for f in FIG_FILES if (PREVIEW / f).exists()]
    if existing:
        raise SystemExit(f"{existing}: already in the preview folder; never overwritten")
    fig = draw(d)
    fig.savefig(local[0])
    fig.savefig(local[1], dpi=300)
    plt.close(fig)
    local[2].write_text(caption(d))
    PREVIEW.mkdir(parents=False, exist_ok=True)
    for p in local:
        _copy_new(p, PREVIEW / p.name)
    print("\n".join(str(PREVIEW / f) for f in FIG_FILES))
    return 0


def report_A(overwrite=False):
    for p in (OUT_JSON, OUT_MD):
        if p.exists() and not overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    d = build_A()
    OUT_JSON.write_text(json.dumps(d, indent=1, default=float) + "\n")
    OUT_MD.write_text(render(d))
    print(OUT_MD.read_text())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("predict")
    for name in ("report-A", "report", "figure"):
        sub.add_parser(name).add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        return predict()
    if a.cmd == "figure":
        return figure(a.overwrite)
    return (report_A if a.cmd == "report-A" else report)(a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
