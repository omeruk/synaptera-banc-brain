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
         "Generated by `scripts/f5_stageA_posthoc.py`; checked by `scripts/verify_report.py`.", "",
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
    L += ["", "## B. Structure", "", "Not yet computed (part 3).", "",
          "## Limitations (prereg/21 section 7)", ""] + [f"- {x}" for x in plan_limitations()] + [""]
    return "\n".join(L)


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
    r = sub.add_parser("report-A")
    r.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        return predict()
    return report_A(a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
