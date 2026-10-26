"""F5 stage A: post-hoc, exploratory analyses (prereg/21). Never part of a stage-A verdict.

Part 1 holds the reader only: the validation data seeds 160-199 at the held-out drum speeds (30 and 90 deg/s), which
``train.f5_stageA.load_trials`` refuses outside the final evaluation. This reader does not use ``final_evaluation()``
and refuses to run inside it. It accepts only those seeds, the stimuli ``rot_left`` / ``rot_right`` and those speeds;
every request is checked in full before any file is opened, and a test seed (200-239) raises ``SealedSplitError``.
Each record is checked against ``runs/f5/stageA/manifest.sha256``, whose own SHA-256 is pinned (prereg/21 section 1).
The 60 and 120 deg/s validation trials are read through ``load_trials`` itself.
"""
from __future__ import annotations

import sys
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
