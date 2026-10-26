"""scripts/f5_stageA_posthoc.py (prereg/21, post-hoc): the reader's refusals, on synthetic records in a temporary
directory. No real record is opened; the real split file is read only to check that its validation seeds are 160-199."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_posthoc as P  # noqa: E402
from train import f5_stageA as A  # noqa: E402

N_DN = 5


def _write_split(path, val=range(160, 200)):
    path.write_text(json.dumps(dict(train=list(range(160)), val=list(val), test=list(range(200, 240)))))
    return path


def _write_record(brain, stim, speed, seed, stim_in_file=None, seed_in_file=None):
    rng = np.random.default_rng(seed * 1000 + speed)
    np.savez(brain / A.record_name(stim, speed, seed), stimulus=np.str_(stim_in_file or stim),
             seed=np.int64(seed if seed_in_file is None else seed_in_file),
             dn_counts=rng.integers(0, 3, (A.T_TRIAL, N_DN)).astype(np.int16), dn_index=np.arange(N_DN))


def _write_manifest(root):
    lines = [f"{A.sha256_file(p)}  brain/{p.name}" for p in sorted((root / "brain").glob("*.npz"))]
    (root / "manifest.sha256").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def world(tmp_path):
    """Synthetic records: validation seeds 160, 161 at 30 / 90 deg/s, a test seed 200 at 30 deg/s, a training-speed
    record of seed 160 and a training seed 0 at 30 deg/s."""
    brain = tmp_path / "brain"
    brain.mkdir()
    for s in (160, 161):
        for v in (30, 90):
            for st in P.STIMULI:
                _write_record(brain, st, v, s)
    _write_record(brain, "rot_left", 30, 200)
    _write_record(brain, "rot_left", 60, 160)
    _write_record(brain, "rot_left", 30, 0)
    _write_manifest(tmp_path)
    split = _write_split(tmp_path / "splits.json")
    setup = SimpleNamespace(fw_index=np.arange(N_DN))
    bridge = SimpleNamespace(units=[0, 1], unit_rates=lambda c: np.asarray(c, np.float32)[:, :2] * 10.0)
    kw = dict(records=tmp_path, split_path=split, manifest_sha256=A.sha256_file(tmp_path / "manifest.sha256"))
    return SimpleNamespace(root=tmp_path, setup=setup, bridge=bridge, kw=kw, split=split)


@pytest.fixture()
def no_file_access(monkeypatch):
    """Fails the test if the reader hashes or opens any file."""
    def boom(*a, **k):
        raise AssertionError("a file was accessed before the refusal")
    monkeypatch.setattr(A, "sha256_file", boom)
    monkeypatch.setattr(A, "open_record", boom)


def _read(w, conditions=(("rot_left", 30),), seeds=(160,), **kw):
    return P.read_heldout(w.setup, w.bridge, conditions, seeds=seeds, **{**w.kw, **kw})


def test_the_real_split_lists_seeds_160_to_199_as_validation():
    P.check_split()
    assert set(P.SEEDS).isdisjoint(A.splits()["test"])


def test_reads_validation_seeds_at_the_held_out_speeds(world):
    cond = (("rot_left", 30), ("rot_right", 30), ("rot_left", 90), ("rot_right", 90))
    d = _read(world, cond, seeds=(160, 161))
    assert len(d) == 8 and d.units.shape == (A.T_TRIAL, 8, 2)
    assert list(d.seed) == [160] * 4 + [161] * 4 and list(d.speed) == [30, 30, 90, 90] * 2
    for b, (st, v) in enumerate(cond):
        assert np.all(d.target[:A.N_GREY, b] == 0)
        assert np.allclose(d.target[A.N_GREY:, b], A.target_value(st, v))
    z = np.load(world.root / "brain" / A.record_name("rot_left", 30, 160))
    assert np.allclose(d.units[:, 0], z["dn_counts"][:, :2] * 10.0 / P.RT.R_IN)


@pytest.mark.parametrize("seed", [200, 219, 239])
def test_a_test_seed_is_refused_before_any_file_is_opened(world, no_file_access, seed):
    with pytest.raises(A.SealedSplitError):
        _read(world, seeds=(seed,))


def test_one_test_seed_in_a_mixed_request_refuses_the_whole_request(world, no_file_access):
    with pytest.raises(A.SealedSplitError):
        _read(world, seeds=(160, 161, 200))


@pytest.mark.parametrize("seed", [0, 159, 240, -1])
def test_an_unlisted_seed_is_refused(world, no_file_access, seed):
    with pytest.raises(P.PosthocRefusal):
        _read(world, seeds=(seed,))


@pytest.mark.parametrize("cond", [("rot_left", 60), ("rot_right", 120), ("rot_left", 45), ("rot_left", None),
                                  ("rot_left", "30"), ("rot_left", 30.0), ("static", 30), ("grey", None),
                                  ("static", None), ("rot_up", 30)])
def test_an_unlisted_speed_or_stimulus_is_refused(world, no_file_access, cond):
    with pytest.raises(P.PosthocRefusal):
        _read(world, (cond,))


def test_duplicate_or_empty_requests_are_refused(world, no_file_access):
    for kw in (dict(seeds=(160, 160)), dict(conditions=(("rot_left", 30), ("rot_left", 30))), dict(seeds=()),
               dict(conditions=())):
        with pytest.raises(P.PosthocRefusal):
            _read(world, **kw)


def test_refused_inside_the_final_evaluation(world, no_file_access):
    with A.final_evaluation():
        with pytest.raises(A.SealedSplitError):
            _read(world)


def test_a_split_file_with_other_validation_seeds_is_refused(world, no_file_access):
    with pytest.raises(P.PosthocRefusal):
        _read(world, split_path=_write_split(world.root / "other.json", val=range(170, 200)))


def test_a_changed_manifest_is_refused(world):
    with pytest.raises(ValueError, match="pinned manifest"):
        _read(world, manifest_sha256="0" * 64)


def test_a_record_that_differs_from_the_manifest_is_refused(world):
    """The record is rewritten under the same name with other counts; the manifest is not updated."""
    np.savez(world.root / "brain" / A.record_name("rot_left", 30, 160), stimulus=np.str_("rot_left"),
             seed=np.int64(160), dn_counts=np.zeros((A.T_TRIAL, N_DN), np.int16), dn_index=np.arange(N_DN))
    with pytest.raises(ValueError, match="differs from the manifest"):
        _read(world)


@pytest.mark.parametrize("field", ["stimulus", "seed"])
def test_a_record_whose_content_differs_from_its_name_is_refused(world, field):
    brain = world.root / "brain"
    _write_record(brain, "rot_left", 30, 161, **({"stim_in_file": "rot_right"} if field == "stimulus"
                                                  else {"seed_in_file": 160}))
    _write_manifest(world.root)
    kw = dict(manifest_sha256=A.sha256_file(world.root / "manifest.sha256"))
    with pytest.raises(ValueError, match="differs from its name"):
        _read(world, seeds=(161,), **kw)


def test_read_group_uses_the_final_eval_groups_and_refuses_others(world, no_file_access):
    assert P.GROUPS["held_out_secondary_30"] == (("rot_left", 30), ("rot_right", 30))
    assert P.GROUPS["held_out_primary_90"] == (("rot_left", 90), ("rot_right", 90))
    with pytest.raises(P.PosthocRefusal):
        P.read_group("training_speeds", world.setup, world.bridge, **world.kw)


# ── analysis A helpers (synthetic numbers only) ─────────────────────────────────────────────────────────────────────
def test_tuning_recovers_gain_offset_and_gain_ratio():
    stim, speed = [], []
    for v in P.TUNING_SPEEDS:
        for st in P.STIMULI:
            stim += [st] * 3
            speed += [v] * 3
    tgt = np.array([A.target_value(s, v) for s, v in zip(stim, speed)])
    t = P.tuning(0.8 * tgt + 0.5, stim, speed)
    assert np.isclose(t["gain"], 0.8) and np.isclose(t["offset"], 0.5) and np.isclose(t["fit_r2"], 1.0)
    assert all(np.isclose(g, 0.8) for g in t["G"].values())        # the offset cancels over both directions
    assert np.isclose(t["response"]["rot_right_v030"], 0.8 * A.K_P * 30 + 0.5)
    assert t["n_trials"] == 24 and set(t["n_per_speed"].values()) == {6}


def test_compare_applies_the_registered_tolerances():
    base = dict(r2=0.9, loss=0.05, sign_accuracy=1.0, n_trials=240, n_rotation=160)
    assert P.compare(dict(base, r2=0.9 + 0.9e-4, loss=0.05 - 0.9e-4), base)["passed"]
    assert not P.compare(dict(base, r2=0.9 + 1.1e-4), base)["passed"]
    assert not P.compare(dict(base, loss=0.05 + 1.1e-4), base)["passed"]
    assert not P.compare(dict(base, sign_accuracy=0.99375), base)["passed"]
