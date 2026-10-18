"""F5 stage-A runner (train/f5_stageA.py, train/f5_gates.py, scripts/f5_train_guard.py) on tiny synthetic data: the
sealed test split, teacher, arms, metrics, data order, checkpoints and bit-identical resume, learning-rate selection,
the final evaluation, the gates' logic and the guard's pause / resume. No data files are needed."""
import hashlib
import zlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from engine import rate_torch as RT
from train import f5_gates as G
from train import f5_stageA as A

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("DNa01", "DNa02", "DNb01", "DNx01")
TRAIN, VAL, TEST = list(range(0, 6)), [6, 7], [8, 9]
SPEEDS = (30, 60, 90, 120)


# ── synthetic world ─────────────────────────────────────────────────────────────────────────────────────────────────
def synthetic_setup(seed=0):
    """VNC of 60 neurons: 10 BANC DNs (4 bridge families x 2 sides, plus DNg02 L/R) clamped; 16 readout MNs."""
    rng = np.random.default_rng(seed)
    n, n_dn = 60, 10
    inputs = np.arange(n_dn)
    pre, post = [], []
    for i in range(n_dn, n):
        cand = rng.choice(np.setdiff1d(np.arange(n), [i]), size=6, replace=False)
        pre += cand.tolist()
        post += [i] * 6
    pre, post = np.array(pre), np.array(post)
    sign_of = rng.choice([-1, 1], size=n, p=[0.3, 0.7])
    g = RT.RateGraph.from_arrays(n, pre, post, rng.integers(1, 20, len(pre)), sign_of[pre], inputs)
    banc_types = [f for f in FAMILIES for _ in (0, 1)] + ["DNg02", "DNg02"]
    banc_side = ["left", "right"] * 5
    base = np.zeros(n_dn, np.float32)
    base[8:] = 1.0                                               # DNg02 baseline (HAND)
    fw_types = [f for f in FAMILIES for _ in range(4)] + ["DNz99"] * 2
    fw_side = (["left", "left", "right", "right"] * 4) + ["left", "right"]
    readout = np.arange(n - 16, n)
    sigma = np.array([1, -1] * 8, np.float32)
    return A.Setup(n=n, pre=g.pre, post=g.post, n_syn=g.n_syn, sign=g.sign, inputs=g.inputs, readout=readout,
                   sigma=sigma, base=base, banc_cell_type=np.array(banc_types), banc_side=np.array(banc_side),
                   banc_index=inputs.astype(np.int64), fw_cell_type=np.array(fw_types), fw_side=np.array(fw_side),
                   fw_index=np.arange(100, 100 + len(fw_types)), families=FAMILIES,
                   meta=dict(readout_names=[f"m{k}" for k in range(16)]))


def write_world(root: Path, setup):
    """Split file, records of every split (rot_left / rot_right at 4 speeds, static, grey) and the manifest. A rotation
    raises the spike rate of the FlyWire DNs on the side the pattern moves to (a learnable direction signal)."""
    (root / "brain").mkdir(parents=True)
    split = root / "splits.json"
    split.write_text(json.dumps(dict(train=TRAIN, val=VAL, test=TEST)))
    side = np.asarray(setup.fw_side)
    lines = []
    for s in TRAIN + VAL + TEST:
        conds = [(st, v) for st in ("rot_left", "rot_right") for v in SPEEDS] + [("static", None), ("grey", None)]
        for st, v in conds:
            rng = np.random.default_rng([s, zlib.crc32(f"{st}{v}".encode())])
            lam = np.full((300, len(side)), 0.2)
            if st.startswith("rot"):
                lam[100:, side == ("right" if st == "rot_right" else "left")] += 0.02 * v
            name = A.record_name(st, v or 0, s)
            np.savez_compressed(root / "brain" / name, stimulus=st, seed=s, dn_counts=rng.poisson(lam).astype(np.int16),
                                dn_index=setup.fw_index)
            lines.append(f"{hashlib.sha256((root / 'brain' / name).read_bytes()).hexdigest()}  brain/{name}")
    (root / "manifest.sha256").write_text("\n".join(lines) + "\n")
    return split


@pytest.fixture
def world(tmp_path):
    setup = synthetic_setup()
    split = write_world(tmp_path / "rec", setup)
    return setup, tmp_path / "rec", split


@pytest.fixture(autouse=True)
def _deterministic():
    yield
    torch.use_deterministic_algorithms(False)


def load(world, split, arm="true", t=0, conditions=A.TRAIN_CONDITIONS):
    setup, rec, sp = world
    br = A.make_bridge(setup, arm, t)
    return br, A.load_trials(split, setup, br, conditions, records=rec, split_path=sp)


def cfg(tmp, arm="true", seed=0, lr=1e-2, steps=6, **kw):
    kw.setdefault("batch", 4)
    kw.setdefault("eval_every", 2)
    kw.setdefault("ckpt_every", 1)
    return A.RunConfig(arm=arm, seed=seed, lr=lr, run_dir=tmp, steps=steps, **kw)


def train(world, run_dir, arm="true", seed=0, lr=1e-2, steps=6, stop_after=None, **kw):
    setup, rec, sp = world
    br, tr = load(world, "train", arm, seed)
    _, va = load(world, "val", arm, seed)
    model, info = A.make_model(arm, seed, setup, br, run_dir.parent / "graphs")
    c = cfg(run_dir, arm, seed, lr, steps, **kw)
    rec_ = A.run_record(c, setup, info, br, tr, va, split_path=sp)
    out = A.train_run(c, model, rec_, tr, va, log_fn=lambda *a: None, stop_after=stop_after)
    return model, out


# ── splits and the sealed test set ──────────────────────────────────────────────────────────────────────────────────
def test_split_file_is_the_registered_one():
    s = A.splits()
    assert s["train"] == tuple(range(0, 160)) and s["val"] == tuple(range(160, 200)) and s["test"] == tuple(range(200, 240))
    d = json.loads(A.SPLIT_FILE.read_text())
    assert d["train_speeds_deg_s"] == [60, 120] and d["held_out_speeds_deg_s"] == {"primary": 90, "secondary": 30}
    assert len(A.split_sha256()) == 64


def test_record_names_match_the_stage_A_driver():
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_stageA_records as R
    for st, v, s in (("rot_left", 60, 0), ("rot_right", 120, 199), ("static", 0, 7), ("grey", 0, 230)):
        assert A.record_name(st, v, s) == R.record_name(st, v, s)
        assert A.seed_of_file(A.record_name(st, v, s)) == s


def test_test_record_is_refused_outside_the_final_evaluation(world):
    _, rec, sp = world
    test_file = rec / "brain" / A.record_name("rot_left", 60, TEST[0])
    with pytest.raises(A.SealedSplitError):
        A.open_record(test_file, sp)
    with pytest.raises(A.SealedSplitError):
        A.trial_specs("test", split_path=sp)
    with pytest.raises(A.SealedSplitError):
        load(world, "test")
    for split in ("train", "val"):                             # held-out speeds: final evaluation only
        with pytest.raises(A.SealedSplitError, match="held-out"):
            load(world, split, conditions=(("rot_left", 90),))
        with pytest.raises(A.SealedSplitError, match="held-out"):
            load(world, split, conditions=(("rot_right", 30),))
    with A.final_evaluation():
        assert int(A.open_record(test_file, sp)["seed"]) == TEST[0]
    assert not A.in_final_evaluation()
    with pytest.raises(A.SealedSplitError):                    # the context does not leak
        A.open_record(test_file, sp)


def test_no_test_seed_reaches_training_or_selection(world, tmp_path, monkeypatch):
    """Every file opened by loading, training (every trained arm), validation, checkpointing and learning-rate
    selection is recorded; none may carry a test seed. The final evaluation is the only path that opens them."""
    opened = []
    real = np.load

    def spy(path, *a, **k):
        opened.append(Path(path).name)
        return real(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    for arm in ("true", "shuffled", "derange", "mlp", "linear", "bridge_off", "untrained"):
        for lr in (A.LR_GRID if arm != "untrained" else (0.0,)):
            train(world, tmp_path / "runs" / arm / f"t0_lr{lr:g}", arm, 0, lr, steps=2)
    seeds = {A.seed_of_file(f) for f in opened if A._SEED_RE.search(f)}
    assert seeds and seeds <= set(TRAIN + VAL)
    assert not seeds & set(TEST)
    for arm in ("true",):
        for lr in A.LR_GRID:                                       # select_lr reads only DONE files
            d = tmp_path / "runs" / arm / f"t0_lr{lr:g}" / "DONE"
            x = json.loads(d.read_text())
            x["final_step"] = A.STEPS
            d.write_text(json.dumps(x))
        n0 = len(opened)
        A.select_lr(arm, tmp_path / "runs")
        assert len(opened) == n0


def test_load_trials_checks_manifest_and_split(world):
    setup, rec, sp = world
    good = sp.read_text()
    sp.write_text(json.dumps(dict(train=TRAIN + [TEST[0]], val=VAL, test=TEST)))   # a test seed listed as training
    with pytest.raises(A.SealedSplitError):
        A.load_trials("train", setup, A.make_bridge(setup, "true", 0), records=rec, split_path=sp)
    sp.write_text(good)
    f = rec / "brain" / A.record_name("grey", 0, TRAIN[0])
    z = dict(np.load(f))
    z["dn_counts"] = z["dn_counts"] + 1
    np.savez_compressed(f, **z)
    with pytest.raises(ValueError, match="manifest"):
        load(world, "train")


# ── teacher, arms, metrics ──────────────────────────────────────────────────────────────────────────────────────────
def test_teacher_targets():
    assert A.K_P == pytest.approx(12.5 / 120)
    for v, d in ((30, 3.125), (60, 6.25), (90, 9.375), (120, 12.5), (240, 12.5)):
        assert A.target_value("rot_right", v) == pytest.approx(d)
        assert A.target_value("rot_left", v) == pytest.approx(-d)
    assert A.target_value("static", None) == 0.0 and A.target_value("grey", None) == 0.0
    z = A.target_series("rot_right", 60)
    assert (z[:100] == 0).all() and (z[100:] == np.float32(6.25)).all() and len(z) == 300


def test_trials_and_bridge_input(world):
    setup, rec, sp = world
    br, tr = load(world, "train")
    assert len(tr) == 6 * len(TRAIN) and tr.units.shape == (300, len(tr), len(br.units))
    z = np.load(rec / "brain" / A.record_name("rot_left", 60, 0))
    np.testing.assert_array_equal(tr.units[:, 0], (br.unit_rates(z["dn_counts"]) / RT.R_IN).astype(np.float32))
    assert tr.rotation().sum() == 4 * len(TRAIN)
    assert set(tr.speed[tr.rotation()]) == {60, 120}


def test_graph_arm_clamp_baseline_bridge_off_and_derange(world):
    setup, _, _ = world
    br = A.make_bridge(setup, "true", 0)
    m = A.GraphArm(setup, setup.graph(), br)
    u = torch.rand(5, 2, len(br.units))
    x = m.clamp(u)
    assert torch.equal(x[:, :, 8:], torch.ones(5, 2, 2))                    # DNg02 = 1 (HAND)
    col = {int(i): k for k, i in enumerate(setup.inputs)}
    for k, i in enumerate(br.target_index):
        assert torch.equal(x[:, :, col[int(i)]], u[:, :, br.unit_of_target()[k]])
    off = A.GraphArm(setup, setup.graph(), None)
    assert torch.equal(off.clamp(torch.zeros(5, 2, 0)), setup_base(setup).expand(5, 2, -1))
    der = A.make_bridge(setup, "derange", 3)
    src = {u["target_family"]: u["source_family"] for u in der.units}
    assert all(s != t for t, s in src.items()) and sorted(src.values()) == sorted(src)
    assert A.derangement_of(3, setup.families) == A.derangement_of(3, setup.families)
    assert A.derangement_of(3, setup.families) != A.derangement_of(4, setup.families)


def setup_base(setup):
    return torch.as_tensor(np.asarray(setup.base, np.float32))


def test_real_ensemble_derangement_is_prereg11_random_t_plus_1():
    sys.path.insert(0, str(ROOT / "scripts"))
    import f3b3_common as C
    for t in (0, 9):
        assert A.derangement_of(t, C.ENSEMBLE) == C.permutations()[f"random_{t + 1}"]


def test_readout_is_the_fixed_lateral_sum(world):
    setup, _, _ = world
    m = A.GraphArm(setup, setup.graph(), A.make_bridge(setup, "true", 0))
    u = torch.rand(30, 3, m.n_units)
    st, _ = m.net.run(m.clamp(u), record=m.readout)
    expected = 0.395 * 100.0 * (st * torch.as_tensor(setup.sigma)).sum(2)
    torch.testing.assert_close(m(u), expected, rtol=0, atol=0)


def test_mlp_width_rule_and_filtered_features(world):
    for nf, target in ((900, 3_044_835), (16, 1000), (8, 200)):
        h = A.mlp_width(nf, target)
        assert A.mlp_param_count(nf, h) >= target > A.mlp_param_count(nf, h - 1)
    setup, _, _ = world
    br = A.make_bridge(setup, "true", 0)
    model, info = A.make_model("mlp", 0, setup, br, true_trainable=500)
    assert info["n_trainable"] >= 500 and A.mlp_param_count(4 * len(br.units), info["hidden"] - 1) < 500
    lin, li = A.make_model("linear", 0, setup, br)
    assert li["n_trainable"] == 4 * len(br.units) * 16 + 16
    u = torch.rand(40, 2, len(br.units), dtype=torch.float64)
    f = lin.double().features(u).numpy()
    for j, tau in enumerate(A.FILTER_MS):
        a, y = np.exp(-5.0 / tau), np.zeros((2, len(br.units)))
        for k in range(40):
            y = a * y + (1 - a) * u[k].numpy()
            np.testing.assert_allclose(f[k][:, j * len(br.units):(j + 1) * len(br.units)], y, rtol=1e-6)
    m2, _ = A.make_model("mlp", 0, setup, br, true_trainable=500)              # the training seed sets the init
    m3, _ = A.make_model("mlp", 1, setup, br, true_trainable=500)
    assert torch.equal(model.net[0].weight, m2.net[0].weight) and not torch.equal(model.net[0].weight, m3.net[0].weight)


def test_metrics_and_samples_to_set_accuracy():
    T, N = 300, 4
    tgt = np.zeros((T, N), np.float32)
    tgt[100:, 0], tgt[100:, 1] = 6.25, -12.5
    d = A.Trials(np.zeros((T, N, 0), np.float32), tgt, ["rot_right", "rot_left", "static", "grey"],
                 np.array([60, 120, 0, 0]), np.arange(N), ["a"] * N, ["h"] * N)
    m = A.metrics(tgt.astype(np.float64), d)
    assert m["loss"] == 0 and m["sign_accuracy"] == 1 and m["r2"] == 1 and m["n_rotation"] == 2
    bad = tgt.copy()
    bad[:, 1] *= -1
    m = A.metrics(bad, d)
    assert m["sign_accuracy"] == 0.5 and m["r2"] < 1
    ev = [dict(step=0, sign_accuracy=1.0), dict(step=50, sign_accuracy=0.5), dict(step=100, sign_accuracy=0.95)]
    assert A.samples_to_set_accuracy(ev) == dict(samples=1600, censored=False)
    assert A.samples_to_set_accuracy(ev[:2]) == dict(samples=A.STEPS * A.BATCH, censored=True)


def test_data_order_is_an_epochwise_shuffle():
    o = A.data_order(3, 960)
    assert len(o) >= A.STEPS * A.BATCH
    for e in range(5):
        assert sorted(o[e * 960:(e + 1) * 960]) == list(range(960))
    np.testing.assert_array_equal(o, A.data_order(3, 960))
    assert not np.array_equal(o[:960], A.data_order(4, 960)[:960])
    rng = np.random.default_rng(1003)
    np.testing.assert_array_equal(o[:960], rng.permutation(960))


# ── training, checkpoints, resume ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("arm", ["true", "mlp"])
def test_resume_is_bit_identical(world, tmp_path, arm):
    torch.manual_seed(0)
    full, _ = train(world, tmp_path / "full", arm, steps=6)
    torch.manual_seed(0)
    train(world, tmp_path / "res", arm, steps=6, stop_after=3)
    torch.manual_seed(999)
    resumed, done = train(world, tmp_path / "res", arm, steps=6)
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k
    log = A.JsonLog(tmp_path / "res" / "log.jsonl").read()
    steps = [r["step"] for r in log if r["event"] == "step"]
    assert steps == list(range(1, 7))                                      # no duplicate or missing step after resume
    assert [r["step"] for r in log if r["event"] == "resume"] == [3]
    full_log = [r for r in A.JsonLog(tmp_path / "full" / "log.jsonl").read() if r["event"] == "step"]
    assert [r["loss"] for r in full_log] == [r["loss"] for r in log if r["event"] == "step"]
    assert done["steps"] == 6 and not done["smoke"]


def test_checkpoints_best_and_done(world, tmp_path):
    _, done = train(world, tmp_path / "r", steps=8, eval_every=2, ckpt_every=3)
    log = A.JsonLog(tmp_path / "r" / "log.jsonl").read()
    ev = [r for r in log if r["event"] == "eval"]
    assert [e["step"] for e in ev] == [0, 2, 4, 6, 8] and not ev[0]["selectable"]
    assert {"loss", "sign_accuracy", "r2", "trials_seen"} <= set(ev[1])
    ck = [r["step"] for r in log if r["event"] == "checkpoint"]
    assert ck == [2, 3, 4, 6, 8]                                           # every ckpt_every and every evaluation
    best = min(ev[1:], key=lambda e: e["loss"])
    assert done["selected_step"] == best["step"] and done["selected_val_loss"] == best["loss"]
    assert torch.load(tmp_path / "r" / "best.pt", weights_only=False)["step"] == best["step"]
    assert done["final_val_loss"] == ev[-1]["loss"] and done["eval_set"] == "validation"
    assert len(list((tmp_path / "r").glob("step_*.pt"))) == 3
    cfg_ = json.loads((tmp_path / "r" / "config.json").read_text())
    assert cfg_["edge_grad"] == "sddmm" and cfg_["checkpoint_every"] == 0 and cfg_["latencies"] == A.LATENCIES
    assert cfg_["labels"]["dA"] == "TRAINED" and cfg_["labels"]["dng02_baseline"] == "HAND"
    assert len(cfg_["split_sha256"]) == 64 and cfg_["model"]["engine"]["edge_grad"] == "sddmm"


def test_training_learns_the_synthetic_direction_signal(world, tmp_path):
    _, done = train(world, tmp_path / "r", "linear", lr=1e-2, steps=40, eval_every=20, ckpt_every=20)
    ev = [r for r in A.JsonLog(tmp_path / "r" / "log.jsonl").read() if r["event"] == "eval"]
    assert ev[-1]["loss"] < ev[0]["loss"]


def test_changed_configuration_is_refused(world, tmp_path):
    train(world, tmp_path / "r", steps=2)
    (tmp_path / "r" / "DONE").unlink()
    with pytest.raises(SystemExit, match="configuration differs"):
        train(world, tmp_path / "r", lr=3e-3, steps=2)


def test_untrained_is_the_initialisation(world, tmp_path):
    model, done = train(world, tmp_path / "u", "untrained", lr=0.0, steps=6)
    assert done["selected_step"] == 0 and done["steps"] == 0 and done["lr"] is None
    ck = torch.load(tmp_path / "u" / "best.pt", weights_only=False)
    assert torch.equal(ck["model"]["net.theta"], torch.zeros_like(ck["model"]["net.theta"]))
    assert not [r for r in A.JsonLog(tmp_path / "u" / "log.jsonl").read() if r["event"] == "step"]


def test_select_lr_rule(tmp_path):
    for lr, final in zip(A.LR_GRID, (0.5, 0.2, 0.2)):
        d = tmp_path / "true" / f"t0_lr{lr:g}"
        d.mkdir(parents=True)
        (d / "DONE").write_text(json.dumps(dict(smoke=False, eval_set="validation", final_step=A.STEPS,
                                                final_val_loss=final, selected_val_loss=final / 2)))
    assert A.select_lr("true", tmp_path)["lr"] == 3e-3                        # tie: the smaller rate
    d = tmp_path / "true" / "t0_lr0.01" / "DONE"
    d.write_text(json.dumps(dict(smoke=True, eval_set="validation", final_step=A.STEPS, final_val_loss=0.1,
                                 selected_val_loss=0.1)))
    with pytest.raises(SystemExit):
        A.select_lr("true", tmp_path)


def test_final_evaluation_runs_once_and_never_on_smoke(world, tmp_path):
    setup, rec, sp = world
    model, _ = train(world, tmp_path / "r", steps=4)
    br = A.make_bridge(setup, "true", 0)
    m2, _ = A.make_model("true", 0, setup, br)
    res = A.final_evaluate(tmp_path / "r", setup, br, m2, records=rec, split_path=sp)
    assert set(res["groups"]) == {"training_speeds", "held_out_primary_90", "held_out_secondary_30"}
    assert res["groups"]["training_speeds"]["n_trials"] == 6 * len(TEST)
    assert res["groups"]["held_out_primary_90"]["n_rotation"] == 2 * len(TEST)
    assert not A.in_final_evaluation()
    with pytest.raises(SystemExit, match="once"):
        A.final_evaluate(tmp_path / "r", setup, br, m2, records=rec, split_path=sp)
    _, rec2, _ = world
    smoke = tmp_path / "s"
    br_, tr = load(world, "train")
    mm, info = A.make_model("true", 0, setup, br_)
    c = cfg(smoke, steps=2, smoke=True)
    A.train_run(c, mm, A.run_record(c, setup, info, br_, tr, None, split_path=sp), tr, None, log_fn=lambda *a: None)
    assert json.loads((smoke / "DONE").read_text())["eval_set"] == "smoke_eval_on_training_trials"
    with pytest.raises(SystemExit, match="smoke"):
        A.final_evaluate(smoke, setup, br_, mm, records=rec, split_path=sp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_final_evaluation_runs_on_the_cpu_and_is_reproducible(world, tmp_path):
    """prereg/19 section 1: a GPU-trained run is evaluated on the CPU from best.pt; two evaluations of the same
    checkpoint give identical numbers."""
    setup, rec, sp = world
    train(world, tmp_path / "r", steps=4, device="cuda")
    res = []
    for k in range(2):
        d = tmp_path / f"copy{k}"
        d.mkdir()
        for f in ("DONE", "best.pt"):
            (d / f).write_bytes((tmp_path / "r" / f).read_bytes())
        br = A.make_bridge(setup, "true", 0)
        m, _ = A.make_model("true", 0, setup, br)
        res.append(A.final_evaluate(d, setup, br, m.cuda(), records=rec, split_path=sp))
    assert res[0]["device"] == "cpu" and res[0]["threads"] == A.TEST_THREADS
    assert res[0]["groups"] == res[1]["groups"]


def test_replicate_is_never_tested_or_selected(world, tmp_path):
    setup, rec, sp = world
    run = A.run_dir_of("true", 0, 3e-3, tmp_path, replicate=True)
    assert run.name == "t0_lr0.003_replicate"
    train(world, run, steps=2, replicate=True)
    assert json.loads((run / "DONE").read_text())["replicate"] is True
    br = A.make_bridge(setup, "true", 0)
    m, _ = A.make_model("true", 0, setup, br)
    with pytest.raises(SystemExit, match="replicate"):
        A.final_evaluate(run, setup, br, m, records=rec, split_path=sp)
    with pytest.raises(SystemExit, match="missing"):                      # select_lr reads only t0_lr<rate>
        A.select_lr("true", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_resume_is_bit_identical(world, tmp_path):
    """On this tiny graph the cuSPARSE products happen to be reproducible; on the full VNC they are not (see
    A.REPRODUCIBILITY), so this checks the resume bookkeeping on the GPU, not GPU bit-reproducibility in general."""
    torch.manual_seed(0)
    full, _ = train(world, tmp_path / "full", steps=4, device="cuda")
    train(world, tmp_path / "res", steps=4, device="cuda", stop_after=2)
    resumed, _ = train(world, tmp_path / "res", steps=4, device="cuda")
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k


# ── gates ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_ridge_cv_alignment():
    rng = np.random.default_rng(0)
    seeds = np.repeat(np.arange(40), 6)
    y = rng.choice([-12.5, -6.25, 0, 6.25, 12.5], size=len(seeds))
    X = np.c_[y + rng.normal(0, 3, len(y)), rng.normal(size=(len(y), 20))]
    good = G.ridge_cv_r2(X, y, seeds)
    assert good["passed"] and good["r2"] > 0.5 and len(good["alphas"]) == 5
    bad = G.ridge_cv_r2(rng.normal(size=(len(y), 20)), y, seeds)
    assert bad["r2"] < 0.05
    wide = G.ridge_cv_r2(np.c_[X, rng.normal(size=(len(y), 400))], y, seeds)   # p > n: dual form
    assert wide["r2"] > 0


def test_causality_check_detects_an_acausal_bridge(world):
    setup, rec, _ = world
    counts = np.load(rec / "brain" / A.record_name("rot_right", 60, 0))["dn_counts"]
    br = A.make_bridge(setup, "true", 0)
    m, _ = A.make_model("true", 0, setup, br)
    ok = G.causality(m, br, counts, j=150)
    assert ok["passed"] and ok["first_changed_interval"] >= 151
    lin, _ = A.make_model("linear", 0, setup, br)
    assert G.causality(lin, br, counts, j=150)["passed"]
    from coupling.dn_bridge import BridgeConfig, DNBridge
    acausal = DNBridge(setup.fw_table(), setup.banc_table(),
                       BridgeConfig(families=FAMILIES, latency_intervals=0, acausal=True))
    assert not G.causality(m, acausal, counts, j=150)["passed"]


def test_engine_gate_on_a_subgraph(world):
    setup, _, _ = world
    sub = G.gate_subgraph(setup, A.make_bridge(setup, "true", 0), n=40)
    assert sub.n == 40
    res = G.engine_gradient_and_forward(sub, T_grad=20)
    assert res["passed"], res


def test_resume_gate(world, tmp_path):
    setup, _, _ = world
    br, tr = load(world, "train")

    def make():
        m, info = A.make_model("true", 0, setup, br)
        return m, dict(model=info)
    assert G.resume_check(make, tr, tmp_path, steps=4)["passed"]


def test_shuffle_gate_and_cache(world, tmp_path):
    setup, _, _ = world
    res = G.shuffle_gate(setup, range(2), tmp_path)
    assert res["passed"] and (tmp_path / "shuffled_t0.npz").exists()
    p0, _ = A.shuffled_post(setup, 0, tmp_path)
    p1, _ = A.shuffled_post(setup, 1, None)
    assert not np.array_equal(p0, p1) and not np.array_equal(p0, setup.post)


def test_symmetry_gate_logic():
    fam = ("F1", "F2")
    rows = lambda g: {"F1": dict(gate=True, log2_total=0.1), "F2": dict(gate=g, log2_total=0.7)}   # noqa: E731
    types = [dict(type=t, log2_total=0.1, log2_exc=0.2, **{"pass": True}) for t in "abcdefgh"]
    assert G.symmetry_gate(fam, rows(True), rows(True), types)["passed"]
    r = G.symmetry_gate(fam, rows(True), rows(False), types)
    assert not r["passed"] and r["failing_families"] == ["F2"]
    types[3]["pass"] = False
    assert G.symmetry_gate(fam, rows(True), rows(True), types)["failing_readout_types"] == ["d"]


def test_records_gate(world, tmp_path):
    setup, rec, _ = world
    files = sorted(A.read_manifest(rec / "manifest.sha256"))
    (rec / "plan.json").write_text(json.dumps(dict(records=[dict(file=f) for f in files])))
    (rec / "input_checks.json").write_text(json.dumps({"p0": dict(passed=True)}))
    (rec / "machine.json").write_text(json.dumps(dict(cpu="x")))
    s2c = tmp_path / "s2c.json"
    s2c.write_text(json.dumps(dict(gate5=dict(bit_identical=True))))
    assert G.records_gate(rec, s2c, n_expected=len(files))["passed"]
    f = rec / files[0]
    f.write_bytes(f.read_bytes() + b"x")
    r = G.records_gate(rec, s2c, n_expected=len(files))
    assert not r["passed"] and r["hash_mismatches"] == [files[0]]


# ── guard ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_guard_pauses_and_resumes_from_the_checkpoint(tmp_path, monkeypatch):
    """A job that counts to 6 with a 'checkpoint' file; the GPU is 'hot' at the first poll: the guard kills the job,
    waits until it is cool, and starts the same command again, which resumes from its checkpoint."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_train_guard as GD
    job = tmp_path / "job.py"
    state = tmp_path / "count"
    job.write_text("import sys, time\nfrom pathlib import Path\np = Path(sys.argv[1])\n"
                   "n = int(p.read_text()) if p.exists() else 0\n"
                   "while n < 6:\n    n += 1\n    p.write_text(str(n))\n    time.sleep(0.05)\n")
    temps = iter([50, 90] + [50] * 1000)                       # start ok, hot at the first poll
    monkeypatch.setattr(GD, "SCOPE", False)
    monkeypatch.setattr(GD, "POLL_S", 0.01)
    monkeypatch.setattr(GD, "WAIT_POLL_S", 0.01)
    monkeypatch.setattr(GD, "LOG", tmp_path / "guard.log")
    monkeypatch.setattr(GD, "PAUSE_FILE", tmp_path / "PAUSE")
    monkeypatch.setattr(GD.B, "gpu", lambda: dict(temp=next(temps), used_mib=0, free_gib=7, power_w="1", throttle="0"))
    monkeypatch.setattr(GD.B, "free_ram_gb", lambda: 20.0)
    monkeypatch.setattr(GD.B, "windows_free_gb", lambda: 12.0)
    monkeypatch.setattr(GD.B, "disk_free_gb", lambda: 60.0)
    monkeypatch.setattr(GD.B, "cuda_processes", lambda: [])
    monkeypatch.setattr(GD.B, "nvlddmkm_errors_since", lambda t: 0)

    class Args:
        memory_max, timeout, max_pauses = "1G", 60, 5
        kill_ram_gb, windows_min_free_gb, disk_min_gb = 4.0, 6.0, 25.0
    from datetime import datetime
    rc = GD.run_job([sys.executable, str(job), str(state)], Args, GD.Limits(Args), datetime.now())
    text = (tmp_path / "guard.log").read_text()
    assert rc == 0 and state.read_text() == "6"
    assert "PAUSE #1: GPU temperature 90 C >= 87" in text and text.count(" start (") == 2


def test_guard_limits():
    sys.path.insert(0, str(ROOT / "scripts"))
    import f5_train_guard as GD

    class Args:
        kill_ram_gb, windows_min_free_gb, disk_min_gb = 4.0, 6.0, 25.0
    lim = GD.Limits(Args)
    g = dict(temp=60)
    assert lim.pause_reason(g, 10.0, 10.0, 50.0) is None
    assert "temperature" in lim.pause_reason(dict(temp=87), 10.0, 10.0, 50.0)
    assert "RAM" in lim.pause_reason(g, 3.9, 10.0, 50.0)
    assert "Windows" in lim.pause_reason(g, 10.0, 5.9, 50.0)
    assert "C:" in lim.pause_reason(g, 10.0, 10.0, 24.0)
