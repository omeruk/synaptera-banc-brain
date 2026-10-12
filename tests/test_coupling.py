"""Causal coupling: command queue, latency, single-run vs per-interval equivalence, acausal refusal."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("brian2")
from brian2 import prefs  # noqa: E402

from conftest import MODEL_PY  # noqa: E402
from connectome.schema import Connectome  # noqa: E402
from coupling.scheduler import CommandQueue, CoupledLoop, LoopConfig  # noqa: E402
from engine import lif_brian2 as L  # noqa: E402
from toy import neurons_table  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PY.exists(), reason="brain_model/model.py missing")


@pytest.fixture(autouse=True, scope="module")
def numpy_target():
    old = prefs.codegen.target
    prefs.codegen.target = "numpy"
    yield
    prefs.codegen.target = old


def toy_cx(n=40, m=260, seed=0):
    rng = np.random.default_rng(seed)
    pair = sorted({(int(a), int(b)) for a, b in zip(rng.integers(0, n, m), rng.integers(0, n, m)) if a != b})
    pre, post = np.array([p for p, _ in pair]), np.array([q for _, q in pair])
    sign_of = np.where(rng.random(n) < 0.7, 1, -1)
    e = pd.DataFrame({"pre": 500 + pre, "post": 500 + post, "n_syn": rng.integers(15, 40, len(pre)),
                      "sign": sign_of[pre]})
    return Connectome("toy", neurons_table([dict(id=500 + i, cell_type="T", side="left") for i in range(n)]), e)


class ToyPlant:
    """Records, per interval, the command it was given and the engine input it set."""

    def __init__(self):
        self.applied, self.sensed = [], []

    def sense(self, k):
        self.sensed.append(k)
        return {"inp": 150.0}

    def act(self, k, command, dt_s):
        self.applied.append(command)
        return dict(k=k, command=command, dt=dt_s)


class ToyDecider:
    """Command = the total spikes of the whole network in interval k (read from the engine history)."""

    neutral = -1

    def __init__(self, dt_ms):
        self.dt_ms = dt_ms
        self.record = None

    def decide(self, k, engine):
        n = int(round(engine.rates(["all"], self.dt_ms)["all"].sum() * self.dt_ms * 1e-3))
        self.record = dict(k=k, spikes=n)
        return n


def make_loop(mode, latency, allow=False, seed=0):
    eng = L.Brian2LIFEngine(toy_cx(), input_groups={"inp": np.arange(8)}, seed=seed)
    cfg = LoopConfig(interval_ms=5.0, latency_intervals=latency, mode=mode, allow_acausal=allow)
    plant, dec, recs = ToyPlant(), ToyDecider(5.0), []
    loop = CoupledLoop(eng, plant, dec, cfg, on_interval=lambda k, r: recs.append((k, r)))
    return eng, loop, plant, recs


def test_command_queue_latency_and_neutral():
    q = CommandQueue(2, "neutral")
    assert q.pop(0) == ("neutral", -1)
    q.push(0, "c0")
    assert q.pop(1) == ("neutral", -1)
    q.push(1, "c1")
    q.push(2, "c2")
    assert q.pop(2) == ("c0", 0) and q.pop(3) == ("c1", 1)
    with pytest.raises(ValueError):
        CommandQueue(-1, None)


def test_latency_zero_is_refused_unless_labelled():
    with pytest.raises(ValueError, match="Y1"):
        LoopConfig(latency_intervals=0)
    c = LoopConfig(latency_intervals=0, allow_acausal=True).record()
    assert c["causality"].startswith("ACAUSAL") and c["latency_intervals"] == 0
    assert "causal" in LoopConfig().record()["causality"] and LoopConfig().latency_intervals == 1


@pytest.mark.parametrize("mode", ["single_run", "per_interval"])
def test_body_is_one_interval_behind_the_engine(mode):
    eng, loop, plant, recs = make_loop(mode, latency=1)
    loop.run(12)
    assert [k for k, _ in recs] == list(range(12))
    # interval 0 gets the neutral command; interval k gets the output decided at interval k - 1
    assert recs[0][1]["cmd_source"] == -1 and plant.applied[0] == -1
    decided = [r["decided"]["spikes"] for _, r in recs]
    for k in range(1, 12):
        assert recs[k][1]["cmd_source"] == k - 1
        assert plant.applied[k] == decided[k - 1]
    assert plant.sensed == list(range(12))                 # input for interval j set at boundary j
    assert eng.t_ms == pytest.approx(60.0) and loop.n_done == 12


def test_acausal_reproduction_uses_the_same_interval():
    _, loop, plant, recs = make_loop("single_run", latency=0, allow=True)
    loop.run(8)
    decided = [r["decided"]["spikes"] for _, r in recs]
    assert plant.applied == decided and all(r["cmd_source"] == k for k, r in recs)


def test_single_run_and_per_interval_are_equivalent():
    e1, l1, p1, r1 = make_loop("single_run", 1)
    e2, l2, p2, r2 = make_loop("per_interval", 1)
    e1.reset(7)                      # Brian2 draws from one global RNG: reset right before each run
    l1.run(20)
    e2.reset(7)
    l2.run(20)
    assert p1.applied == p2.applied
    np.testing.assert_array_equal(e1.spike_counts(), e2.spike_counts())
    assert e1.spike_counts().sum() > 0


def test_inputs_set_at_a_boundary_act_in_the_next_interval_only():
    eng = L.Brian2LIFEngine(toy_cx(), input_groups={"inp": np.arange(8)}, seed=1)
    seen = []

    class Switch:
        def sense(self, k):
            return {"inp": 0.0 if k < 5 else 400.0}

        def act(self, k, command, dt_s):
            return {}

    class Dec:
        neutral = 0
        record = None

        def decide(self, k, engine):
            seen.append(float(engine.rates(["inp"], 5.0)["inp"].sum()))
            return 0

    CoupledLoop(eng, Switch(), Dec(), LoopConfig(5.0, 1)).run(10)
    assert max(seen[:5]) == 0.0                    # intervals 0-4 had no drive
    assert min(seen[5:]) > 0.0                     # drive set at boundary 5 acts in interval 5 (not before)


# ── recorder and replay ──────────────────────────────────────────────────────
import h5py  # noqa: E402

from coupling.recorder import FieldSpec, Recorder  # noqa: E402
from coupling.replay import replay  # noqa: E402


def _recorded_run(tmp_path, name="run.h5", n=10, seed=3):
    cx = toy_cx()
    rec = Recorder(tmp_path / name, fields=(FieldSpec("cmd", (), np.float64, "BRAIN"),
                                            FieldSpec("cmd_source", (), np.int32, "STATE")),
                   meta=dict(interval_ms=5.0, seed=seed, latency_intervals=1), n_intervals=n,
                   input_groups={"inp": 8})
    eng = L.Brian2LIFEngine(cx, input_groups={"inp": np.arange(8)}, seed=seed, spike_sink=rec.spike_sink)

    class P:
        def sense(self, k):
            d = {"inp": np.full(8, 100.0 + 20 * k)}
            rec.put_inputs(k, d)
            return d

        def act(self, k, command, dt_s):
            return {}

    class D:
        neutral = 0.0
        record = None

        def decide(self, k, engine):
            return float(k)

    loop = CoupledLoop(eng, P(), D(), LoopConfig(5.0, 1),
                       on_interval=lambda k, r: rec.put(k, cmd=r["cmd_source"], cmd_source=r["cmd_source"]))
    eng.reset(seed)
    loop.run(n)
    rec.close()
    return cx, rec


def test_recorder_labels_streams_and_protects_results(tmp_path):
    cx, rec = _recorded_run(tmp_path)
    with h5py.File(rec.path) as f:
        assert f["behavior/cmd"].attrs["label"] == "BRAIN" and f["behavior/cmd_source"].attrs["label"] == "STATE"
        assert f["behavior/cmd_source"][:].tolist() == [-1] + list(range(9))
        assert f.attrs["complete"] and f["spikes/t"].shape[0] == f["spikes/i"].shape[0] > 0
        assert f["inputs/inp"].shape == (10, 8)
        assert json_meta(f, "latency_intervals") == 1 and f["meta"].attrs["overwrite"] == False  # noqa: E712
    assert not rec.part.exists()
    with pytest.raises(FileExistsError, match="never overwritten"):
        Recorder(rec.path)
    r2 = Recorder(rec.path, overwrite=True)                 # explicit, and recorded
    r2.close()
    with h5py.File(rec.path) as f:
        assert f["meta"].attrs["overwrite"] == True  # noqa: E712


def json_meta(f, key):
    return f["meta"].attrs[key]


def test_crash_leaves_only_a_part_file(tmp_path):
    with pytest.raises(RuntimeError):
        with Recorder(tmp_path / "crash.h5", fields=(FieldSpec("x"),), n_intervals=3) as r:
            r.put(0, x=1.0)
            raise RuntimeError("boom")
    assert not (tmp_path / "crash.h5").exists() and (tmp_path / "crash.h5.part").exists()


def test_unknown_field_and_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="label"):
        FieldSpec("x", label="MAGIC")
    with Recorder(tmp_path / "a.h5", fields=(FieldSpec("x"),), n_intervals=2) as r:
        with pytest.raises(KeyError):
            r.put(0, y=1.0)
        r.put(0, x=1.0)


def test_replay_reproduces_the_record_bit_exactly(tmp_path):
    cx, rec = _recorded_run(tmp_path)
    fresh = L.Brian2LIFEngine(cx, input_groups={"inp": np.arange(8)}, seed=0)
    rep = replay(fresh, rec.path)
    assert rep["identical"] and rep["n_spikes_record"] == rep["n_spikes_replay"] > 0 and rep["spikes_differing"] == 0
    other = L.Brian2LIFEngine(cx, input_groups={"inp": np.arange(8)}, seed=0)
    bad = replay(other, rec.path, seed=99)                  # a different noise stream is a different record
    assert not bad["identical"]
    changed = L.Brian2LIFEngine(toy_cx(seed=5), input_groups={"inp": np.arange(8)}, seed=0)
    assert not replay(changed, rec.path)["identical"]       # a different graph is detected
