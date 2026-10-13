"""Brian2 LIF engine: protocol, equivalence with Shiu's create_model, timing, determinism, controls.

Toy networks run on the numpy code generator (no compilation). Full-brain tests are marked slow."""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("brian2")
from brian2 import mV, ms, prefs  # noqa: E402

from conftest import MODEL_PY, ROOT  # noqa: E402
from connectome import controls as C  # noqa: E402
from connectome.schema import Connectome  # noqa: E402
from engine import lif_brian2 as L  # noqa: E402
from engine.protocol import NeuralEngine  # noqa: E402
from toy import edges_table, neurons_table  # noqa: E402

needs_model = pytest.mark.skipif(not MODEL_PY.exists(), reason="brain_model/model.py missing (fetch_data.py)")
pytestmark = needs_model


@pytest.fixture(autouse=True, scope="module")
def numpy_target():
    old = prefs.codegen.target
    prefs.codegen.target = "numpy"
    yield
    prefs.codegen.target = old


def random_cx(n=40, m=260, seed=0, scale=1, lo=15, hi=40):
    """Random signed graph: unique pairs, sign a function of the presynaptic neuron (70 % excitatory)."""
    rng = np.random.default_rng(seed)
    pair = sorted({(int(a), int(b)) for a, b in zip(rng.integers(0, n, m), rng.integers(0, n, m)) if a != b})
    pre, post = np.array([p for p, _ in pair]), np.array([q for _, q in pair])
    sign_of = np.where(rng.random(n) < 0.7, 1, -1)
    nrn = neurons_table([dict(id=500 + i, cell_type="T", side="left") for i in range(n)])
    e = pd.DataFrame({"pre": 500 + pre, "post": 500 + post,
                      "n_syn": rng.integers(lo, hi, len(pre)) * scale, "sign": sign_of[pre]})
    return Connectome("toy", nrn, e)


def make_engine(cx=None, **kw):
    cx = random_cx() if cx is None else cx
    kw.setdefault("input_groups", {"inp": np.arange(8)})
    kw.setdefault("groups", {"out": np.arange(30, 40)})
    return L.Brian2LIFEngine(cx, **kw)


def delay_ms(syn):
    """Homogeneous synaptic delay in ms (scalar variable)."""
    return float(np.ravel(np.asarray(syn.delay[:]))[0]) * 1e3


def spikes_of(eng):
    return np.asarray(eng.spk_mon.i[:]).copy(), np.asarray(eng.spk_mon.t_[:]).copy()


def test_implements_protocol_and_rejects_bad_input():
    eng = make_engine()
    assert isinstance(eng, NeuralEngine)
    with pytest.raises(KeyError):
        eng.set_inputs({"nope": 10})
    with pytest.raises(ValueError, match="negative"):
        eng.set_inputs({"inp": -1})
    with pytest.raises(ValueError, match="rates for"):
        eng.set_inputs({"inp": np.ones(3)})
    with pytest.raises(ValueError, match="multiple"):
        eng.advance(0.25)
    with pytest.raises(ValueError, match="more than one"):
        make_engine(input_groups={"a": [0, 1], "b": [1, 2]})
    with pytest.raises(ValueError, match="signed"):
        L.Brian2LIFEngine(Connectome("u", random_cx().neurons, random_cx().edges.drop(columns="sign")))


def test_build_equals_shiu_create_model(tmp_path):
    """Neuron model, parameters, synapse list and weights are those of model.py:create_model; with the same
    deterministic initial kicks the spike trains are identical (no RNG involved)."""
    from brian2 import Network, defaultclock
    model, _ = L.load_shiu_model(MODEL_PY)
    cx = random_cx(lo=150, hi=300)          # strong synapses: the initial kicks propagate through the graph
    pre, post = cx.edge_endpoints()
    pd.DataFrame({"Completed": True}, index=pd.Index(cx.neurons["id"].to_numpy())).to_csv(tmp_path / "comp.csv")
    pd.DataFrame({"Presynaptic_Index": pre, "Postsynaptic_Index": post,
                  "Excitatory x Connectivity": cx.signed_weights()}).to_parquet(tmp_path / "con.parquet")
    defaultclock.dt = 0.1 * ms
    neu_ref, syn_ref, mon_ref = model.create_model(str(tmp_path / "comp.csv"), str(tmp_path / "con.parquet"),
                                                   dict(model.default_params))
    net_ref = Network(neu_ref, syn_ref, mon_ref)
    eng = L.Brian2LIFEngine(cx)

    np.testing.assert_array_equal(np.asarray(eng.syn.i[:]), np.asarray(syn_ref.i[:]))
    np.testing.assert_array_equal(np.asarray(eng.syn.j[:]), np.asarray(syn_ref.j[:]))
    np.testing.assert_array_equal(np.asarray(eng.syn.w_[:]), np.asarray(syn_ref.w_[:]))
    np.testing.assert_array_equal(np.asarray(eng.neu.rfc_[:]), np.asarray(neu_ref.rfc_[:]))
    np.testing.assert_array_equal(np.asarray(eng.neu.v_[:]), np.asarray(neu_ref.v_[:]))
    assert delay_ms(eng.syn) == pytest.approx(1.8)

    kick = np.arange(6)
    eng.neu.v[kick] = -40 * mV
    neu_ref.v[kick] = -40 * mV
    eng.net.run(40 * ms)
    net_ref.run(40 * ms)
    i_a, t_a = spikes_of(eng)
    assert len(i_a) > 12 and len(set(i_a.tolist()) - set(kick.tolist())) > 0
    np.testing.assert_array_equal(i_a, np.asarray(mon_ref.i[:]))
    np.testing.assert_allclose(t_a, np.asarray(mon_ref.t_[:]), atol=1e-12)


def test_poisson_path_has_no_delay_and_targets_have_no_refractory_period():
    eng = make_engine()
    for s in eng.poisson_synapses.values():
        assert np.all(np.asarray(s.delay[:]) == 0)           # M4
    rfc = np.asarray(eng.neu.rfc_[:])
    assert np.all(rfc[:8] == 0)
    assert np.allclose(rfc[8:], 2.2e-3)                       # model.py t_rfc
    assert delay_ms(eng.syn) == pytest.approx(1.8)  # recurrent delay unchanged


def test_quiet_logging_raises_brian_handlers():
    import logging
    from brian2.utils.logger import BrianLogger
    make_engine()
    for h in (BrianLogger.file_handler, BrianLogger.console_handler):
        assert h is None or h.level >= logging.WARNING


def test_interval_comes_from_the_caller_and_time_advances_exactly():
    eng = make_engine()
    eng.advance(5)
    eng.advance(2.5)
    assert eng.t_ms == pytest.approx(7.5)
    assert float(eng.net.t / ms) == pytest.approx(7.5)
    assert eng.state_dict()["steps"] == 75


def test_input_drives_spikes_and_rates_window():
    eng = make_engine()
    assert eng.rates(["inp"], 10)["inp"].sum() == 0           # t = 0: nothing yet
    eng.set_inputs({"inp": 150.0})
    for _ in range(20):
        eng.advance(5)
    r = eng.rates(["inp", "out", "all"], 50)
    assert r["inp"].shape == (8,) and r["all"].shape == (40,)
    assert 50 < r["inp"].mean() < 400                         # each Poisson kick forces a spike (~150 Hz)
    assert eng.spike_counts().sum() > 0
    with pytest.raises(ValueError, match="history"):
        eng.rates(["inp"], 500)
    # a window longer than the elapsed time uses the elapsed time
    e2 = make_engine()
    e2.set_inputs({"inp": 150.0})
    e2.advance(10)
    assert e2.rates(["inp"], 100)["inp"].mean() == pytest.approx(
        e2.spike_counts()[:8].sum() / 8 / 0.010)


def test_rates_use_only_recent_spikes():
    eng = make_engine(history_ms=100.0)
    eng.set_inputs({"inp": 300.0})
    for _ in range(10):
        eng.advance(5)
    eng.set_inputs({"inp": 0.0})
    for _ in range(25):
        eng.advance(5)
    quiet = eng.rates(["inp"], 20)["inp"]
    assert quiet.max() < 150                                   # inputs off: only a trace of recurrent activity
    assert len(eng._hist) <= 100 / 5 + 1                      # history is trimmed


def test_reset_replays_and_seed_changes_noise():
    eng = make_engine()

    def run(seed):
        eng.reset(seed)
        eng.set_inputs({"inp": 200.0})
        for _ in range(12):
            eng.advance(5)
        return spikes_of(eng)

    a, b, c = run(7), run(7), run(8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not (len(a[0]) == len(c[0]) and np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))
    assert eng.state_dict()["seed"] == 8 and eng.state_dict()["input_rates"]["inp"] == 200.0
    eng.reset(1)
    assert eng.t_ms == 0.0 and eng.state_dict()["input_rates"]["inp"] == 0.0 and eng.spike_counts().sum() == 0


def test_state_dict_audit_fields():
    eng = make_engine()
    s = eng.state_dict()
    assert s["model_sha256"] == L.SHIU_MODEL_SHA256 and len(s["weights_sha256"]) == 64
    assert s["params"]["w_syn"] == pytest.approx(0.275e-3) and s["params"]["f_poi"] == 250
    assert s["n_neurons"] == 40 and s["dev_subnet"] is False and s["control"] is None
    d = eng.state_dict(include_dynamics=True)
    assert d["v"].shape == (40,) and d["g"].shape == (40,)
    eng.set_weight_scale(0.5)
    assert eng.state_dict()["weights_sha256"] != s["weights_sha256"] and eng.state_dict()["weight_scale"] == 0.5


def test_per_post_weight_scale_variant():
    eng = make_engine()
    w0 = np.asarray(eng.syn.w_[:]).copy()
    post = np.asarray(eng.syn.j[:])
    f = np.where(np.arange(40) % 2 == 0, 2.0, 0.5)
    with pytest.raises(ValueError):
        eng.set_weight_scale_per_post(f[:10])
    eng.set_weight_scale_per_post(f)
    np.testing.assert_allclose(np.asarray(eng.syn.w_[:]), w0 * f[post])
    s = eng.state_dict()
    assert s["post_scale_sha256"] and len(s["post_scale_sha256"]) == 64
    w_in = [np.asarray(v.w_[:]).copy() for v in eng.poisson_synapses.values()]
    eng.set_weight_scale(1.0)          # the global scale replaces it; the record forgets the per-post factors
    assert eng.state_dict()["post_scale_sha256"] is None
    for v, w in zip(eng.poisson_synapses.values(), w_in):
        np.testing.assert_array_equal(np.asarray(v.w_[:]), w)


def test_model_hash_is_enforced(tmp_path):
    bad = tmp_path / "model.py"
    bad.write_text(MODEL_PY.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="pinned"):
        L.load_shiu_model(bad)


def test_sign_rules_reach_the_weights_without_touching_the_engine():
    from connectome import sign_rules as SR
    cx = random_cx()
    silent = SR.apply_sign_rules(cx, [C.ablation_rule([0, 1, 2])])
    e = L.Brian2LIFEngine(silent)
    pre = np.asarray(e.syn.i[:])
    w = np.asarray(e.syn.w_[:])
    assert np.all(w[np.isin(pre, [0, 1, 2])] == 0) and np.any(w[~np.isin(pre, [0, 1, 2])] != 0)
    assert len(e.syn) == len(cx.edges)                         # same synapse list
    assert e.state_dict()["sign_rules"][0]["name"] == "ablation"


def test_dev_subnet_spikes_use_global_indices_and_warn():
    cx = random_cx(n=60, m=500)
    sub = C.dev_subnet(cx, core=[0, 1, 2], min_syn=1)
    got = []
    with pytest.warns(UserWarning, match="DEV subnet"):
        eng = L.Brian2LIFEngine(sub, input_groups={"inp": np.arange(2)},
                                spike_sink=lambda t, i: got.append(i.copy()))
    eng.set_inputs({"inp": 300.0})
    for _ in range(10):
        eng.advance(5)
    spk = np.concatenate(got)
    assert len(spk) == eng.spike_counts().sum() > 0
    assert np.isin(spk, sub.meta["global_index"]).all()
    assert eng.state_dict()["dev_subnet"] is True


def test_dev_plus_control_is_not_silent():
    """L6: v1 ignored shuffle_seed on the DEV subnet without a word."""
    cx = random_cx(n=60, m=500)
    sub = C.shuffle_targets(C.dev_subnet(cx, core=[0, 1], min_syn=1), seed=0)
    with pytest.warns(UserWarning, match="not a valid null control"):
        L.Brian2LIFEngine(sub)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        L.Brian2LIFEngine(cx)                                  # intact full graph: no DEV / control warning
    assert not [x for x in w if "DEV" in str(x.message) or "control" in str(x.message)]


def test_activity_matched_scale_recovers_the_weight_level():
    """A control whose weights are 3x too strong is brought back to the reference rate by one global scale."""
    ref = make_engine(random_cx(seed=3))
    drive = {"inp": 120.0}
    kw = dict(duration_ms=200.0, discard_ms=50.0, interval_ms=5.0, seed=0)
    target = L.measure_rate(ref, drive, **kw)
    strong = make_engine(random_cx(seed=3, scale=3))
    assert L.measure_rate(strong, drive, **kw) > 1.3 * target     # the control is hotter than the reference
    res = L.match_weight_scale(strong, target, drive, rel_tol=0.1, **kw)
    assert res["converged"] and abs(res["rate_hz"] - target) <= 0.1 * target
    assert 0.1 < res["scale"] < 0.8
    assert strong.state_dict()["weight_scale"] == pytest.approx(res["scale"])
    with pytest.raises(RuntimeError, match="bracketed"):
        L.match_weight_scale(ref, 1e5, drive, hi=1.0, **kw)


@pytest.mark.slow
def test_full_brain_matches_v1_totals():
    """FlyWire v783: 138,639 neurons, 15,091,983 synapses, signed weight sum, LIF parameters untouched."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    eng = L.Brian2LIFEngine(cx, input_groups={"sugar": cx.by_type(["LB3"])[:20]}, seed=0)
    assert eng.n_neurons == 138_639 and len(eng.syn) == 15_091_983
    w = np.asarray(eng.syn.w_[:])
    np.testing.assert_allclose(w.sum(), 10_496_512 * float(eng.params["w_syn"]), rtol=1e-9)
    p = eng.params
    assert (float(p["v_0"] / mV), float(p["v_th"] / mV), float(p["v_rst"] / mV)) == pytest.approx((-52, -45, -52))
    assert float(p["t_mbr"] / ms) == pytest.approx(20) and float(p["tau"] / ms) == pytest.approx(5)
    assert float(p["t_rfc"] / ms) == pytest.approx(2.2) and float(p["t_dly"] / ms) == pytest.approx(1.8)
    eng.set_inputs({"sugar": 100.0})
    eng.advance(25)
    assert eng.spike_counts()[eng.groups["sugar"]].sum() > 0


def test_weight_scale_after_a_run_stores_the_pristine_state():
    """Regression (found in the F2 calibration): set_weight_scale after a run must not store the dynamic state of
    that run as the new initial state (silent engine with no input fired after the scale change)."""
    eng = make_engine()
    first = next(iter(eng.poisson_synapses))
    eng.reset(1)
    eng.set_inputs({first: 300.0})
    eng.advance(100.0)
    assert eng.spike_counts().sum() > 0
    eng.set_weight_scale(2.0)
    eng.reset(2)
    eng.advance(100.0)
    assert eng.spike_counts().sum() == 0 and float(np.abs(np.asarray(eng.neu.g_[:])).max()) == 0.0
    eng.set_inputs({first: 300.0})
    eng.advance(50.0)
    eng.set_weight_scale_per_post(np.full(40, 1.5))
    eng.reset(3)
    eng.advance(100.0)
    assert eng.spike_counts().sum() == 0
