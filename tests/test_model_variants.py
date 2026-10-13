"""Neuron model v2 (engine/model_variants.py): the named variants adapt / cond / gap and the arms R, A, AC, ACG.

Toy networks on the numpy code generator; no data files except the pinned brain_model/model.py."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("brian2")
from brian2 import mV, ms, prefs  # noqa: E402

from conftest import MODEL_PY, ROOT  # noqa: E402
from connectome.schema import Connectome  # noqa: E402
from engine import lif_brian2 as L  # noqa: E402
from engine import model_variants as MV  # noqa: E402
from test_engine_lif import make_engine, random_cx, spikes_of  # noqa: E402
from toy import neurons_table  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PY.exists(), reason="brain_model/model.py missing (fetch_data.py)")


@pytest.fixture(autouse=True, scope="module")
def numpy_target():
    old = prefs.codegen.target
    prefs.codegen.target = "numpy"
    yield
    prefs.codegen.target = old


def params():
    return L.load_shiu_model(MODEL_PY)[0].default_params


def drive(eng, rate=150.0, n=20, seed=3):
    eng.reset(seed)
    eng.set_inputs({"inp": rate})
    for _ in range(n):
        eng.advance(5)
    return spikes_of(eng)


def chain_cx(n_syn=1, sign=1, extra=()):
    """0 -> 1 with n_syn synapses of the given sign, plus extra (pre, post, n, sign) rows."""
    nrn = neurons_table([dict(id=i, cell_type=t, side="left") for i, t in enumerate(["P", "Q", "R", "S"])])
    rows = [(0, 1, n_syn, sign)] + list(extra)
    e = pd.DataFrame(rows, columns=["pre", "post", "n_syn", "sign"])
    return Connectome("chain", nrn, e)


def test_arms_and_names():
    assert MV.ARMS == {"R": (), "A": ("adapt",), "AC": ("adapt", "cond"), "ACG": ("adapt", "cond", "gap")}
    for k in ("A", "AC", "ACG"):
        assert MV.ModelVariant.arm(k).name == k
    with pytest.raises(ValueError):
        MV.ModelVariant.arm("R")
    # gap pairs are kept only when the arm has the gap switch
    assert MV.ModelVariant.arm("AC", gap_pairs=[(0, 1)]).gap_pairs == ()
    assert MV.ModelVariant.arm("ACG", gap_pairs=[(0, 1)]).gap_pairs == ((0, 1),)


def test_registered_constants_follow_their_sources():
    assert MV.ADAPT_A == pytest.approx(4.0 / 30.0)                 # Brette & Gerstner 2005 Table 1: a / g_L
    assert MV.ADAPT_B_V == pytest.approx(0.0805e-9 / 30e-9)        # b / g_L = 2.683 mV
    assert MV.ADAPT_TAU_W_S == 0.144
    assert MV.COND_E_EXC_V == 0.0 and MV.COND_E_INH_V == -0.074
    p = params()
    assert MV.GAP_DV_V == pytest.approx(float(p["w_syn"]) * float(p["f_poi"]))


def test_conductance_calibration_reproduces_the_registered_values():
    p = params()
    target = MV.shiu_psp_peak(p)
    for e, dg in ((MV.COND_E_EXC_V, MV.COND_DG_EXC), (MV.COND_E_INH_V, MV.COND_DG_INH)):
        assert MV.cond_psp_peak(dg, e, p) == pytest.approx(target, rel=1e-6)
        assert MV.calibrate_dg(p, e, tol=1e-9) == pytest.approx(dg, rel=1e-6)
    # small-PSP limit: dg ~ w_syn / |E - v_0|
    assert MV.COND_DG_EXC == pytest.approx(0.275 / 52, rel=0.01)
    assert MV.COND_DG_INH == pytest.approx(0.275 / 22, rel=0.01)


def _psp(model_variant, sign):
    from brian2 import NeuronGroup  # noqa: F401
    eng = L.Brian2LIFEngine(chain_cx(1, sign), model_variant=model_variant)
    eng.neu.v[0] = -40 * mV                                 # one presynaptic spike, no noise
    v = []
    for _ in range(300):
        eng.advance(0.1)
        v.append(float(eng.neu.v_[1]))
    return np.max(np.abs(np.array(v) - (-0.052)))


@pytest.mark.parametrize("sign", [1, -1])
def test_single_synapse_psp_at_rest_equals_shiu(sign):
    ref = _psp(None, sign)
    assert ref == pytest.approx(MV.shiu_psp_peak(params()), rel=0.02)
    assert _psp(MV.ModelVariant.arm("AC"), sign) == pytest.approx(ref, rel=0.02)


def test_reference_path_is_unchanged_when_no_variant():
    e1, e2 = make_engine(), make_engine(model_variant=None)
    assert e1.state_dict()["weights_sha256"] == e2.state_dict()["weights_sha256"]
    assert e1.state_dict()["model_variant"] is None
    i1, t1 = drive(e1)
    i2, t2 = drive(e2)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(t1, t2)


def test_adapt_with_zero_adaptation_equals_reference(monkeypatch):
    monkeypatch.setattr(MV, "ADAPT_A", 0.0)
    monkeypatch.setattr(MV, "ADAPT_B_V", 0.0)
    cx = random_cx(lo=15, hi=40)
    i0, t0 = drive(make_engine(cx))
    i1, t1 = drive(make_engine(cx, model_variant=MV.ModelVariant.arm("A")))
    assert len(i0) > 50
    np.testing.assert_array_equal(i0, i1)
    np.testing.assert_allclose(t0, t1, atol=1e-12)


def test_adaptation_lowers_the_rate_of_a_driven_neuron():
    # one neuron held above threshold by a strong constant excitatory input through recurrent kicks is not possible
    # in a toy graph, so test the adaptation variable directly: after spikes w_ad > 0 and grows with the spike count
    eng = make_engine(model_variant=MV.ModelVariant.arm("A"))
    drive(eng, n=40)
    w = np.asarray(eng.neu.w_ad_[:])
    cnt = eng.spike_counts()
    assert w[cnt > 0].min() > 0
    assert np.corrcoef(cnt[8:], w[8:])[0, 1] > 0.5
    assert eng.state_dict(include_dynamics=True)["w_ad"].shape == (eng.n_neurons,)


def test_adaptation_slows_firing_under_the_same_input():
    cx = random_cx(lo=60, hi=120)
    n0 = len(drive(make_engine(cx), n=60)[0])
    n1 = len(drive(make_engine(cx, model_variant=MV.ModelVariant.arm("A")), n=60)[0])
    assert n1 < n0


def test_cond_splits_synapses_by_sign_and_records_the_variant():
    cx = random_cx()
    eng = make_engine(cx, model_variant=MV.ModelVariant.arm("AC"))
    w = cx.signed_weights()
    assert len(eng._syns) == 2
    assert len(eng._syns[0]) == int((w > 0).sum()) and len(eng._syns[1]) == int((w < 0).sum())
    np.testing.assert_allclose(np.asarray(eng._syns[0].w_[:]).sum(), w[w > 0].sum() * MV.COND_DG_EXC)
    np.testing.assert_allclose(np.asarray(eng._syns[1].w_[:]).sum(), -w[w < 0].sum() * MV.COND_DG_INH)
    sd = eng.state_dict()
    assert sd["model_variant"]["name"] == "AC" and sd["model_variant"]["cond"]
    # the Poisson input path is unchanged
    p = params()
    for s in eng.poisson_synapses.values():
        assert np.allclose(np.asarray(s.w_[:]), float(p["w_syn"]) * float(p["f_poi"]))
        assert np.all(np.asarray(s.delay[:]) == 0)
    # weight scale applies to both conductance synapse sets
    eng.set_weight_scale(2.0)
    np.testing.assert_allclose(np.asarray(eng._syns[1].w_[:]).sum(), -2 * w[w < 0].sum() * MV.COND_DG_INH)


def test_conductance_inhibition_saturates_at_the_reversal_potential():
    # 1000 inhibitory synapses at once: Shiu current synapses push v far below E_inh, conductance synapses cannot
    def trough(mv):
        eng = L.Brian2LIFEngine(chain_cx(1000, -1), model_variant=mv)
        eng.neu.v[0] = -40 * mV
        v = []
        for _ in range(300):
            eng.advance(0.1)
            v.append(float(eng.neu.v_[1]))
        return min(v)
    assert trough(None) < -0.074
    assert trough(MV.ModelVariant.arm("AC")) > -0.074


def test_gap_junction_is_fast_one_way_and_only_on_registered_pairs():
    cx = chain_cx(1, 1, extra=[(2, 3, 1, 1)])              # 0->1 and 2->3 chemical, one synapse each (subthreshold)
    mv = MV.ModelVariant.arm("ACG", gap_pairs=[(0, 1)])
    eng = L.Brian2LIFEngine(cx, model_variant=mv)
    eng.neu.v[0] = -40 * mV
    eng.neu.v[2] = -40 * mV
    eng.advance(3.0)
    i, t = spikes_of(eng)
    t1 = t[i == 1]
    assert len(t1) == 1 and t1[0] - t[i == 0][0] < 1e-3   # < 1 ms after the GF-like spike
    assert 3 not in i                                       # the unregistered pair transmits only chemically (subthreshold)
    # rectifying: a spike of the postsynaptic cell does not reach the presynaptic one
    eng2 = L.Brian2LIFEngine(cx, model_variant=mv)
    eng2.neu.v[1] = -40 * mV
    eng2.advance(3.0)
    i2, _ = spikes_of(eng2)
    assert 0 not in i2
    # AC (no gap switch): no electrical transmission
    eng3 = L.Brian2LIFEngine(cx, model_variant=MV.ModelVariant.arm("AC", gap_pairs=[(0, 1)]))
    eng3.neu.v[0] = -40 * mV
    eng3.advance(3.0)
    assert 1 not in spikes_of(eng3)[0]


def test_gap_pairs_follow_the_registered_rule():
    cfg = MV.load_gap_config()
    assert [j["name"] for j in cfg["used"]] == ["GF-TTMn", "GF-PSI"]
    assert all(j["rectifying"] and j["evidence"] for j in cfg["used"])
    assert cfg["pairing"]["min_chemical_synapses"] == 1
    nrn = neurons_table([dict(id=0, cell_type="DNp01", side="left"), dict(id=1, cell_type="DNp01", side="right"),
                         dict(id=2, cell_type="TTMn", side="left"), dict(id=3, cell_type="TTMn", side="right"),
                         dict(id=4, cell_type="PSI", side="left"), dict(id=5, cell_type="X", side="left")])
    e = pd.DataFrame([(0, 2, 5, 1), (1, 3, 12, 1), (0, 4, 3, 1), (1, 4, 4, 1), (0, 5, 9, 1)],
                     columns=["pre", "post", "n_syn", "sign"])
    pairs = MV.gap_pairs(Connectome("t", nrn, e), cfg)
    assert pairs == [(0, 2), (0, 4), (1, 3), (1, 4)]
    # a dataset without the types (e.g. the FlyWire brain) gives no pair
    assert MV.gap_pairs(random_cx(), cfg) == []


def test_assumptions_register_the_v2_constants():
    import json
    reg = json.loads((ROOT / "configs" / "assumptions.json").read_text())["constants"]
    for k in ("model_v2.adapt_a", "model_v2.adapt_b_v", "model_v2.adapt_tau_w_s", "model_v2.e_exc_v",
              "model_v2.e_inh_v", "model_v2.dg_exc", "model_v2.dg_inh", "model_v2.gap_dv_v"):
        assert k in reg
