"""Readouts (symmetry gate, crosswalk names) and the hand-made VNC bridge (control arm)."""
import json

import numpy as np
import pytest

from connectome.schema import Connectome
from conftest import ROOT
from motor import legacy_bridge as LB
from motor.legacy_bridge import K_STEER, MN9_THRESHOLD_HZ, READOUT_TAU, Bridge, BaselineMissing
from motor.readouts import READOUT_TYPES, GateFailure, Readouts
from toy import edges_table, neurons_table

DT = 0.025
NT = len(READOUT_TYPES)


def toy_readouts(asym_p15=1.66, asym_p10=0.24):
    """Every readout type has one neuron per side with 100 input synapses on the left; the right side gets
    100 * ratio (DNp15 and DNp10 asymmetric, as in FlyWire v783)."""
    rows, tr = [], []
    nid = 1000
    src = 1
    rows.append(dict(id=src, cell_type="SRC", side="left"))
    for t in READOUT_TYPES:
        name = "CB0701" if t == "MN9" else t
        ratio = {"DNp15": asym_p15, "DNp10": asym_p10}.get(t, 1.0)
        for side, w in (("left", 100), ("right", int(round(100 * ratio)))):
            rows.append(dict(id=nid, cell_type=name, side=side, super_class="descending"))
            tr.append((src, nid, w))
            nid += 1
    cx = Connectome("toy", neurons_table(rows), edges_table(tr))
    return cx


@pytest.fixture(scope="module")
def ro():
    return Readouts(toy_readouts(), "flywire783")


def test_sets_found_by_name_one_per_side_with_alias(ro):
    assert ro.n.tolist() == [[1, 1]] * NT
    assert ro.cx.neurons.set_index("id").loc[ro.cx.index.ids_of(ro.sets["MN9"]["L"]), "cell_type"].tolist() == ["CB0701"]
    with pytest.raises(ValueError, match="missing side"):
        bad = toy_readouts()
        drop = bad.neurons[(bad.neurons["cell_type"] == "DNa02") & (bad.neurons["side"] == "left")]["id"]
        n = bad.neurons[~bad.neurons["id"].isin(drop)].reset_index(drop=True)
        Readouts(Connectome("b", n, bad.edges[~bad.edges["post"].isin(drop)]), "flywire783")


def test_symmetry_gate_rejects_dnp15_and_dnp10_and_passes_dna02(ro):
    assert ro.gate["DNa02"].passed and not ro.gate["DNp15"].passed and not ro.gate["DNp10"].passed
    assert ro.gate["DNp15"].log2_total == pytest.approx(np.log2(1.66), abs=0.01)
    assert ro.usable_types() == ("DNa02",)
    with pytest.raises(GateFailure, match="DNp15"):
        ro.require("DNp15")
    rec = ro.record()
    assert rec["gate"]["DNp15"]["passed"] is False and rec["waived"] == []


def test_waiver_is_recorded():
    r = Readouts(toy_readouts(), "flywire783", waive=("DNp15",))
    r.require("DNp15")
    assert r.usable_types() == ("DNa02", "DNp15") and r.record()["waived"] == ["DNp15"]


def test_counts_rates_and_groups(ro):
    per = np.zeros(ro.cx.n_neurons, int)
    per[ro.sets["DNa02"]["L"]] = 3
    per[ro.sets["MN9"]["R"]] = 2
    c = ro.counts(per)
    assert c[ro.row("DNa02")].tolist() == [3, 0] and c[ro.row("MN9")].tolist() == [0, 2]
    np.testing.assert_allclose(ro.rates(c, DT)[ro.row("DNa02"), 0], 120.0)
    g = ro.group_names()
    assert len(g) == 2 * NT and g["ro_DNa02_L"].tolist() == ro.sets["DNa02"]["L"].tolist()


class _Engine:
    """Stub engine: rates() returns constant per-neuron rates by group."""

    def __init__(self, table):
        self.table = table

    def rates(self, names, window_ms):
        return {n: np.full(1, self.table.get(n, 0.0)) for n in names}


def test_hz_from_engine_groups(ro):
    eng = _Engine({"ro_DNa02_L": 40.0, "ro_MN9_R": 7.0})
    hz = ro.hz(eng, 25)
    assert hz[ro.row("DNa02")].tolist() == [40.0, 0.0] and hz[ro.row("MN9")].tolist() == [0.0, 7.0]


# ── bridge ───────────────────────────────────────────────────────────────────
def _counts(ro, mn9_hz):
    c = np.zeros((NT, 2))
    c[ro.row("MN9")] = mn9_hz * DT
    return c


def test_constants_fixed():
    assert MN9_THRESHOLD_HZ == 10.0 and READOUT_TAU == 0.05 and K_STEER == 1.0


def test_filter_is_first_order(ro):
    br = Bridge(ro, DT)
    br.set_baseline(np.zeros((NT, 2)))
    # rates() divides by the neuron count (1 per side) and dt: counts of 40 Hz * dt -> 40 Hz
    _, used = br.update(_counts(ro, 40.0))
    np.testing.assert_allclose(br.mn9_rate(used), 40.0 * (1 - np.exp(-DT / READOUT_TAU)))


def test_proboscis_threshold_and_ablation(ro):
    br = Bridge(ro, DT)
    br.set_baseline(np.zeros((NT, 2)))
    for _ in range(10):
        _, used = br.update(_counts(ro, 60.0))
    assert br.proboscis(used)["proboscis_extended"] == 1
    for _ in range(20):
        _, used = br.update(_counts(ro, 0.0))
    assert br.proboscis(used)["proboscis_extended"] == 0
    abl = Bridge(ro, DT, ablate=("mn9",))
    abl.set_baseline(np.zeros((NT, 2)))
    for _ in range(10):
        raw, used = abl.update(_counts(ro, 60.0))
    assert abl.proboscis(used)["proboscis_extended"] == 0 and raw[ro.row("MN9")].mean() == 60.0


def test_l2_ablation_without_baseline_is_an_explicit_error(ro):
    br = Bridge(ro, DT, ablate=("steer",))
    with pytest.raises(BaselineMissing, match="set_baseline"):
        br.update(_counts(ro, 10.0))
    Bridge(ro, DT).update_hz(np.zeros((NT, 2)))          # no ablation: no baseline needed


def _ref(ro):
    ref = np.ones((NT, 2))
    ref[ro.row("DNa02")] = (30.0, 10.0)
    ref[ro.row("DNp15")] = (15.0, 60.0)
    return ref


def _steer_counts(ro, a02, p15):
    c = np.zeros((NT, 2))
    c[ro.row("DNa02")] = np.asarray(a02) * DT
    c[ro.row("DNp15")] = np.asarray(p15) * DT
    return c


def _settle(br, c, n=40):
    br.set_baseline(np.zeros((NT, 2)))
    for _ in range(n):
        _, used = br.update(c)
    return br.steer(used)


def test_steer_drive_is_the_gated_type_and_dnp15_is_rejected(ro):
    """DNa02 passes the gate and drives; DNp15 (R/L 1.66x) is not a drive type unless waived."""
    br = Bridge(ro, DT, reference=_ref(ro))
    assert br.drive == ("DNa02",)
    s = _settle(br, _steer_counts(ro, (60, 10), (45, 60)))
    np.testing.assert_allclose(s["steer_norm_t"], [(60 - 30) / 20.0 - 0, (45 - 15) / 37.5], rtol=1e-6)
    np.testing.assert_allclose(s["steer_norm"], 1.5, rtol=1e-6)                 # DNa02 only
    np.testing.assert_allclose(s["turn_dn"], -K_STEER * 1.5, rtol=1e-6)
    np.testing.assert_allclose(s["steer_raw"], [50.0, -15.0], rtol=1e-6)
    with pytest.raises(GateFailure):
        Bridge(ro, DT, reference=_ref(ro), drive=("DNp15",))
    s0 = _settle(Bridge(ro, DT, reference=_ref(ro)), _steer_counts(ro, (30, 10), (15, 60)))
    assert abs(s0["steer_norm"]) < 1e-6                                          # the reference gives 0


def test_waived_drive_uses_dnp15_as_in_v1():
    r = Readouts(toy_readouts(), "flywire783", waive=("DNp15",))
    br = Bridge(r, DT, reference=_ref(r), drive=("DNp15",))
    s = _settle(br, _steer_counts(r, (0, 90), (45, 60)))
    np.testing.assert_allclose(s["steer_norm"], 0.8, rtol=1e-6)                  # v1 test value
    np.testing.assert_allclose(s["turn_dn"], -0.8, rtol=1e-6)


def test_swap_flips_ablation_holds_baseline_and_turn_is_clipped(ro):
    c = _steer_counts(ro, (60, 10), (45, 60))
    s = _settle(Bridge(ro, DT, reference=_ref(ro)), c)
    sw = _settle(Bridge(ro, DT, reference=_ref(ro), swap=("steer",)), c)
    assert sw["turn_dn"] == pytest.approx(-s["turn_dn"])
    ab = Bridge(ro, DT, reference=_ref(ro), ablate=("steer",))
    base = np.zeros((NT, 2))
    base[ro.row("DNa02")] = (30.0, 10.0)
    base[ro.row("DNp15")] = (15.0, 60.0)
    ab.set_baseline(base)
    for _ in range(20):
        _, used = ab.update(c)
    assert ab.steer(used)["turn_dn"] == pytest.approx(0.0)
    big = _settle(Bridge(ro, DT, reference=_ref(ro)), _steer_counts(ro, (3000, 0), (0, 0)))
    assert big["turn_dn"] == -LB.TURN_BIAS_MAX


def test_zero_reference_rejected_and_missing_reference_is_explicit(tmp_path, ro):
    ref = {t: dict(L=1.0, R=1.0) for t in READOUT_TYPES}
    ref["DNp15"] = dict(L=0.0, R=0.0)
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(dict(rates_hz=ref)))
    with pytest.raises(ValueError):
        LB.load_reference(p)
    with pytest.raises(RuntimeError, match="reference"):
        Bridge(ro, DT).steer(np.zeros((NT, 2)))


# ── real FlyWire v783 ────────────────────────────────────────────────────────
@pytest.mark.skipif(not (ROOT / "brain_model" / "Connectivity_783.parquet").exists(), reason="needs brain_model/")
def test_flywire_gate_matches_review_h2():
    """H2 of the review: DNp15 R/L = 1.66x and DNp10 = 0.24x fail; the pre-registered primary DNa02 (1.07x)
    passes; MN9 by its CB0701 alias is found (2 neurons)."""
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    r = Readouts(cx, "flywire783")
    assert r.gate["DNa02"].passed and not r.gate["DNp15"].passed and not r.gate["DNp10"].passed
    assert r.gate["DNp15"].in_right / r.gate["DNp15"].in_left == pytest.approx(1.66, abs=0.02)
    assert r.gate["DNp10"].in_right / r.gate["DNp10"].in_left == pytest.approx(0.24, abs=0.02)
    mn9 = cx.index.ids_of(np.concatenate([r.sets["MN9"]["L"], r.sets["MN9"]["R"]]))
    assert set(mn9) == {720575940618238523, 720575940660219265}
    assert r.n.tolist() == [[1, 1]] * NT
    ref = LB.load_reference()
    assert np.all(ref[[r.row(t) for t in ("DNa02", "DNp15")]] > 0)


def test_zero_turn_control_arm(ro):
    """The named control arm of new experiments: yaw command exactly 0, s_hat and the readouts still recorded."""
    c = _steer_counts(ro, (60, 10), (45, 60))
    s = _settle(Bridge(ro, DT, reference=_ref(ro)), c)
    z = _settle(Bridge(ro, DT, reference=_ref(ro), zero_turn=True), c)
    assert s["turn_dn"] != 0.0 and z["turn_dn"] == 0.0
    assert z["steer_norm"] == pytest.approx(s["steer_norm"]) and np.allclose(z["steer_raw"], s["steer_raw"])


def test_hover_yaw_cli_defaults_and_control_exclusion():
    """--waive defaults to no waiver (rule 5); the two control arms cannot be combined."""
    import pytest
    from tasks import hover_yaw
    base = ["--name", "x", "--v1", "v30"]
    assert hover_yaw.parse_args(base).waive == []
    assert hover_yaw.parse_args(base + ["--waive", "DNp15"]).waive == ["DNp15"]
    with pytest.raises(SystemExit):
        hover_yaw.parse_args(base + ["--zero-turn", "--ablate-steer"])


def test_to_intervals_does_not_mutate_args():
    import argparse
    from tasks.f0_flight import to_intervals
    a = argparse.Namespace(n_steps=60, calib_steps=10, persist_steps=8)
    b = to_intervals(a, 5)
    assert (a.n_steps, a.calib_steps, a.persist_steps) == (60, 10, 8)
    assert (b.n_steps, b.calib_steps, b.persist_steps) == (300, 50, 40)
    assert b.v1_steps == dict(n_steps=60, calib_steps=10, persist_steps=8) and b.stride == 5
