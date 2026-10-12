"""ArenaSpec, wing model (aero, clip, options), physical noise, FlyGym private-API adapter: no MuJoCo stepping."""
import dataclasses

import numpy as np
import pytest

from body import quasi_steady as qs
from body.arena import ArenaSpec, load_arena
from body.noise import PhysicalNoise
from configs import load

W = load("body_flight")


@pytest.fixture(scope="module")
def params():
    I = np.diag([1.2e-4, 5.1e-4, 5.1e-4])   # measured whole-fly inertia (tuck), g mm^2
    return qs.make_params(1.027e-3, 9810.0, I, W)


# ── arena ────────────────────────────────────────────────────────────────────
def test_arena_values_and_frozen():
    a = load_arena()
    assert a.scale == 20.0 and a.tower1 == ((160.0, 200.0), (-100.0, 160.0), (0.0, 200.0))
    assert a.takeoff_pedestal == (0.0, 0.0, 5.0, 19.5) and a.food_platform == (440.0, 80.0, 10.0, 169.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.scale = 1.0
    assert a.physics_steps(0.025) == 250 and a.physics_steps(0.005) == 50
    with pytest.raises(ValueError):
        a.physics_steps(0.00015)


def test_start_offset_returns_new_spec_and_checks_geometry():
    a = load_arena()
    b = a.with_start_offset(-40, 40)
    assert b.takeoff_pedestal == (-40.0, 40.0, 5.0, 19.5)
    np.testing.assert_allclose(b.start_pos, [-40, 40, 20.0])
    assert a.takeoff_pedestal == (0.0, 0.0, 5.0, 19.5)         # the original is untouched
    with pytest.raises(ValueError, match="tower"):
        a.with_start_offset(180, 0)
    with pytest.raises(ValueError, match="odor grid"):
        a.with_start_offset(-1000, 0)


def test_surface_below_and_clearance():
    a = load_arena()
    assert a.surface_below(180.0, 0.0) == 200.0            # tower 1 top
    assert a.surface_below(0.0, 0.0) == 19.5               # pedestal
    assert a.surface_below(440.0, 80.0) == 169.5           # food platform
    assert a.surface_below(-50.0, -50.0) == 0.0
    assert a.tower_clearance([180.0, 0.0, 100.0]) == 0.0
    assert a.tower_clearance([150.0, 0.0, 100.0]) == pytest.approx(10.0)
    assert set(a.geometry_record()) == {"tower1", "tower2", "pedestal", "food_platform", "food"}


# ── wing model ───────────────────────────────────────────────────────────────
def test_hover_amplitude_gives_weight(params):
    a = qs.hover_amplitude(params)
    assert a == pytest.approx(W["phi0_deg"])
    F, T, info = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), qs.WingCommand(a, a, W["f0_hz"]), params)
    np.testing.assert_allclose(F, [0, 0, params.weight], rtol=1e-12)
    np.testing.assert_allclose(T, 0, atol=1e-12)
    assert not info["clipped_L"] and not info["clipped_R"]
    th = np.radians(12)
    assert 2 * qs.wing_lift(qs.hover_amplitude(params, th), W["f0_hz"], params) * np.cos(th) == pytest.approx(params.weight)


def test_turn_sign_in_wrench(params):
    aL, aR = qs.turn_to_amplitudes(1.0, W["phi0_deg"], W["turn_dphi_deg"])
    assert aL > aR
    _, T, _ = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), qs.WingCommand(aL, aR, W["f0_hz"]), params)
    assert T[2] < 0      # clockwise from above = right turn
    assert T[0] > 0      # left wing lifts more -> bank right


def test_counter_force_opposes_velocity(params):
    a = W["phi0_deg"]
    v = np.array([100.0, -50.0, 20.0])
    F, _, _ = qs.aero_wrench(np.eye(3), v, np.zeros(3), qs.WingCommand(a, a, W["f0_hz"]), params)
    np.testing.assert_allclose(F - np.array([0, 0, params.weight]), -params.c_F * v)


def test_folded_wings_give_no_wrench(params):
    F, T, info = qs.aero_wrench(np.eye(3), np.ones(3), np.ones(3), qs.WingCommand(0, 0, 0, on=False), params)
    assert not F.any() and not T.any() and info["s"] == 0.0


def test_desired_up_and_yaw_turn_inverse(params):
    np.testing.assert_allclose(qs.desired_up(0.3, 0.0), [0, 0, 1])
    z = qs.desired_up(0.0, np.radians(12))
    assert z[0] > 0 and z[1] == pytest.approx(0)
    assert qs.desired_up(0.0, 0.0, np.radians(10))[1] < 0
    a = qs.hover_amplitude(params)
    turn = 0.7
    aL, aR = qs.turn_to_amplitudes(turn, a, W["turn_dphi_deg"])
    _, T, _ = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), qs.WingCommand(aL, aR, params.f0), params)
    assert qs.yaw_torque_to_turn(T[2], params, W["turn_dphi_deg"]) == pytest.approx(turn)


def test_amplitude_limit_and_clip_record(params):
    """Review H5: v1 reached 193.8 / 206.3 deg; now clipped at phi_max with a record."""
    f = params.f0
    big = params.phi_max + 26.3
    F, T, info = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), qs.WingCommand(big, params.phi_max, f), params)
    assert info["clipped_L"] and not info["clipped_R"]
    np.testing.assert_allclose(T[0], 0, atol=1e-9)         # both wings at the limit: no roll torque
    np.testing.assert_allclose(T[2], 0, atol=1e-9)
    assert F[2] == pytest.approx(2 * qs.wing_lift(params.phi_max, f, params))
    assert qs.clip_amplitude(-5.0, params) == (0.0, True)
    assert qs.clip_amplitude(100.0, params) == (100.0, False)


def test_m9_options_default_to_v1_and_change_the_force(params):
    a = qs.hover_amplitude(params)
    cmd = qs.WingCommand(a, a, params.f0)
    v = np.array([100.0, 0.0, 50.0])
    F0, _, _ = qs.aero_wrench(np.eye(3), v, np.zeros(3), cmd, params)
    tilt = dataclasses.replace(params, stroke_plane_rad=np.radians(10.0))
    Ft, _, _ = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), cmd, tilt)
    assert Ft[0] > 0 and Ft[2] < params.weight                  # lift tilted forward
    aniso = dataclasses.replace(params, damp_axial_ratio=3.0)
    Fa, _, _ = qs.aero_wrench(np.eye(3), v, np.zeros(3), cmd, aniso)
    assert Fa[0] == pytest.approx(F0[0]) and Fa[2] < F0[2]      # extra damping along body z only


# ── physical noise ───────────────────────────────────────────────────────────
def test_noise_off_by_default_and_seeded():
    n0 = PhysicalNoise(1)
    assert not n0.active and not any(map(np.any, n0.step(0.025)))
    a = PhysicalNoise(7, force_sigma=2.0, torque_sigma=3.0, tau=0.05)
    b = PhysicalNoise(7, force_sigma=2.0, torque_sigma=3.0, tau=0.05)
    c = PhysicalNoise(8, force_sigma=2.0, torque_sigma=3.0, tau=0.05)
    sa = np.array([np.r_[a.step(0.025)] for _ in range(4000)])
    sb = np.array([np.r_[b.step(0.025)] for _ in range(4000)])
    sc = np.array([np.r_[c.step(0.025)] for _ in range(4000)])
    np.testing.assert_array_equal(sa, sb)
    assert not np.array_equal(sa, sc)
    assert sa[:, :3].std() == pytest.approx(2.0, rel=0.1) and sa[:, 3:].std() == pytest.approx(3.0, rel=0.1)


def test_noise_statistics_do_not_depend_on_interval():
    fine = PhysicalNoise(3, force_sigma=1.0, torque_sigma=1.0, tau=0.05)
    coarse = PhysicalNoise(3, force_sigma=1.0, torque_sigma=1.0, tau=0.05)
    f = np.array([fine.step(0.005)[0] for _ in range(20000)])
    c = np.array([coarse.step(0.025)[0] for _ in range(20000)])
    assert f.std() == pytest.approx(c.std(), rel=0.1)


# ── FlyGym private API (M6) ──────────────────────────────────────────────────
def test_flygym_private_api_is_where_the_adapter_expects_it():
    flygym = pytest.importorskip("flygym")
    from body import flygym_private as P
    assert P.check_private_api(flygym.Fly) == []
    import importlib.metadata as md
    assert md.version("flygym") == P.REQUIRED_FLYGYM
