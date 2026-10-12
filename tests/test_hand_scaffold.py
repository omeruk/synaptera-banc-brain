"""HAND scaffold (take-off, cruise, approach, descend, touchdown): terms, phases, term bookkeeping.

Adapted from v1 tests/flight/test_hybrid.py; the turn_hand tests are removed (NeuroFly adaptation, not carried)."""
import numpy as np
import pytest

from body import quasi_steady as qs
from body.arena import load_arena
from configs import load
from motor import hand_scaffold as HS

A = load_arena()
S = HS.HandScaffold(A)
TOP = A.food_platform[3]
CX, CY = A.food_platform[:2]
H = load("hand_scaffold")
W = load("body_flight")
DT = 0.025


def test_no_odour_turn_map_is_carried():
    assert not hasattr(HS, "turn_hand") and not hasattr(S, "turn_hand")
    assert "odor_turn_k" not in H and "flyvis_t5_gain" not in H


def test_altitude_hand_terms():
    th = S.thrust_hand("cruise", 50.0, 0.0, I_grad=0.0)
    assert th["alt"] > 0 and th["odor"] == 0                       # climb assist, never a brake
    assert S.thrust_hand("cruise", 190.0, 100.0)["alt"] == 0.0
    assert S.thrust_hand("approach", TOP + H["approach_clear"] + 10, 0.0)["alt"] < 0
    assert S.thrust_hand("approach", TOP + H["approach_clear"] - 10, 0.0)["alt"] > 0
    assert S.thrust_hand("approach", TOP + H["approach_clear"], 0.0, I_grad=0.5)["total"] == pytest.approx(0.0)
    assert S.thrust_hand("descend", 180.0, -H["land_vz"])["total"] == pytest.approx(0.0)
    assert S.thrust_hand("takeoff", 20.0, 0.0)["boost"] == H["takeoff_lift"]
    # the odour vertical-gradient term exists only with a gradient and is switchable
    assert S.thrust_hand("cruise", 190.0, 100.0, I_grad=0.3)["odor"] == pytest.approx(0.8 * np.tanh(3.0))
    assert S.thrust_hand("cruise", 190.0, 100.0, I_grad=0.3, ablate_odor=True)["odor"] == 0.0


def test_approach_tilt_points_at_the_platform_in_the_world_frame():
    for heading in (0.0, 1.0, -2.5):
        pos = np.array([CX - 50.0, CY, 177.0])
        pd, rr, v = S.tilt_hand("approach", pos, np.zeros(3), heading)
        h = np.array([np.cos(heading), np.sin(heading)])
        r = np.array([np.sin(heading), -np.cos(heading)])
        tilt = pd * h + rr * r
        assert tilt[0] > 0 and abs(tilt[1]) < 1e-9
        assert np.hypot(pd, rr) <= np.radians(H["tilt_max_deg"]) + 1e-12
        assert v == pytest.approx(min(W["terminal_speed_ref"], H["v_gain"] * 50.0))
    pd, _, _ = S.tilt_hand("approach", np.array([CX - 5.0, CY, 177.0]), np.array([200.0, 0, 0]), 0.0)
    assert pd < 0                                                   # fast and close: brake
    assert S.tilt_hand("cruise", np.zeros(3), np.zeros(3), 0.0)[0] == pytest.approx(np.radians(W["cruise_pitch_deg"]))
    assert S.tilt_hand("landed", np.zeros(3), np.zeros(3), 0.0) == (0.0, 0.0, 0.0)


def test_phase_sequence_and_interval_independence():
    for dt in (0.025, 0.005):                                       # step counts follow from seconds
        pm = HS.HandPhases(S, dt)
        far = np.array([0.0, 0.0, 50.0])
        n_to = int(round(0.25 / dt))
        for _ in range(n_to - 1):
            pm.update(far, np.zeros(3), 0)
        assert pm.phase == "takeoff"
        pm.update(far, np.zeros(3), 0)
        assert pm.phase == "cruise" and pm.wings_on and not pm.legs_extended
        pm.update(np.array([CX - 50, CY, 177.0]), np.zeros(3), 0)
        assert pm.phase == "approach" and pm.legs_extended
        pm.update(np.array([CX - 2, CY, 177.0]), np.array([100.0, 0, 0]), 0)
        assert pm.phase == "approach"                               # too fast to descend
        pm.update(np.array([CX - 2, CY, 177.0]), np.array([10.0, 0, 0]), 0)
        assert pm.phase == "descend" and pm.wings_on
        for _ in range(int(round(0.05 / dt))):
            pm.update(np.array([CX, CY, 170.5]), np.zeros(3), 2)
        assert pm.phase == "touchdown" and pm.landed and not pm.wings_on
        pm.update(np.array([CX, CY, 170.5]), np.zeros(3), 6)
        assert pm.phase == "landed"
    assert HS.PHASE_CODE["descend"] == 9 and HS.PHASE_CODE["landed"] == 8      # v1 codes kept


def test_combine_turn_and_reflex_inverse():
    assert HS.combine_turn(0.3, -0.5, 0.1) == pytest.approx(-0.1)
    assert HS.combine_turn(2.0, 2.0, 0.0) == 2.5
    assert HS.combine_turn(0.0) == 0.0
    p = qs.make_params(1e-3, 9810.0, np.diag([1e-4, 1e-4, 1e-4]), W)
    turn = 0.7
    aL, aR = qs.turn_to_amplitudes(turn, 140.0, W["turn_dphi_deg"])
    _, tq, _ = qs.aero_wrench(np.eye(3), np.zeros(3), np.zeros(3), qs.WingCommand(aL, aR, p.f0), p)
    assert qs.yaw_torque_to_turn(tq[2], p, W["turn_dphi_deg"]) == pytest.approx(turn)


def test_wing_command_lift_and_limits():
    p = qs.make_params(1e-3, 9810.0, np.diag([1e-4, 1e-4, 1e-4]), W)
    off, lift = S.wing_command(0.0, 0.5, 0.0, 0.0, p, wings_on=False)
    assert not off.on and lift == 1.0
    cmd, lift = S.wing_command(0.0, 0.5, 0.0, 0.0, p)
    assert lift == pytest.approx(1.5) and cmd.amp_L == pytest.approx(cmd.amp_R)
    assert cmd.amp_L == pytest.approx(qs.hover_amplitude(p) * np.sqrt(1.5))
    _, lift = S.wing_command(0.0, 9.0, 0.0, 0.0, p)
    assert lift == 1.8                                               # lift_frac_range upper bound
    cmd, _ = S.wing_command(1.0, 0.0, 0.0, 0.0, p)
    assert cmd.amp_L > cmd.amp_R                                     # turn > 0 -> larger left amplitude
