"""Stage 3 body physics (FlyGym + stroke-averaged xfrc), no brain input. ~4 ms per 0.1 ms physics step."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from body import quasi_steady as qs
from body.arena import load_arena
from body.flygym_body import FlightBody
from body.noise import PhysicalNoise
from configs import load
from motor import legacy_reflexes as RF

W = load("body_flight")
ARENA = load_arena()
OPEN_AIR = (-40.0, -140.0, 100.0)   # away from towers, pedestal and platform
DT = 0.025                          # one 25 ms interval = 250 physics steps (the v1 decision step)


def intervals(seconds, dt=DT):
    return int(round(seconds / dt))


@pytest.fixture(scope="module")
def air_body():
    return FlightBody(spawn_pos=OPEN_AIR, legs="tuck")


def test_hover_without_brain():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    assert p.weight == pytest.approx(b.mass * 9810.0)
    a = qs.hover_amplitude(p)
    cmd = qs.WingCommand(a, a, p.f0)
    halt = RF.HaltereReflex(p)
    z0 = b.state()["pos"][2]
    zmax = 0.0
    for _ in range(intervals(0.5)):
        r = b.step(cmd, DT, inner=[halt])
        zmax = max(zmax, abs(b.state()["pos"][2] - z0))
    assert zmax < 0.5
    assert r["badqacc"] == 0 and r["clipped_substeps"] == 0 and r["n_substeps"] == 250


def test_interval_is_a_duration_not_a_constant():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    a = qs.hover_amplitude(b.params)
    cmd = qs.WingCommand(a, a, b.params.f0)
    t0 = b.state()["time"]
    b.step(cmd, 0.005)
    r = b.step(cmd, 0.0025)
    assert r["n_substeps"] == 25
    assert b.state()["time"] - t0 == pytest.approx(0.0075)
    with pytest.raises(ValueError):
        b.step(cmd, 0.00015)


def test_tower_crash_400mm_s():
    b = FlightBody(spawn_pos=(148.0, 30.0, 100.0), legs="tuck")   # 12 mm in front of tower 1's face (x = 160)
    p = b.params
    a = qs.hover_amplitude(p)
    b.set_velocity([400.0, 0.0, 0.0])
    cmd = qs.WingCommand(a, a, p.f0)
    xmax = -np.inf
    for _ in range(4):
        r = b.step(cmd, DT)
        xmax = max(xmax, b.state()["pos"][0])
    assert b.state()["vel"][0] < 0          # it hit the wall and bounced
    assert 0 < b.max_tower_penetration < 0.1
    assert r["badqacc"] == 0
    assert xmax < ARENA.tower1[0][0]        # COM never inside the tower


def test_platform_touchdown_detected():
    cx, cy, _, zt = ARENA.food_platform
    b = FlightBody(spawn_pos=(cx, cy, zt + 1.5), legs="stand")
    b.step(qs.WingCommand(0, 0, 0, on=False), 0.05)
    c = b.contacts()
    assert c["platform_legs"].sum() >= 2 and not c["tower_contact"]
    assert b.state()["pos"][2] > zt       # resting on top, not fallen through


def test_terminal_speed_at_cruise_pitch():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    th = np.radians(W["cruise_pitch_deg"])
    cmd = qs.WingCommand(qs.hover_amplitude(p, th), qs.hover_amplitude(p, th), p.f0)
    halt = RF.HaltereReflex(p)
    halt.set_target(pitch_down=th)
    for _ in range(intervals(1.0)):
        b.step(cmd, DT, inner=[halt])
    v = b.state()["vel"]
    speed = float(np.hypot(v[0], v[1]))
    print(f"\n[terminal speed] tilt {W['cruise_pitch_deg']} deg: {speed:.1f} mm/s, vz {v[2]:+.2f}")
    assert 200.0 <= speed <= 400.0 and v[0] > 0 and abs(v[2]) < 5.0


def test_positive_turn_bias_yaws_right():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    aL, aR = qs.turn_to_amplitudes(0.5, qs.hover_amplitude(p), W["turn_dphi_deg"])
    cmd = qs.WingCommand(aL, aR, p.f0)
    halt = RF.HaltereReflex(p)
    h0 = b.state()["heading"]
    for _ in range(intervals(0.25)):
        b.step(cmd, DT, inner=[halt])
    s = b.state()
    dh = np.angle(np.exp(1j * (s["heading"] - h0)))
    assert dh < 0 and s["omega"][2] < 0


def _tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _gust_response(axis, with_haltere=True, pulse=0.05, after=0.2, chunk=0.005):
    """Hover, then a 50 ms body-axis torque pulse (roll: axis 0, pitch: axis 1) of the reflex's static
    stiffness x 0.35 rad (~20 deg if held)."""
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    a = qs.hover_amplitude(p)
    cmd = qs.WingCommand(a, a, p.f0)
    halt = RF.HaltereReflex(p)
    inner = [halt] if with_haltere else []
    b.step(cmd, 0.1, inner=inner)                        # settle
    tau_b = np.zeros(3)
    tau_b[axis] = halt.kp[axis] * 0.35
    peak = 0.0
    for _ in range(int(round(pulse / chunk))):
        R = b.state()["R"]
        b.step(cmd, chunk, inner=inner, ext_torque=R @ tau_b)
        peak = max(peak, _tilt_deg(b.state()["R"]))
    for _ in range(int(round(after / chunk))):
        b.step(cmd, chunk, inner=inner)
        peak = max(peak, _tilt_deg(b.state()["R"]))
    s = b.state()
    return peak, _tilt_deg(s["R"]), s["R"].T @ s["omega"], b.badqacc_count() - b.badqacc0


@pytest.mark.parametrize("axis,name", [(0, "roll"), (1, "pitch")])
def test_haltere_reflex_recovers_gust_within_200ms(axis, name):
    peak, tilt, w_b, badq = _gust_response(axis)
    assert peak > 5.0 and tilt < 1.0 and tilt < 0.1 * peak
    assert np.all(np.abs(w_b[:2]) < 0.5) and badq == 0


def test_without_haltere_gust_is_not_recovered():
    """Control: the recovery comes from the (HAND-MADE) REFLEX arm, not the aero model."""
    peak, tilt, _, _ = _gust_response(0, with_haltere=False)
    assert tilt > 0.5 * peak


@pytest.mark.parametrize("deg", [30.0, -30.0])
def test_yaw_pulse_passive_rotation(deg):
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    a = qs.hover_amplitude(p)
    cmd = qs.WingCommand(a, a, p.f0)
    halt = RF.HaltereReflex(p)
    b.step(cmd, 0.1, inner=[halt])
    T = b.yaw_pulse_torque(deg, 0.1)
    h0 = b.state()["heading"]
    for _ in range(intervals(0.1)):
        b.step(cmd, DT, inner=[halt], ext_torque=b.state()["R"][:, 2] * T)
    for _ in range(intervals(0.15)):
        b.step(cmd, DT, inner=[halt])
    s = b.state()
    dh = float(np.degrees(np.angle(np.exp(1j * (s["heading"] - h0)))))
    assert abs(dh - deg) < 3.0 and abs(s["omega"][2]) < 0.2


def test_amplitude_clip_is_recorded_in_the_body():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    p = b.params
    r = b.step(qs.WingCommand(p.phi_max + 20, p.phi_max, p.f0), 0.005)
    assert r["clipped_substeps"] == r["n_substeps"] == 50 and b.clip_events == 50


def test_physical_noise_changes_the_trajectory_with_the_seed():
    def run(seed):
        b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
        a = qs.hover_amplitude(b.params)
        cmd = qs.WingCommand(a, a, b.params.f0)
        n = PhysicalNoise(seed, force_sigma=3.0, torque_sigma=0.0, tau=0.05)
        for _ in range(8):
            b.step(cmd, 0.005, noise=n)
        return b.state()["pos"]
    p1, p1b, p2 = run(1), run(1), run(2)
    np.testing.assert_allclose(p1, p1b)
    assert np.abs(p1 - p2).max() > 1e-6


def test_textured_arena_and_subframe_render():
    """Obstacles drawn by visual-only checker meshes, collision primitives invisible but unchanged;
    body.step(render_at=...) returns the eye frames."""
    b = FlightBody(spawn_pos=(120.0, 20.0, 100.0), legs="tuck", enable_vision=True)
    m = b.m
    for name in ("tower1", "tower2", "pedestal", "food_platform"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        v = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"viz_tex_{name}")
        assert g >= 0 and v >= 0
        assert m.geom_rgba[g][3] == 0 and m.geom_rgba[v][3] == 1
        assert m.geom_contype[v] == 0 and m.geom_conaffinity[v] == 0
        np.testing.assert_allclose(m.geom_solref[g], [0.02, 1])
    viz = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"viz_tex_{n}") for n in ("tower1", "food_platform")}
    assert not viz & set(m.pair_geom1) and not viz & set(m.pair_geom2)
    v_tex = b.update_vision().max(axis=2)
    v_flat = FlightBody(spawn_pos=(120.0, 20.0, 100.0), legs="tuck", enable_vision=True,
                        textured=False).update_vision().max(axis=2)
    vals, cnt = np.unique(np.round(v_flat, 3), return_counts=True)
    tower = np.abs(v_flat - vals[cnt.argmax()]) < 2e-3
    assert tower.sum() > 200 and v_flat[tower].std() < 0.01 and v_tex[tower].std() > 0.1
    r = b.step(qs.WingCommand(0, 0, 0, on=False), 0.002, render_at=(10, 20))
    assert len(r["frames"]) == 2 and r["frames"][0].shape == (2, 721, 2)


def test_dark_platform_is_visual_only():
    b = FlightBody(spawn_pos=(440.0, -70.0, 150.0), spawn_yaw=np.pi / 2, legs="tuck", enable_vision=True,
                   platform="dark")
    m = b.m
    v = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "viz_tex_food_platform")
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "food_platform")
    np.testing.assert_allclose(m.geom_rgba[v], ARENA.platform_dark_rgba)
    assert m.geom_matid[v] == -1 and m.geom_contype[v] == 0 and m.geom_rgba[g][3] == 0
    np.testing.assert_allclose(m.geom_solref[g], [0.02, 1])
    assert m.geom_matid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "viz_tex_tower1")] >= 0
    dark = b.update_vision().max(axis=2)
    neutral = FlightBody(spawn_pos=(440.0, -70.0, 150.0), spawn_yaw=np.pi / 2, legs="tuck", enable_vision=True,
                         platform="neutral").update_vision().max(axis=2)
    diff = neutral - dark
    assert (diff > 0.05).sum() > 5 and (diff < -0.05).sum() == 0


def test_leg_transition_ramps_without_single_step_jump():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="stand")
    p = b.params
    a = qs.hover_amplitude(p)
    cmd = qs.WingCommand(a, a, p.f0)
    pdt = ARENA.physics_dt
    for legs, ramp_s, start, end in (("tuck", W["leg_tuck_ramp_s"], b.stand_pose, b.tuck_pose),
                                     ("stand", W["leg_extend_ramp_s"], b.tuck_pose, b.stand_pose)):
        full = np.abs(end - start).max()
        n = int(round(ramp_s / pdt))
        b.set_legs(legs)
        targets = [b.leg_target.copy()]
        for _ in range(n):
            b.step(cmd, pdt)
            targets.append(b.leg_target.copy())
            assert np.allclose(b.d.ctrl[b._act_ids], b.leg_target)
        steps = np.abs(np.diff(targets, axis=0)).max(axis=1)
        assert full > np.radians(30) and steps.max() <= np.pi / 2 * full / n * 1.01
        frac = np.abs(targets[250] - start).max() / full       # after one 25 ms interval
        assert 0.0 < frac < 0.8
        assert np.allclose(targets[-1], end) and not b.leg_ramp_active
        b.set_legs(legs)                                       # same pose again: no new ramp
        assert not b.leg_ramp_active
    assert b.badqacc_count() - b.badqacc0 == 0


# ── head reflex (REFLEX arm) ─────────────────────────────────────────────────
def test_head_joint_axes_and_no_actuators_without_flag():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck")
    m, d = b.m, b.d
    R = d.xmat[b.thorax].reshape(3, 3)
    for ax in RF.AXES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{b.fly.name}/{RF.JOINTS[ax]}")
        v = R.T @ d.xaxis[jid]
        assert np.argmax(np.abs(v)) == RF.THORAX_AXIS[ax] and v[RF.THORAX_AXIS[ax]] > 0.99
    for i in range(m.ncam):
        bid = m.cam_bodyid[i]
        while bid and bid != b.head_body:
            bid = m.body_parentid[bid]
        assert bid == b.head_body
    hb = FlightBody(spawn_pos=OPEN_AIR, legs="tuck", neck_actuators=True)
    assert hb.m.nu == m.nu + 3
    with pytest.raises(RuntimeError, match="neck"):
        b.step(qs.WingCommand(0, 0, 0, on=False), 0.001, inner=[RF.HeadReflex()])


def test_head_reflex_turn_stabilises_gaze_within_limits():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="tuck", neck_actuators=True)
    p = b.params
    a = qs.hover_amplitude(p)
    halt, head = RF.HaltereReflex(p), RF.HeadReflex()
    n_turn = n_lt = 0
    qmax = np.zeros(3)
    for turn in (0.5, -0.5):
        aL, aR = qs.turn_to_amplitudes(turn, a, W["turn_dphi_deg"])
        for k in range(intervals(0.25)):
            r = b.step(qs.WingCommand(aL, aR, p.f0), DT, inner=[halt, head])
            h = r["head"]
            n_turn += h["turn_sub"]
            n_lt += h["turn_sub_lt"]
            qmax = np.maximum(qmax, h["q_absmax"])
            if k == 4:
                assert np.sign(h["target"][0]) == -np.sign(h["body_yaw_rate"]) != 0
                assert abs(h["gaze_yaw_rate"]) < abs(h["body_yaw_rate"])
    assert b.badqacc_count() - b.badqacc0 == 0
    assert np.all(qmax <= head.theta_max + 1e-9)
    assert n_turn > 1000 and n_lt / n_turn > 0.5


# ── postures ─────────────────────────────────────────────────────────────────
def test_flight_pose_and_feed_pose():
    b = FlightBody(spawn_pos=OPEN_AIR, legs="stand", postures=True)
    p = b.params
    a = qs.hover_amplitude(p)
    cmd = qs.WingCommand(a, a, p.f0)
    b.set_legs("tuck")
    for _ in range(intervals(0.25)):
        b.step(cmd, DT)
    m, d = b.m, b.d
    R = d.xmat[b.thorax].reshape(3, 3)

    def x_body(name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{b.fly.name}/{name}")
        return float(R[:, 0] @ (d.xpos[bid] - d.xpos[b.thorax]))
    for s in "LR":
        assert x_body(f"{s}FTarsus5") > x_body(f"{s}MCoxa")
        assert x_body(f"{s}HTarsus5") < x_body(f"{s}HCoxa")
    names = list(b.fly.actuated_joints)
    q = np.array([d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{b.fly.name}/{n}")]]
                  for n in names])
    assert np.degrees(np.abs(q - b.tuck_pose)).max() < 3.0 and b.badqacc_count() - b.badqacc0 == 0
    with pytest.raises(ValueError):
        FlightBody(spawn_pos=OPEN_AIR, legs="tuck").set_legs("feed")


def test_feed_pose_leans_forward_and_back_smoothly():
    b = FlightBody(legs="stand", postures=True)
    b.set_legs("stand", adhesion=1.0)
    off = qs.WingCommand(0, 0, 0, on=False)

    def pitch():
        return float(np.degrees(np.arcsin(-b.state()["R"][2, 0])))     # + = nose down
    for _ in range(8):
        b.step(off, DT)
    p0 = pitch()
    b.set_legs("feed")
    assert b._ramp[3] == int(round(W["leg_feed_ramp_s"] / ARENA.physics_dt))
    for _ in range(intervals(0.5)):
        b.step(off, DT)
    p1 = pitch()
    b.set_legs("stand")
    for _ in range(intervals(0.5)):
        b.step(off, DT)
    p2 = pitch()
    assert 3.0 <= p1 - p0 <= 10.0 and abs(p2 - p0) < 1.0 and b.badqacc_count() - b.badqacc0 == 0


# ── start offset / yaw (arena spec) ──────────────────────────────────────────
def test_body_spawn_follows_the_arena_spec():
    arena = ARENA.with_start_offset(0, -40)
    body = FlightBody(arena=arena, spawn_yaw=np.radians(60))
    s = body.state()
    assert s["pos"][0] == pytest.approx(0.0, abs=0.5) and s["pos"][1] == pytest.approx(-40.0, abs=0.5)
    assert np.degrees(s["heading"]) == pytest.approx(60.0, abs=1.0)
    g = next(i for i in range(body.m.ngeom) if (body.m.geom(i).name or "").endswith("pedestal")
             and not (body.m.geom(i).name or "").startswith("viz"))
    np.testing.assert_allclose(body.d.geom_xpos[g][:2], [0.0, -40.0], atol=1e-9)
