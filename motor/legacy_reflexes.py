"""REFLEX arm: hand-made inner loops that run every physics sub-step (label REFLEX, not BRAIN / VNC).

Ported from synaptera@c94796d:flight/quasi_steady.py (haltere_torque) and flight/head_reflex.py. They stand in
for the VNC haltere-wing reflex and the neck gaze reflex until those are handed to the connectome (RQ2).
Constants are in configs/reflex.yaml with origin classes.

Both implement the :class:`InnerLoop` protocol of body/flygym_body.py:
    update(R, v, omega, dt) -> InnerOutput(torque=..., neck_target=...)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from body import quasi_steady as qs
from configs import load

AXES = ("yaw", "pitch", "roll")
# FlyGym joint names do not match the physical axes (measured in v1): joint_Head_roll turns about thorax z
# (yaw), joint_Head about y (pitch, + = nose down), joint_Head_yaw about x (roll, + = right side down).
JOINTS = {"yaw": "joint_Head_roll", "pitch": "joint_Head", "roll": "joint_Head_yaw"}
THORAX_AXIS = {"yaw": 2, "pitch": 1, "roll": 0}          # component of omega_b (thorax x, y, z)


@dataclass
class InnerOutput:
    torque: np.ndarray | None = None        # world-frame torque (uN mm) about the COM
    force: np.ndarray | None = None
    neck_target: np.ndarray | None = None   # (yaw, pitch, roll) joint targets (rad)


class HaltereReflex:
    """Attitude PD on roll / pitch towards a desired body 'up' vector, rate damping on yaw (HAND-MADE).
    The attitude target (pitch_down, roll_right in rad) is set by the controller once per interval."""

    label = "REFLEX"

    def __init__(self, params: qs.FlightParams, reflex: dict | None = None):
        r = load("reflex") if reflex is None else reflex
        wn, z = r["haltere_omega_n"], r["haltere_zeta"]
        ixx, iyy, izz = params.inertia
        self.kp = np.array([ixx, iyy]) * wn ** 2
        self.kd = 2 * z * np.array([ixx, iyy]) * wn
        c_T = params.c_T                                  # = Izz / rot_damp_tau
        self.kd_yaw = max(params.d_yaw_total - c_T, 0.0)
        self.pitch_down = 0.0
        self.roll_right = 0.0
        self.yaw_torque_sum = 0.0                         # body-z haltere torque, summed over sub-steps
        self.n = 0

    def set_target(self, pitch_down: float = 0.0, roll_right: float = 0.0):
        self.pitch_down, self.roll_right = float(pitch_down), float(roll_right)

    def torque(self, R, omega):
        z_des = qs.desired_up(qs.heading_of(R), self.pitch_down, self.roll_right)
        w_b = R.T @ np.asarray(omega)
        e_b = R.T @ np.cross(R[:, 2], z_des)       # rotation that brings body z to z_des
        tau_b = np.array([self.kp[0] * e_b[0] - self.kd[0] * w_b[0],
                          self.kp[1] * e_b[1] - self.kd[1] * w_b[1],
                          -self.kd_yaw * w_b[2]])
        return R @ tau_b

    def update(self, R, v, omega, dt):
        t = self.torque(R, omega)
        self.yaw_torque_sum += float(R[:, 2] @ t)
        self.n += 1
        return InnerOutput(torque=t)

    def pop_yaw_torque(self) -> float:
        """Mean body-z haltere torque since the last call (recorded as turn_reflex)."""
        m = self.yaw_torque_sum / self.n if self.n else 0.0
        self.yaw_torque_sum, self.n = 0.0, 0
        return m


class HeadReflex:
    """Head (gaze) stabilisation (HAND reflex): per axis the joint target integrates the counter-rotation of
    the body, d theta*/dt = -G omega_b - theta*/TAU_RC, |theta*| <= THETA_MAX; yaw makes a reset saccade
    (theta*_yaw = 0) when |theta*_yaw| >= YAW_RESET. Position actuators on the three neck joints track it."""

    label = "REFLEX"

    def __init__(self, reflex: dict | None = None):
        r = load("reflex") if reflex is None else reflex
        self.G = np.asarray(r["head_gain"], float)
        self.theta_max = np.radians(r["head_theta_max_deg"])
        self.yaw_reset = np.radians(r["head_yaw_reset_deg"])
        self.tau_rc = r["head_tau_rc"]
        self.kp = r["head_kp"]
        self.turn_rate = r["head_turn_rate"]
        self.target = np.zeros(3)             # yaw, pitch, roll (rad)
        self.n_resets = 0

    def update_body_rates(self, omega_b, dt):
        """omega_b: body angular velocity in thorax axes (x, y, z). Returns joint targets (yaw, pitch, roll)."""
        w = np.array([omega_b[THORAX_AXIS[a]] for a in AXES])
        self.target += dt * (-self.G * w - self.target / self.tau_rc)
        np.clip(self.target, -self.theta_max, self.theta_max, out=self.target)
        if abs(self.target[0]) >= self.yaw_reset:
            self.target[0] = 0.0
            self.n_resets += 1
        return self.target

    def update(self, R, v, omega, dt):
        return InnerOutput(neck_target=self.update_body_rates(R.T @ np.asarray(omega), dt).copy())
