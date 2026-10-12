"""Stroke-averaged (quasi-steady) wing forces. Pure numpy.

Ported from synaptera@c94796d:flight/quasi_steady.py. Wings are not simulated; each wing produces a
stroke-averaged lift F_w = (W/2) (Phi_w/Phi0)^2 (f/f0)^2 along the stroke-plane normal (body +z, optionally
tilted by ``stroke_plane_deg`` about body y), with the flapping counter-force / torque of Hedrick et al. 2009:
    FCF = -c_F s v,   FCT = -c_T s omega,   s = f mean(Phi) / (f0 Phi0)
Wing torques (body frame): roll = (F_L - F_R) arm_lat; yaw = -k_yaw (Phi_L - Phi_R)/Phi0.

Changes with respect to v1: amplitude limit ``phi_max_deg`` with a clip record (review H5); stroke-plane angle
and anisotropic damping options (M9, default off); the haltere attitude reflex is no longer part of the body
model, it is the REFLEX arm in motor/legacy_reflexes.py.

Frames: world z up; body x forward, y left, z up (FlyGym / MuJoCo). Units: mm, s, g -> force in uN, torque
in uN*mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WingCommand:
    amp_L: float     # deg
    amp_R: float     # deg
    freq: float      # Hz
    on: bool = True  # False: wings folded, no aerodynamic force / torque


@dataclass
class FlightParams:
    weight: float        # uN (m g)
    phi0: float          # deg
    f0: float            # Hz
    c_F: float           # uN s/mm  translational counter-force coefficient
    c_T: float           # uN mm s  rotational counter-torque coefficient
    arm_lat: float       # mm
    k_yaw: float         # uN mm per unit (Phi_L - Phi_R)/Phi0
    inertia: np.ndarray  # (3,) principal moments about the COM in body axes, g mm^2
    d_yaw_total: float   # total yaw damping, uN mm s (Izz / yaw_damping_tau)
    phi_max: float       # deg, amplitude limit (H5)
    stroke_plane_rad: float = 0.0
    damp_axial_ratio: float = 1.0


def make_params(mass, gravity, inertia_body, wing: dict) -> FlightParams:
    """Coefficients from the body (mass g, |g| mm/s^2, 3x3 inertia about the COM in body axes, g mm^2) and the
    ``configs/body_flight.yaml`` constants ``wing`` (see :func:`configs.load`)."""
    W = mass * gravity
    theta = np.radians(wing["cruise_pitch_deg"])
    s_cruise = 1.0 / np.sqrt(np.cos(theta))
    c_F = W * np.tan(theta) / (s_cruise * wing["terminal_speed_ref"])
    diag = np.diag(inertia_body).astype(float)
    c_T = diag[2] / wing["rot_damp_tau"]
    d_yaw_total = diag[2] / wing["yaw_damping_tau"]
    k_yaw = d_yaw_total * wing["yaw_rate_per_turn"] / (2 * wing["turn_dphi_deg"] / wing["phi0_deg"])
    return FlightParams(W, wing["phi0_deg"], wing["f0_hz"], c_F, c_T, wing["wing_arm_lat"], k_yaw, diag,
                        d_yaw_total, wing["phi_max_deg"], np.radians(wing.get("stroke_plane_deg", 0.0)),
                        float(wing.get("damp_axial_ratio", 1.0)))


def wing_lift(amp, freq, p: FlightParams):
    return 0.5 * p.weight * (amp / p.phi0) ** 2 * (freq / p.f0) ** 2


def hover_amplitude(p: FlightParams, tilt_rad=0.0, freq=None):
    """Symmetric amplitude whose vertical lift component equals the weight."""
    f = p.f0 if freq is None else freq
    return p.phi0 * (p.f0 / f) / np.sqrt(np.cos(tilt_rad))


def clip_amplitude(amp, p: FlightParams):
    """(clipped amplitude, bool clipped). Amplitudes are limited to [0, phi_max] (review H5)."""
    c = float(np.clip(amp, 0.0, p.phi_max))
    return c, c != float(amp)


def stroke_drive(cmd: WingCommand, p: FlightParams):
    return cmd.freq * 0.5 * (cmd.amp_L + cmd.amp_R) / (p.f0 * p.phi0)


def aero_wrench(R, v, omega, cmd: WingCommand, p: FlightParams):
    """Force / torque (world frame, about the COM) from the wings, with the amplitude limit applied.
    R: body->world 3x3; v: COM velocity (world); omega: angular velocity (world).
    Returns (force, torque, info) with info = F_L, F_R, s and clipped_L / clipped_R."""
    if not cmd.on:
        return np.zeros(3), np.zeros(3), dict(F_L=0.0, F_R=0.0, s=0.0, clipped_L=False, clipped_R=False)
    amp_L, clip_L = clip_amplitude(cmd.amp_L, p)
    amp_R, clip_R = clip_amplitude(cmd.amp_R, p)
    F_L = wing_lift(amp_L, cmd.freq, p)
    F_R = wing_lift(amp_R, cmd.freq, p)
    s = cmd.freq * 0.5 * (amp_L + amp_R) / (p.f0 * p.phi0)
    n = R[:, 2]
    if p.stroke_plane_rad:                       # tilt the stroke-plane normal about body y (M9 option)
        c, sn = np.cos(p.stroke_plane_rad), np.sin(p.stroke_plane_rad)
        n = R @ np.array([sn, 0.0, c])
    v = np.asarray(v, float)
    drag = p.c_F * s * v
    if p.damp_axial_ratio != 1.0:                # anisotropic counter-force along body z (M9 option)
        z = R[:, 2]
        drag = drag + (p.damp_axial_ratio - 1.0) * p.c_F * s * (z @ v) * z
    force = (F_L + F_R) * n - drag
    tau_b = np.array([(F_L - F_R) * p.arm_lat, 0.0, -p.k_yaw * (amp_L - amp_R) / p.phi0])
    torque = R @ tau_b - p.c_T * s * np.asarray(omega)
    return force, torque, dict(F_L=F_L, F_R=F_R, s=s, clipped_L=clip_L, clipped_R=clip_R)


def desired_up(heading, pitch_down, roll_right=0.0):
    """Desired body +z (world) for a heading (rad, yaw), nose-down tilt and right bank."""
    h = np.array([np.cos(heading), np.sin(heading), 0.0])
    r = np.array([np.sin(heading), -np.cos(heading), 0.0])
    z = (np.cos(pitch_down) * np.cos(roll_right) * np.array([0.0, 0.0, 1.0])
         + np.sin(pitch_down) * h + np.sin(roll_right) * r)
    return z / np.linalg.norm(z)


def heading_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_torque_to_turn(tau_z, p: FlightParams, dphi_per_turn):
    """turn_bias whose wing yaw torque equals tau_z (body z, uN mm): the inverse of the yaw term of
    aero_wrench, tau_z = -k_yaw 2 dphi turn / phi0."""
    return -float(tau_z) * p.phi0 / (p.k_yaw * 2.0 * dphi_per_turn)


def turn_to_amplitudes(turn_bias, amp_collective, dphi_per_turn):
    """turn_bias > 0 -> turn right -> larger LEFT amplitude."""
    d = dphi_per_turn * turn_bias
    return amp_collective + d, amp_collective - d
