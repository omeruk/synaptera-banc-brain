"""HAND scaffold: take-off timer, cruise, approach, descend, touchdown and the HAND collective / tilt terms.

Ported from synaptera@c94796d:flight/hybrid.py (new checklist item 55, own code). Every term is HAND (label HAND,
recorded separately); the constants are in configs/hand_scaffold.yaml with origin classes and were set in
brain-free runs from geometry and the cruise calibration, never fitted to success (v1 scripts/diag/hyb_landing.py).

NOT carried over: ``turn_hand`` (the odour->turn map K tanh(G I_asym)) and its constants; it is a NeuroFly
adaptation (docs/ADAPTED_CODE.md D.3). The yaw command is the sum of whatever terms the experiment provides
(BRAIN / FLYVIS / REFLEX labels), clipped by :func:`combine_turn`.

Why the approach is world-frame (v1 diagnosis of the v7 hang 29 mm from the food): an altitude target above the
platform top (the extended tarsi hang 1.5-3 mm lower), a horizontal position controller independent of heading,
and a vertical descent only once the thorax is over the platform.
"""
from __future__ import annotations

import numpy as np

from body import quasi_steady as qs
from body.arena import ArenaSpec
from configs import load

H = load("hand_scaffold")
W = load("body_flight")
R = load("readouts")
PHASES = ("takeoff", "cruise", "approach", "descend", "touchdown", "landed")
PHASE_CODE = {p: i for i, p in enumerate(("perch", "takeoff", "cruise", "approach", "touchdown", "feed_extend",
                                          "feed_eat", "feed_retract", "landed", "descend"))}   # v1 codes kept


class HandScaffold:
    """Terms of the HAND scaffold for one arena."""

    def __init__(self, arena: ArenaSpec, wing: dict | None = None, hand: dict | None = None):
        self.arena = arena
        self.w = W if wing is None else wing
        self.h = H if hand is None else hand
        self.platform_top = arena.food_platform[3]

    def altitude_target(self, phase=None) -> float:
        return self.platform_top + self.h["approach_clear"]

    def thrust_hand(self, phase, z, v_z, I_grad=0.0, ablate_odor=False) -> dict:
        """Collective (lift fraction - 1). Returns dict total, odor, alt, boost, vz_des.
        takeoff / cruise: odour vertical gradient + a climb-only assist towards the floor platform top +
        approach_clear (never brakes a climb); approach: altitude target (no odour term); descend: sink at land_vz."""
        h = self.h
        odor, boost = 0.0, 0.0
        if phase == "descend":
            vz_des = -h["land_vz"]
            alt = (vz_des - float(v_z)) / h["vz_damp_ref"]
        elif phase == "approach":
            vz_des = float(np.clip(h["z_gain"] * (self.altitude_target(phase) - z), -h["vz_max"], h["vz_max"]))
            alt = (vz_des - float(v_z)) / h["vz_damp_ref"]
        else:
            vz_des = float(np.clip(h["z_gain"] * (self.altitude_target(phase) - z), 0.0, h["vz_max"]))
            alt = max(vz_des - float(v_z), 0.0) / h["vz_damp_ref"]
            if not ablate_odor:
                odor = h["odor_grad_k"] * float(np.tanh(h["odor_grad_gain"] * I_grad))
            boost = h["takeoff_lift"] if phase == "takeoff" else 0.0
        return dict(total=odor + alt + boost, odor=odor, alt=alt, boost=boost, vz_des=vz_des)

    def platform_offset(self, pos) -> np.ndarray:
        cx, cy = self.arena.food_platform[:2]
        return np.array([cx - pos[0], cy - pos[1]])

    def tilt_hand(self, phase, pos, vel, heading):
        """(pitch_down, roll_right) body tilt targets in rad and the horizontal speed target. Cruise: fixed
        cruise pitch. Approach / descend: world-frame horizontal velocity loop towards the platform centre,
        |v_des| = min(terminal speed, v_gain d); tilt = feed-forward (linear drag, cruise calibration) - brake
        slope (v - v_des)."""
        h, w = self.h, self.w
        if phase in ("takeoff", "touchdown", "landed"):
            return 0.0, 0.0, 0.0
        if phase == "cruise":
            return np.radians(w["cruise_pitch_deg"]), 0.0, w["terminal_speed_ref"]
        off = self.platform_offset(pos)
        d = float(np.linalg.norm(off))
        speed = min(w["terminal_speed_ref"], h["v_gain"] * d)
        v_des = off / max(d, 1e-9) * speed
        ff = np.radians(w["cruise_pitch_deg"]) / w["terminal_speed_ref"]
        brake = np.radians(h["brake_deg_per_100"]) / 100.0
        tilt = ff * v_des - brake * (np.asarray(vel[:2]) - v_des)
        n = float(np.linalg.norm(tilt))
        tmax = np.radians(h["tilt_max_deg"])
        if n > tmax:
            tilt *= tmax / n
        hv = np.array([np.cos(heading), np.sin(heading)])
        rv = np.array([np.sin(heading), -np.cos(heading)])
        return float(tilt @ hv), float(tilt @ rv), speed

    def wing_command(self, turn_total, thrust, pitch_down, roll_right, params: qs.FlightParams, wings_on=True):
        """Wing command and lift fraction for the total tilt (hover amplitude x sqrt(lift))."""
        if not wings_on:
            return qs.WingCommand(0.0, 0.0, 0.0, on=False), 1.0
        lift = float(np.clip(1.0 + thrust, *self.h["lift_frac_range"]))
        tilt = float(np.arccos(np.clip(np.cos(pitch_down) * np.cos(roll_right), -1.0, 1.0)))
        a = qs.hover_amplitude(params, tilt) * np.sqrt(lift)
        aL, aR = qs.turn_to_amplitudes(turn_total, a, self.w["turn_dphi_deg"])
        return qs.WingCommand(aL, aR, params.f0), lift


class HandPhases:
    """takeoff (timed) -> cruise -> approach (d_xy < r_approach) -> descend (over the platform, slow) -> touchdown
    (``touchdown_legs`` tarsi on the platform for ``touchdown_s``) -> landed. All HAND. Feeding is not a phase: MN9
    above threshold while landed (motor/legacy_bridge.py). ``dt``: the coupling interval in s."""

    def __init__(self, scaffold: HandScaffold, dt: float):
        self.s = scaffold
        h = scaffold.h
        self.n_takeoff = max(1, int(round(h["takeoff_s"] / dt)))
        self.n_touch = max(1, int(round(h["touchdown_s"] / dt)))
        self.phase = "takeoff"
        self.t_in = 0
        self.touch_run = 0

    def update(self, pos, vel, platform_legs):
        s, h = self.s, self.s.h
        p = self.phase
        self.t_in += 1
        nxt = p
        d = float(np.linalg.norm(s.platform_offset(pos)))
        v_h = float(np.linalg.norm(vel[:2]))
        if p == "takeoff" and self.t_in >= self.n_takeoff:
            nxt = "cruise"
        elif p == "cruise" and d < h["r_approach"]:
            nxt = "approach"
        elif p == "approach" and d < h["r_descend"] and v_h < h["v_descend"] and pos[2] > s.platform_top:
            nxt = "descend"
        if p in ("cruise", "approach", "descend"):
            self.touch_run = self.touch_run + 1 if platform_legs >= h["touchdown_legs"] else 0
            if self.touch_run >= self.n_touch:
                nxt = "touchdown"
        elif p == "touchdown":
            nxt = "landed"
        if nxt != p:
            self.phase, self.t_in = nxt, 0
        return self.phase

    @property
    def wings_on(self):
        return self.phase in ("takeoff", "cruise", "approach", "descend")

    @property
    def legs_extended(self):
        return self.phase in ("approach", "descend", "touchdown", "landed")

    @property
    def landed(self):
        return self.phase in ("touchdown", "landed")


def combine_turn(*terms, limit=None) -> float:
    """Clipped sum of the turn terms (BRAIN / HAND / FLYVIS ...); each is recorded separately by the caller."""
    lim = R["turn_bias_max"] if limit is None else limit
    return float(np.clip(sum(terms), -lim, lim))
