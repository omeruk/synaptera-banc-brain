"""Geometric sensors of the flight loop: gaze frame, platform azimuth and retinal expansion.

Ported from synaptera@c94796d:flight/sensors.py. NOT carried over (checklist item 19): ``LoomBias`` (a NeuroFly
adaptation, docs/ADAPTED_CODE.md D.4), the ``l_eff = 10 mm`` antenna separation and the ``antenna_real`` flag.
The odour sampling (``sample_odor``, ``odor_norm``) belongs to the deferred olfaction sub-plan (items 20-22),
and ``surface_below`` is now ``ArenaSpec.surface_below``.

Everything here is computed from the body state; nothing reads the food position except the geometric platform
signals of the HAND landing trigger (recorded and labelled as such).
"""
from __future__ import annotations

import numpy as np

from body.arena import ArenaSpec


def gaze_frame(R) -> np.ndarray:
    """Head sampling frame: body heading only, roll / pitch removed (columns = forward, left, world up). HAND
    stand-in for head / gaze stabilisation: with the body frame the cruise tilt and the bank of a turn mixed
    sin(pitch) x (forward gradient) into the vertical gradient and the vertical gradient into the lateral one."""
    h = float(np.arctan2(R[1, 0], R[0, 0]))
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def platform_azimuth(arena: ArenaSpec, pos, heading) -> float:
    """Horizontal bearing of the food platform relative to the heading (deg, + = left). Recorded only."""
    cx, cy = arena.food_platform[:2]
    a = np.arctan2(cy - pos[1], cx - pos[0]) - heading
    return float(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))


def platform_angular_radius(arena: ArenaSpec, pos) -> float:
    cx, cy, r, zt = arena.food_platform
    d = float(np.linalg.norm(np.asarray(pos) - np.array([cx, cy, zt])))
    return float(np.arctan2(r, max(d, 1e-6)))


class ExpansionRate:
    """(d theta / dt) / theta of the platform's angular radius between intervals (geometric landing signal)."""

    def __init__(self, arena: ArenaSpec, dt: float):
        self.arena = arena
        self.dt = dt
        self.prev = None

    def __call__(self, pos) -> float:
        th = platform_angular_radius(self.arena, pos)
        e = 0.0 if self.prev is None else (th - self.prev) / (self.dt * th)
        self.prev = th
        return e
