"""Seeded physical noise for the body (review M5: v1 seeds changed only Brian2, the physics was deterministic).

An Ornstein-Uhlenbeck gust force / torque on the thorax, drawn from its own ``numpy`` generator so that
the run seed changes the physics. Amplitudes are 0 by default (``configs/noise.yaml``): the deterministic
reference. Exact discretisation, so the statistics do not depend on the coupling interval.
"""
from __future__ import annotations

import numpy as np

from configs import load


class PhysicalNoise:
    def __init__(self, seed: int, force_sigma=None, torque_sigma=None, tau=None):
        c = load("noise")
        self.force_sigma = c["gust_force_sigma"] if force_sigma is None else float(force_sigma)
        self.torque_sigma = c["gust_torque_sigma"] if torque_sigma is None else float(torque_sigma)
        self.tau = c["gust_tau"] if tau is None else float(tau)
        self.rng = np.random.default_rng([int(seed), 0xB0D7])
        self.x = np.zeros(6)                       # force (3), torque (3), stationary std = sigma

    @property
    def active(self) -> bool:
        return self.force_sigma > 0 or self.torque_sigma > 0

    def step(self, dt: float):
        """Advance the process by dt (s). Returns (force, torque) in the world frame (uN, uN*mm)."""
        if not self.active:
            return np.zeros(3), np.zeros(3)
        a = np.exp(-dt / self.tau)
        sig = np.r_[np.full(3, self.force_sigma), np.full(3, self.torque_sigma)]
        self.x = a * self.x + np.sqrt(1.0 - a * a) * sig * self.rng.standard_normal(6)
        return self.x[:3].copy(), self.x[3:].copy()
