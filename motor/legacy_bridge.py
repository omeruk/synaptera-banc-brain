"""The hand-made VNC bridge: brain readouts -> motor decisions (CONTROL ARM, label HAND).

Ported from synaptera@c94796d:flight/vnc_bridge.py. Constants live in configs/readouts.yaml; the readout
filter, the MN9 feeding threshold and the yaw normalisation are unchanged:

  filter   per-neuron rates low-pass filtered across intervals (first order, exact discretisation, tau 50 ms)
  feeding  extended = filtered MN9 rate > MN9_THRESHOLD_HZ (FlyGym has no proboscis joint: recorded only)
  yaw      n_t = [(r_L - ref_L) - (r_R - ref_R)] / mean(ref_L, ref_R); s_hat = mean of n over the DRIVE types;
           turn_dn = -K_STEER * s_hat, clipped to +-TURN_BIAS_MAX (a DN more active on the left turns the fly
           to the left, ipsilateral). The reference is the mirror-symmetric front-to-back grating response.
  swap     the two side channels of the steer types are exchanged (s_hat -> -s_hat)
  ablate   the readout is replaced by its perch-calibration baseline (the brain keeps running); v1 reproduction
           only: the constant it produces is realisation-dependent (docs/results/f0_latency_hover_yaw.md)
  zero_turn  control arm for new experiments: the yaw command is exactly 0, s_hat and the readouts are recorded

Changes: the yaw DRIVE types are those of the steer types that pass the symmetry gate (the v1 post-hoc
"DNp15 only" is not inherited; a waived type is a labelled exception recorded in ``Readouts.record``);
ablating or updating without a baseline is an explicit error (review L2); the decision is returned as plain
values for the coupling scheduler's command queue, so the body is never driven by the interval that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from configs import load
from motor.readouts import READOUT_TYPES, STEER_TYPES, Readouts

CFG = load("readouts")
K_STEER = CFG["k_steer"]
READOUT_TAU = CFG["readout_tau"]
MN9_THRESHOLD_HZ = CFG["mn9_threshold_hz"]
TURN_BIAS_MAX = CFG["turn_bias_max"]
PATH_DN_REFERENCE = Path(__file__).resolve().parent.parent / "data" / "dn_lr_reference.json"


class BaselineMissing(RuntimeError):
    """An ablation (or a baseline-dependent update) was used before ``set_baseline`` (review L2)."""


class Bridge:
    def __init__(self, readouts: Readouts, dt: float, ablate=(), swap=(), reference=None, drive=None,
                 zero_turn: bool = False):
        self.ro = readouts
        self.dt = dt
        self.alpha = 1.0 - np.exp(-dt / READOUT_TAU)
        self.types = readouts.types
        self.filt = np.zeros((len(self.types), 2))
        self.baseline = None
        sets = {"steer": STEER_TYPES, "mn9": ("MN9",)}
        self.ablate_rows = sorted({self.types.index(t) for s in ablate for t in sets.get(s, (s,))})
        self.swap_steer = "steer" in swap
        if drive is None:
            drive = readouts.usable_types(STEER_TYPES)
        for t in drive:
            readouts.require(t)                     # a drive type must pass the gate or be waived
        if not drive:
            raise ValueError("no steer type passes the symmetry gate; the yaw drive would be undefined")
        self.drive = tuple(drive)
        self.zero_turn = bool(zero_turn)      # named control arm: yaw command exactly 0, readouts still recorded
        self.ref = None if reference is None else np.asarray(reference, float)

    def set_baseline(self, rates):
        self.baseline = np.asarray(rates, float)
        self.filt = self.baseline.copy()

    def update_hz(self, raw_hz):
        """raw_hz: (n_types, 2) mean rate per neuron in the last interval. Returns (raw, filtered / ablated)."""
        raw = np.asarray(raw_hz, float)
        if self.ablate_rows and self.baseline is None:
            raise BaselineMissing("ablation needs set_baseline() first (perch calibration)")
        self.filt += self.alpha * (raw - self.filt)
        used = self.filt.copy()
        if self.ablate_rows:
            used[self.ablate_rows] = self.baseline[self.ablate_rows]
        return raw, used

    def update(self, counts):
        """counts: (n_types, 2) spikes in the last interval."""
        return self.update_hz(self.ro.rates(np.asarray(counts, float), self.dt))

    def mn9_rate(self, used):
        return float(used[self.types.index("MN9")].mean())

    def proboscis(self, used):
        r = self.mn9_rate(used)
        return dict(mn9_rate=r, proboscis_extended=int(r > MN9_THRESHOLD_HZ))

    def steer(self, used):
        if self.ref is None:
            raise RuntimeError("no steering reference (data/dn_lr_reference.json): pass reference=")
        rows = [self.types.index(t) for t in STEER_TYPES]
        u, ref = used[rows], self.ref[rows]
        norm = steer_norm(u, ref)
        s_hat = float(norm[[STEER_TYPES.index(t) for t in self.drive]].mean()) * (-1.0 if self.swap_steer else 1.0)
        turn = float(np.clip(-K_STEER * s_hat, -TURN_BIAS_MAX, TURN_BIAS_MAX))
        if self.zero_turn:
            turn = 0.0
        return dict(steer_raw=u[:, 0] - u[:, 1], steer_norm_t=norm, steer_norm=s_hat, turn_dn=turn)


def steer_norm(u, ref):
    """(n, 2) rates and references (Hz, columns L, R) -> per-type normalised L-R:
    [(r_L - ref_L) - (r_R - ref_R)] / mean(ref_L, ref_R)."""
    return ((u[:, 0] - ref[:, 0]) - (u[:, 1] - ref[:, 1])) / (0.5 * (ref[:, 0] + ref[:, 1]))


def load_reference(path=PATH_DN_REFERENCE, types=READOUT_TYPES):
    """(n_types, 2) reference rates (Hz per neuron) for the steering normalisation."""
    with open(path) as f:
        r = json.load(f)["rates_hz"]
    ref = np.array([[r[t]["L"], r[t]["R"]] for t in types])
    rows = [types.index(t) for t in STEER_TYPES]
    if np.any(ref[rows].sum(1) <= 0):
        raise ValueError("steering reference rates are 0 on both sides; normalisation undefined")
    return ref
