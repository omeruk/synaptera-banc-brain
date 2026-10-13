"""Constants of prereg/11_f3b3_specificity.md (committed with the prereg, before any run).

Stimuli, intervals, readout window, bridge window and latency, DNg02 baseline, primary MN types and the 112-family
ensemble are those of prereg/09-10 (`f3b1_common`, `f3b2_common`). New here: fresh brain seeds, the VNC seed rule,
the gate-2 seeds and 39 random family derangements.
"""
import hashlib
import json

from f3b1_common import (BASELINE_HZ, BRIDGE_LATENCY_INTERVALS, BRIDGE_WINDOW_INTERVALS, INPUT_CHECK_MIN,  # noqa: F401
                         INTERVAL_MS, N_GREY, N_STIM, PRIMARY_MN, STIMULI, WINDOW)
from f3b2_common import ENSEMBLE, STEER_MN  # noqa: F401

SEEDS = tuple(range(10, 30))                 # fresh brain seeds; the F3b-1 seeds 0-9 are not used
GATE1_SEEDS, GATE2_SEEDS = tuple(range(10, 20)), tuple(range(250, 300))
N_RANDOM = 39
# one seed per random wiring random_k, k = 1..39 (k-th seed for random_k)
PERM_SEEDS = tuple(range(20261101, 20261101 + N_RANDOM))
RANDOM_ARMS = tuple(f"random_{k}" for k in range(1, N_RANDOM + 1))
ARMS = ("ensemble", "bridge_off") + RANDOM_ARMS
PERM_ALPHA = 0.05
# SHA-256 of permutations_json() (fixed in prereg/11 section 5)
PERMUTATIONS_SHA256 = "cf26524b6c1f32d6ee57b9d7ef13858e80d905c8ceab5daee30f99e8fac84f1f"


def vnc_seed(stimulus, seed):
    """VNC engine seed: 1000 x stimulus code + brain seed (common noise across arms; disjoint from the gate seeds)."""
    return 1000 * STIMULI[stimulus][1] + seed


def derangement(items, seed):
    """Target -> source map of a uniform random derangement of ``items``: numpy default_rng(seed).permutation,
    redrawn from the same generator until no item maps to itself."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(items)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return {items[i]: items[int(p[i])] for i in range(n)}


def permutations():
    """{random_k: target -> source map}, k = 1..39."""
    return {arm: derangement(ENSEMBLE, s) for arm, s in zip(RANDOM_ARMS, PERM_SEEDS)}


def permutations_json():
    return json.dumps(permutations(), sort_keys=True, separators=(",", ":"))


def permutations_sha256():
    return hashlib.sha256(permutations_json().encode()).hexdigest()


def bridge_config(arm, perms=None):
    """BridgeConfig of an arm (None for bridge_off); gain 1 everywhere."""
    from coupling.dn_bridge import BridgeConfig
    common = dict(families=ENSEMBLE, interval_ms=INTERVAL_MS, latency_intervals=BRIDGE_LATENCY_INTERVALS,
                  window_intervals=BRIDGE_WINDOW_INTERVALS)
    if arm == "ensemble":
        return BridgeConfig(**common)
    if arm == "bridge_off":
        return None
    if arm in RANDOM_ARMS:
        return BridgeConfig(source=dict((perms or permutations())[arm]), **common)
    raise ValueError(arm)
