"""Constants of prereg/10_f3b2_dn_ensemble.md (committed with the prereg, before any run).

Stimuli, seeds, windows, bridge window and latency, baseline and primary MN types are those of prereg/09
(`f3b1_common`); nothing new is fitted. The F3b-1 brain records are reused (prereg/10 section 2).
"""
from f3b1_common import (BASELINE_HZ, BRIDGE_LATENCY_INTERVALS, BRIDGE_WINDOW_INTERVALS, GATE1_SEEDS,  # noqa: F401
                         GATE2_SEEDS, INTERVAL_MS, N_GREY, N_STIM, PRIMARY_MN, SEEDS, STIMULI, WINDOW, vnc_seed)

# the 112 shared DN families passing the symmetry gate in both datasets (docs/results/f3_dn_crosswalk.json
# summary.shared_gate_both; docs/07 section 2); DNg02 is not among them and stays the HAND baseline
ENSEMBLE = (
    "DNa02", "DNa03", "DNa04", "DNa05", "DNa06", "DNa07", "DNa08", "DNa09", "DNa10", "DNa11", "DNa13", "DNa14",
    "DNa15", "DNa16", "DNae002", "DNae003", "DNb01", "DNb02", "DNb05", "DNb06", "DNb07", "DNbe002", "DNg04", "DNg06",
    "DNg07", "DNg09", "DNg100", "DNg103", "DNg104", "DNg106", "DNg111", "DNg13", "DNg14", "DNg18", "DNg19", "DNg20",
    "DNg33", "DNg37", "DNg41", "DNg43", "DNg45", "DNg46", "DNg51", "DNg52", "DNg54", "DNg56", "DNg59", "DNg65",
    "DNg67", "DNg68", "DNg69", "DNg70", "DNg72", "DNg75", "DNg79", "DNg80", "DNg92", "DNg93", "DNg97", "DNg98",
    "DNge024", "DNge051", "DNge053", "DNge062", "DNge084", "DNge122", "DNge142", "DNge148", "DNp01", "DNp04",
    "DNp05", "DNp06", "DNp07", "DNp08", "DNp103", "DNp12", "DNp13", "DNp14", "DNp18", "DNp19", "DNp20", "DNp22",
    "DNp25", "DNp31", "DNp35", "DNp38", "DNp39", "DNp43", "DNp44", "DNp47", "DNp49", "DNp54", "DNp55", "DNp58",
    "DNp59", "DNp60", "DNp62", "DNp64", "DNp66", "DNp67", "DNp69", "DNp70", "DNp73", "DNpe001", "DNpe003", "DNpe008",
    "DNpe014", "DNpe031", "DNpe041", "DNpe048", "MDN", "aSP22",
)
# owner's steering MN list of prereg/08 R3; the secondary test uses those passing the symmetry gate on the subgraph
STEER_MN = ("b1", "b2", "b3", "i1", "i2", "iii1", "iii3", "iii4", "hg1", "hg2", "hg3", "hg4", "tp1", "tp2", "tpn", "ps1",
            "ps2")
# BANC target family <- FlyWire source family: a single cycle over ENSEMBLE (Sattolo's algorithm,
# numpy default_rng(PERMUTATION_SEED), list order as ENSEMBLE); side kept
PERMUTATION_SEED = 20261013
PERMUTATION = {
    "DNa02": "DNp54", "DNa03": "DNpe008", "DNa04": "DNa06", "DNa05": "DNp64", "DNa06": "DNp01", "DNa07": "DNg67",
    "DNa08": "DNg69", "DNa09": "DNg75", "DNa10": "DNge084", "DNa11": "DNbe002", "DNa13": "DNp103", "DNa14": "DNg33",
    "DNa15": "DNpe014", "DNa16": "DNg06", "DNae002": "DNp67", "DNae003": "DNge148", "DNb01": "DNg51",
    "DNb02": "DNg37", "DNb05": "DNge062", "DNb06": "DNg98", "DNb07": "DNa13", "DNbe002": "DNg45", "DNg04": "DNg68",
    "DNg06": "DNg56", "DNg07": "DNp69", "DNg09": "DNge053", "DNg100": "DNa03", "DNg103": "DNg93", "DNg104": "DNp62",
    "DNg106": "DNge122", "DNg111": "DNge051", "DNg13": "DNa15", "DNg14": "MDN", "DNg18": "DNb05", "DNg19": "DNb06",
    "DNg20": "DNae002", "DNg33": "DNg18", "DNg37": "DNp06", "DNg41": "DNg72", "DNg43": "DNg19", "DNg45": "DNp08",
    "DNg46": "DNg20", "DNg51": "DNg04", "DNg52": "DNge024", "DNg54": "DNpe031", "DNg56": "DNp22", "DNg59": "DNa04",
    "DNg65": "DNp44", "DNg67": "DNp13", "DNg68": "DNp43", "DNg69": "aSP22", "DNg70": "DNa02", "DNg72": "DNp38",
    "DNg75": "DNg103", "DNg79": "DNg07", "DNg80": "DNp49", "DNg92": "DNge142", "DNg93": "DNp47", "DNg97": "DNp12",
    "DNg98": "DNg46", "DNge024": "DNg14", "DNge051": "DNa16", "DNge053": "DNb02", "DNge062": "DNa05",
    "DNge084": "DNg13", "DNge122": "DNg54", "DNge142": "DNb07", "DNge148": "DNg97", "DNp01": "DNg79",
    "DNp04": "DNg80", "DNp05": "DNpe048", "DNp06": "DNp20", "DNp07": "DNp31", "DNp08": "DNg59", "DNp103": "DNpe001",
    "DNp12": "DNg106", "DNp13": "DNg111", "DNp14": "DNpe003", "DNp18": "DNp19", "DNp19": "DNpe041", "DNp20": "DNp04",
    "DNp22": "DNg104", "DNp25": "DNg70", "DNp31": "DNp39", "DNp35": "DNp59", "DNp38": "DNp66", "DNp39": "DNp05",
    "DNp43": "DNp14", "DNp44": "DNae003", "DNp47": "DNp58", "DNp49": "DNb01", "DNp54": "DNp70", "DNp55": "DNg52",
    "DNp58": "DNa08", "DNp59": "DNa10", "DNp60": "DNp35", "DNp62": "DNa09", "DNp64": "DNp60", "DNp66": "DNg100",
    "DNp67": "DNa07", "DNp69": "DNg41", "DNp70": "DNp73", "DNp73": "DNp18", "DNpe001": "DNp25", "DNpe003": "DNa14",
    "DNpe008": "DNp55", "DNpe014": "DNp07", "DNpe031": "DNg92", "DNpe041": "DNg09", "DNpe048": "DNa11",
    "MDN": "DNg43", "aSP22": "DNg65",
}
ARMS = ("ensemble", "ensemble_shuffled", "bridge_off")
CONTROLS = {"ensemble": ("bridge_off", "ensemble_shuffled")}


def sattolo(items, seed):
    """Target -> source map of a single random cycle (a derangement) over ``items``."""
    import numpy as np
    rng = np.random.default_rng(seed)
    p = list(range(len(items)))
    for i in range(len(p) - 1, 0, -1):
        j = int(rng.integers(0, i))
        p[i], p[j] = p[j], p[i]
    return {items[i]: items[p[i]] for i in range(len(items))}


def bridge_config(arm):
    """BridgeConfig of an arm (None for bridge_off); gain 1 everywhere."""
    from coupling.dn_bridge import BridgeConfig
    common = dict(families=ENSEMBLE, interval_ms=INTERVAL_MS, latency_intervals=BRIDGE_LATENCY_INTERVALS,
                  window_intervals=BRIDGE_WINDOW_INTERVALS)
    if arm == "ensemble":
        return BridgeConfig(**common)
    if arm == "ensemble_shuffled":
        return BridgeConfig(source=dict(PERMUTATION), **common)
    if arm == "bridge_off":
        return None
    raise ValueError(arm)
