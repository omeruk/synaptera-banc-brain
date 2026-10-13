"""Constants of prereg/09_f3b1_optomotor.md (committed with the prereg, before any run)."""

INTERVAL_MS = 5.0
N_GREY, N_STIM = 100, 200                    # 0.5 s grey lead-in + 1 s stimulus, 5 ms intervals
WINDOW = (N_GREY, N_GREY + N_STIM)           # readout window (interval indices, end exclusive)
SEEDS = tuple(range(10))
GATE1_SEEDS, GATE2_SEEDS = tuple(range(10)), tuple(range(200, 250))
# name -> (tasks/t0_stimuli.py condition, stimulus code of the VNC seed rule)
STIMULI = {"rot_left": (2, 1), "rot_right": (1, 2), "static": (3, 3), "grey": (10, 4)}
STEER_FAMILIES = ("DNa04", "DNa10", "DNa15")
BRAIN_FAMILIES = STEER_FAMILIES + ("DNp31",)
PRIMARY_MN = ("b3", "i1", "i2", "hg1", "hg4")
BASELINE_HZ = 100.0                          # HAND DNg02, both sides

# F: lowest steering rate (F3a-3 r3.lowest_rate) / FlyWire F0 flight median (C1); prereg/09 section 6
RATE_GAIN = {"DNa04": 1.1649922720247299, "DNa10": 7.5187032418952615, "DNa15": 9.248466257668714}
BRIDGE_WINDOW_INTERVALS = 5                  # E, 25 ms
BRIDGE_LATENCY_INTERVALS = 1                 # E, 5 ms
PERMUTATION = {"DNa04": "DNa10", "DNa10": "DNa15", "DNa15": "DNa04"}     # BANC target <- FlyWire source
ARMS = ("ref", "rate_gain", "bridge_off", "shuffled", "shuffled_gain")
CONTROLS = {"ref": ("bridge_off", "shuffled"), "rate_gain": ("bridge_off", "shuffled_gain")}
INPUT_CHECK_MIN = 0.05


def vnc_seed(stimulus, seed):
    return 100 * STIMULI[stimulus][1] + seed


def bridge_config(arm):
    """BridgeConfig of an arm (None for bridge_off)."""
    from coupling.dn_bridge import BridgeConfig
    common = dict(families=STEER_FAMILIES, interval_ms=INTERVAL_MS, latency_intervals=BRIDGE_LATENCY_INTERVALS,
                  window_intervals=BRIDGE_WINDOW_INTERVALS)
    if arm == "ref":
        return BridgeConfig(**common)
    if arm == "rate_gain":
        return BridgeConfig(gains=dict(RATE_GAIN), **common)
    if arm == "shuffled":
        return BridgeConfig(source=dict(PERMUTATION), **common)
    if arm == "shuffled_gain":
        return BridgeConfig(source=dict(PERMUTATION), gains=dict(RATE_GAIN), gain_by="source", **common)
    if arm == "bridge_off":
        return None
    raise ValueError(arm)
