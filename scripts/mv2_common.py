"""Constants of prereg/12_neuron_model_v2.md (committed with the prereg, before any run).

Arms R, A, AC, ACG (engine/model_variants.py). Tests (i) FlyWire sugar GRN -> MN9 and ignition, (ii) BANC VNC subgraph
(the prereg/08 protocol), (iii) hybrid optomotor (the prereg/11 protocol, fresh brain seeds 30-39, 19 random wirings),
(iv) GF -> TTMn on the BANC VNC subgraph.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "mv2"
ARMS = ("R", "A", "AC", "ACG")
VARIANT_ARMS = ("A", "AC", "ACG")
REDUCED_ARMS = ("R", "ACG")                 # if the projected run time exceeds RUNTIME_LIMIT_H (owner rule)
RUNTIME_LIMIT_H = 6.0

# (i) FlyWire: 36 labellar sugar GRNs (data/sugar_grn_783.csv) at 100 Hz for 1 s, then a 200 ms cut; MN9 = CB0701
T1_SEEDS = tuple(range(300, 350))           # 50 fresh seeds
T1_STEP_MS, T1_N_ON, T1_N_CUT = 25.0, 40, 8
T1_RATE_HZ = 100.0
T1_MN9_TYPE = "CB0701"
T1_MN9_MIN_FRACTION = 0.10                  # pass: MN9 rate >= 10 % of R
T1_IGNITION_HZ = 1.0                        # network rate in the last cut step > 1 Hz

# (ii) BANC VNC subgraph: prereg/08 conditions and seeds (scripts/f3a3_vnc_run.py), unchanged

# (iii) hybrid optomotor: prereg/11 protocol
T3_SEEDS = tuple(range(30, 40))             # fresh brain seeds (F3b-1 0-9 and F3b-3 10-29 are not used)
T3_N_RANDOM = 19                            # random_1 .. random_19 of prereg/11 (same permutation seeds)
T3_ARMS = ("ensemble", "bridge_off") + tuple(f"random_{k}" for k in range(1, T3_N_RANDOM + 1))
T3_ALPHA = 0.05
T3_PRIMARY_SET = "lateral"                  # scripts/f4_design_numbers.py SETS_V2 (Lindsay 2017 signs)
T3_SECONDARY_SETS = ("direct", "lateral+iii4(+)", "lateral+iii4(-)", "lateral+iii1(+),iii4(+) (text)",
                     "lateral+iii1(-),iii4(-) (Fig. 6)")

# (iv) GF -> TTMn on the BANC VNC subgraph
T4_SEEDS = tuple(range(0, 10))
T4_RATE_HZ = 100.0
T4_STEP_MS, T4_N_ON, T4_N_CUT = 25.0, 40, 8
T4_ARMS = ("R", "ACG")
T4_FOLLOW_MIN = 0.9                         # pass: TTMn spikes / GF spikes >= 0.9 per side
T4_LATENCY_MAX_MS = 1.0                     # pass: median GF -> TTMn latency < 1 ms (Phelan et al. 2008)


def model_variant(arm, cx):
    """ModelVariant of an arm on connectome ``cx`` (None for R). The gap pairs come from configs/gap_junctions.yaml."""
    from engine import model_variants as MV
    if arm == "R":
        return None
    return MV.ModelVariant.arm(arm, gap_pairs=MV.gap_pairs(cx) if "gap" in MV.ARMS[arm] else ())
