# Pre-registration 08 — F3a-3: BANC VNC subgraph; recruitment (R1), persistence (R2), steering dose-response (R3)

Status: committed **before any run on the subgraph** (CLAUDE.md rule 6). Not to be edited afterwards; additions go in a
new file and are labelled post-hoc. Decision context (project owner, 2026-10-13, after PR #11; recorded in
`docs/05_ARCHITECTURE_PLAN.md`): wing power in the hybrid comes from a `HAND`-labelled tonic DNg02 baseline; the brain's
steering candidates are DNa04, DNa10 and DNa15; DNa02 is dropped as the planned yaw drive; DNp31 is recorded as a
non-lateral modulator; the hybrid uses a BANC VNC **subgraph**, because the FlyWire brain replaces the BANC brain. This
is open loop on that subgraph; nothing here builds or runs the hybrid. Earlier pre-registrations are not changed.

## 1. The subgraph (defined from data only, before any run)
- **Source graph:** `banc888_full_v3`, Codex set, loader `scripts/f3a_vnc_run.py:load` (as prereg/06, 07).
- **Kept neurons:** `region` in {`VNC`, `neck`} (region of the neuron's dominant neuropil, synapse-weighted in + out,
  `connectome.adapters.banc.region_of_neuropil`), **plus** every `super_class == motor`, every
  `super_class == descending` and every ascending neuron (`super_class` in `ascending`, `ascending_visceral_circulatory`,
  `sensory_ascending`), whatever their region.
- **Kept edges:** every edge between two kept neurons, **except edges onto a DN**: DNs are **input nodes** (driven from
  outside, in the hybrid by the FlyWire brain), so they receive no synapse. Ascending neurons are kept as **outputs** and
  recorded; their inputs from removed (BANC brain / SEZ) neurons are removed with those neurons, their inputs from kept
  neurons stay. Motor neurons outside the VNC (head MNs, region SEZ / brain) are kept as the owner asked; they lose their
  brain/SEZ inputs.
- No weight, sign, threshold or parameter changes: the edge rows are the whole graph's rows (signs from the reference
  Shiu rule, per presynaptic neuron, unchanged).
- **Code:** `scripts/f3a3_subgraph.py:vnc_subgraph` (unit test `tests/test_f3a3_subgraph.py`). **Counts** (data only,
  produced by the script, committed with this file as `docs/results/f3a3_subgraph.json`): kept 25,818 of 158,262
  neurons, 3,020,333 of 13,414,125 pairs, 11,585,331 of 41,938,506 synapses. By reason: region 21,327, motor 805,
  descending 1,316, ascending 2,370. **No neuron has `neck` as its dominant region** (0), so the `neck` clause keeps
  nothing beyond the classes above. Removed: the edges onto DNs from kept neurons (136,202 pairs, 518,030 synapses), and
  every edge with a removed endpoint. Ascending: 2,370 neurons (1,933 VNC, 200 SEZ, 226 brain, 11 without region);
  inputs kept 1,621,235 synapses, inputs removed 123,808; outputs kept 1,277,387, outputs removed (to the BANC brain/SEZ)
  715,699. These are the figures the report repeats; `scripts/verify_report.py` checks them against the JSON.
- **Engine:** reference Shiu arm R as prereg/06 sections 1-2 (`w_syn` 0.275 mV, scale 1.0, Poisson input weight
  unscaled), on the subgraph. Spikes are indexed by subgraph index; `meta.global_index` maps to the whole graph.
- **Symmetry gate:** `connectome.qa_report.type_symmetry` recomputed **on the subgraph** for every readout type; that
  value decides which steering MN types enter R3 (the whole-graph value from prereg/07 is reported next to it). The DN
  families are input nodes without inputs here; their gate is the whole-graph value of `docs/results/f3_dn_crosswalk.json`
  (DNa04, DNa10, DNa15 pass in both datasets).

## 2. Protocol (common)
25 ms steps; drive 1 s (40 steps); cut 200 ms (8 steps) unless stated. Seeds 0-9 for every condition (the pairing unit
is the seed); gate 2 seeds 200-249. Readout types and DN groups as prereg/06 (`readout_types`) and prereg/07
(`groups_of`: DNg02 = `cell_type` starts with `DNg02`; other families by family key and `side`). Rates are per-neuron
means over the 1 s drive window. **Network rate** = all subgraph neurons (driven DNs included, as in prereg/07 C2); the
rate of the non-driven neurons is reported next to it.

## 3. R1 — recruitment on the subgraph
- Drives as F3a (prereg/06): `G20`, `G50`, `G100` (DNg02 both sides at 20 / 50 / 100 Hz), `GL100`, `GR100` (one side,
  100 Hz). 10 seeds each.
- **Readout:** per readout type and side, mean and s.d. over seeds, next to the F3a whole-graph value of the same drive
  (`runs/f3a/f3a_v3.json`, same seeds 0-9) and the difference (subgraph - whole). Descriptive; no criterion. Summary per
  drive: the types active (seed mean >= 1 Hz, the evaluation floor of prereg/06) in the subgraph only, in the whole graph
  only, and in both.

## 4. R2 — persistence on the subgraph
- As prereg/07 C2: DNg02 both sides at 100 Hz for 1 s, then a **1 s cut** (40 steps). Seeds 0-9. Per-step network and
  readout MN rates, carriers of the cut activity (with `super_class`, `cell_class`, `cell_type`, `side`, `region`).
- **Criterion (fixed by the owner):** per seed, `A` = network rate in the last 100 ms of the cut (the last 4 steps),
  `B` = network rate in the last 100 ms of the drive (steps 37-40). The seed **decays** if `A < 0.1 B`. **R2 passes** if
  at least 9 of the 10 seeds decay. Reported next to the same quantities from prereg/07 C2 on the whole graph
  (`runs/f3a2/f3a2_v3.json`; descriptive). The non-driven-neuron version of `A / B` is reported, not used.

## 5. R3 — steering dose-response on the subgraph
- **Baseline (`HAND`):** DNg02 both sides at 100 Hz, tonic, in every R3 condition (as prereg/07 C3).
- **Steering drive:** on top of the baseline, unilateral drive of one of `DNa04, DNa10, DNa15`, left only or right only,
  at **10, 25, 50 and 100 Hz** (Poisson input rate). Conditions: `B` and `<fam>_<side>_<rate>`: 1 + 3 x 2 x 4 = 25.
  1 s drive, 200 ms cut, seeds 0-9. `B` is shared by all rates (same seeds).
- **Test (as prereg/07 C3, applied per rate):** per family `f`, side `s`, rate `r`, gate-passing steering MN type `m`
  (owner's list `b1 b2 b3 i1 i2 iii1 iii3 iii4 hg1 hg2 hg3 hg4 tp1 tp2 tpn ps1 ps2`, gate on the subgraph):
  `D = rate_right - rate_left`, `d_seed = D(f_s_r, seed) - D(B, seed)`; two-sided one-sample t-test of `d` against 0
  over the 10 seeds (`d` identically 0: p = 1; zero variance, non-zero mean: p = 0). **Holm per rate** over all tests
  of that rate jointly: 3 families x 2 sides x the gate-passing types (the same joint reading as prereg/07).
- **Decision per family and rate:** the family **steers at rate r** if at least one gate-passing type has both its
  left-drive and right-drive tests significant after Holm (alpha 0.05) with opposite mean `d` signs; sign reported
  (`ipsilateral`: left drive mean `d` < 0 and right drive > 0; `contralateral` the reverse).
- **Lowest steering rate** per family = the smallest r in {10, 25, 50, 100} at which it steers (`none` if none). All
  four rates' outcomes are reported, so a non-monotone pattern is visible. Reported next to the family's FlyWire F0
  flight-window per-neuron rates (left / right) and range from prereg/07 C1 (`docs/results/f3a2_checks.json`), and next
  to the **realised rate** of the driven DN in the subgraph (Poisson input rate is the registered axis; the DN's own
  rate is reported because it can differ, prereg/07 post-hoc).
- **Sensitivity (not used for the decision):** Wilcoxon signed-rank (as prereg/07), Holm per rate. Known limit, written
  now: the smallest exact p at 10 seeds is 0.00195, above the first Holm threshold whenever a rate has more than 25
  tests, so it cannot reach significance then; the uncorrected p values are reported.
- **Other readouts:** network and non-driven rates per condition; the baseline power output (DLM1-4, DLM5, DVM1a-c,
  tp2, ps1) in `B` next to R1 `G100`.

## 6. Gates (definitions of prereg/06 and 07)
1. **No input:** no drive, seeds 0-9: 0 spikes.
2. **No ignition after the cut:** every group driven here (DNg02, DNa04, DNa10, DNa15, both sides) at 100 Hz for 1 s,
   200 ms cut, 50 seeds 200-249; ignition = network rate > 1 Hz in the last cut step; passes at 0 / 50;
   Clopper-Pearson 95 % CI reported.
3. **Runaway bound (gate 3'):** network rate (1 s drive window, seed mean) < 5 Hz in every R1, R2, R3 condition and the
   gate-2 condition; reported as the maximum.

## 7. Decision rules
- If gate 1, 2 or 3' fails: stop and report; R1-R3 are reported but not used for the interface.
- R2 pass/fail as section 4. R1 is descriptive. R3: steering families and lowest steering rate per family, as section 5.
- What follows for F3b (subgraph or not, which families, which rates) is the project owner's decision; nothing is chosen
  automatically here. No threshold, window, seed, test or protocol is changed afterwards; negative results are reported
  with equal weight. Controls (rule 7) are not part of this open-loop pre-check, as in prereg/06 and 07.

## 8. Origin classes (CLAUDE.md rule 2)
No new model constant. The subgraph definition is a structural choice of the owner (E, engineering scaffold, defined
from data). Drive rates, the 10 % decay threshold and the 9 / 10 seeds rule are protocol settings of this document (E);
the baseline is labelled `HAND`.
