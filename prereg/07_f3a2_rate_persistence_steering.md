# Pre-registration 07 — F3a-2: DN rate range (C1), persistence after the cut (C2), steering on top of power (C3)

Status: committed **before any run or analysis of this file's checks** (CLAUDE.md rule 6). Not to be edited afterwards;
additions go in a new file and are labelled post-hoc. Decision context (project owner, 2026-10-13, after PR #10):
`docs/07_HYBRID_INTERFACE.md` section 5 was answered (matching by family with exact type first; per-neuron mean rate as
default, total conservation as a registered sensitivity arm later; first DN set DNg02, DNa02, DNp31, DNa04, DNa10, DNa15,
DNa16; the 112 gate-passing families later as a second arm), and three pre-checks are run **before any implementation
of the hybrid**. Nothing here builds or runs the hybrid. This file does not change `prereg/06` or any earlier one.

## 0. The seven DN families (fixed)
`DNg02, DNa02, DNp31, DNa04, DNa10, DNa15, DNa16`. Family key = `cell_type` with a single trailing `_<letter>` removed
(`scripts/f3_dn_crosswalk.py:family`); DN = `super_class == descending`. Counts per side and symmetry gate (input
synapses, total and excitatory, |log2(R/L)| < 0.5) from `docs/results/f3_dn_crosswalk.json` (data only, already
committed): DNg02 FlyWire 13/12, gate FAIL (+0.58); BANC 17/18, pass. The other six are 1/1 in both datasets and pass
the gate in both.

## 1. C1 — FlyWire per-neuron rates of the seven families (data only, no simulation)
**Which case applies (decided before any analysis):** the existing records hold **per-neuron spikes**, so they are used
and **no F0 re-run is made**:
- **F0 records:** `runs/f0/f0_5ms_seed{0..4}_data.h5` (the F0 baseline at the current defaults: coupling interval
  5 ms, latency 1, seeds 0-4; `meta/coupling`). `spikes/i, spikes/t` hold every spike of all 138,639 neurons (engine
  index = row of `Completeness_783.csv`, SHA-256 `bbb847a4…`, the `meta/sha256` completeness hash). Time origin: engine
  start; perch calibration 50 intervals (0-0.25 s), input cut 40 intervals (0.25-0.45 s), flight interval k at
  [0.45 + 0.005 k, 0.45 + 0.005 (k + 1)) s, k = 0..1499.
- **v1 visual DN screen records:** `~/Documents/NeuroFly/logs/vis_dn/main/vd_c{01..10}_s{601..605}.npz` (Synaptera v1
  validation seeds 601-605, the 10 visual conditions of `scripts/diag/vis_dn_report.py:COND_NAMES`; per-step 25 ms spike
  counts of all 1,299 FlyWire DNs; `dn_root_idx` = engine index; checked: same `Completeness_783.csv` hash and the same
  DN order as `brain_model/descending_neurons.csv`). Window as in v1: steps 28-59 (200-1000 ms after stimulus onset,
  0.8 s). These are v1 records (v1 engine and input path), reported as such; nothing is re-run.

**Neurons:** FlyWire `cell_type` and `side` of this repository's FlyWire adapter (`connectome.adapters.flywire783`), by
engine index; family key as in section 0.

**Windows (F0, fixed):** `perch` = perch calibration, 0-0.25 s (wings off); `flight` = flight intervals of phases
takeoff, cruise, approach and descend (`behavior/phase` codes 1, 2, 3, 9); `landing` = touchdown and landed (codes 4, 8).
The input-cut window (0.25-0.45 s) is excluded. (In these records the phase sequence is identical in all five seeds,
because the hand/FlyVis terms are replayed from the v1 record; F0 is a port regression, not a behaviour result.)

**Readouts (reported, no criterion on the hybrid):**
- Per neuron, window and seed: mean rate (Hz) over the window; per neuron and window: mean over seeds, and the maximum
  over 25 ms bins (5 consecutive intervals) pooled over seeds (transient peak).
- Per family and window: number of neurons per side, median / min / max of the per-neuron seed-mean rate, and the
  fraction of neurons in each **BANC recruitment bin** taken from `docs/results/f3a_banc_vnc.json`: `< 20 Hz` (below the
  rate at which F3a drove no wing MN, G20), `20-50 Hz`, `50-100 Hz` (the range in which F3a recruited DLM/DVM/tp2), and
  `>= 100 Hz`. Same table for the v1 screen (per condition).
- **DNg02 "nearly silent" (docs/02 §1.5), decision quantity fixed now:** confirmed if, in every F0 window, the median
  over DNg02 neurons of the seed-mean rate is < 1 Hz **and** no DNg02 neuron has a seed-mean rate >= 20 Hz; refuted
  otherwise. The same statement is reported for the v1 screen (every condition) as a second, separate line.

## 2. C2 — BANC v3 VNC: persistence after the cut
- **Graph and engine:** as `prereg/06` sections 1-2 (`banc888_full_v3`, Codex set, reference Shiu arm R: `w_syn`
  0.275 mV, scale 1.0, no scaling, Poisson input weight unscaled; loader `scripts/f3a_vnc_run.py:load`). Nothing tuned.
- **Drive:** DNg02 (both sides; `cell_type` starts with `DNg02`, as prereg/06) at 100 Hz for 1 s (40 steps of 25 ms),
  then a **1 s cut** (40 steps, all inputs 0). Seeds 0-9.
- **Readouts:** per 25 ms step: network rate (all neurons) and the rate of every readout MN type and side of `prereg/06`
  section 2 (wing, other wing, haltere, neck, TTMn); mean over seeds. Per seed: network rate in the last cut step, and
  the time of the last spike in the network after the cut. **Carriers:** every neuron with >= 1 spike during the cut in
  any seed, with `super_class`, `cell_class`, `cell_type`, `side`, number of seeds in which it spiked during the cut,
  and its total cut spikes; aggregated per `super_class` and per `cell_class`. Descriptive; no criterion.

## 3. C3 — BANC v3 VNC: steering on top of power
- **Graph and engine:** as C2.
- **Baseline (labelled `HAND`):** DNg02 both sides at 100 Hz, tonic, in every C3 condition. Label `HAND` because it is
  a flight-power scaffold chosen by hand (the role of flybody's wingbeat pattern generator and of the programmed wing
  power of the Uploading Lab), not a brain output. It is recorded as such in the record (`baseline_label: HAND`).
- **Steering drive:** on top of the baseline, unilateral 100 Hz drive of one family: `DNa02, DNp31, DNa04, DNa10,
  DNa15, DNa16`, left only and right only (BANC family key, `side`). Conditions: `B` (baseline alone) and `<fam>_L`,
  `<fam>_R` for the six families: 13 conditions. Each 1 s drive (40 steps of 25 ms), then a 200 ms cut (8 steps), seeds
  0-9 for every condition (same seed list in every condition: the pairing unit is the seed).
- **Steering MN types (owner's list):** `b1, b2, b3, i1, i2, iii1, iii3, iii4, hg1, hg2, hg3, hg4, tp1, tp2, tpn, ps1,
  ps2`. Symmetry gate per type by `connectome.qa_report.type_symmetry` on this graph (as prereg/06). **Only the types that
  pass enter the decision**; the others are reported separately, descriptively. (F3a found b1 and iii3 failing; the
  gate is recomputed in the run and that value is used.)
- **Quantity:** per condition, seed and MN type: `D = rate_right - rate_left` (per-neuron mean rate over the 1 s drive
  window, Hz). Per family `f`, drive side `s` and MN type `m`: `d_seed = D(f_s, seed) - D(B, seed)` over seeds 0-9.
- **Primary test:** two-sided one-sample t-test of `d` against 0 over the 10 seeds (= the paired t-test of the two
  conditions' `D`). A `d` that is identically 0 gets p = 1; a `d` with zero variance and non-zero mean gets p = 0.
  **Holm correction over all tests that enter the decision** jointly: 6 families x 2 drive sides x the gate-passing MN
  types (the joint set is the conservative reading of "over families x gate-passing MN types"; it is fixed now).
- **Decision per family:** the family **steers** if there is at least one gate-passing MN type `m` for which the
  left-drive test and the right-drive test are both significant after Holm (alpha 0.05) **and** their mean `d` have
  opposite signs. The sign is reported, not required: `ipsilateral` if left drive gives mean `d` < 0 (the left MN gains
  relative to the right) and right drive gives mean `d` > 0; `contralateral` for the reverse.
- **Sensitivity (reported next to it, not used for the decision):** the same tests with the two-sided Wilcoxon
  signed-rank test (`scipy.stats.wilcoxon`, `zero_method="wilcox"`, exact where scipy uses it; `d` identically 0 gives
  p = 1), Holm over the same set. Known limit written now: with 10 seeds the smallest exact two-sided Wilcoxon p is
  2/1024 = 0.00195, which is above the Holm threshold for the first rank (0.05 / number of tests) whenever there are
  more than 25 tests; the uncorrected Wilcoxon p values are therefore reported as well.
- **Other readouts:** per condition mean and s.d. over seeds of every readout MN type and side (as prereg/06), network
  rate, non-driven rate, driven-neuron rates; the baseline power output (DLM, DVM, tp2, ps1) in `B` against F3a `G100`.

## 4. Gates (as `prereg/06` section 5; same definitions)
1. **No input:** D0, seeds 0-4: 0 spikes.
2. **No ignition after the cut:** every group driven in C2/C3 (DNg02 both sides and the six families both sides) at
   100 Hz for 1 s, 200 ms cut, 50 seeds 200-249; ignition = network rate > 1 Hz in the last cut step; passes at 0 / 50;
   Clopper-Pearson 95 % CI reported.
3. **Runaway bound (gate 3'):** network rate (all neurons, 1 s drive window, seed mean) < 5 Hz in every C2 and C3
   condition and in the gate-2 condition; reported as the maximum.

## 5. Decision rules
- If gate 1, 2 or 3' fails: stop and report; C2/C3 results are reported but not used for the interface.
- C1 and C2 are descriptive pre-checks for the interface pre-registration (F3b); they have no pass/fail beyond the DNg02
  confirm/refute statement of section 1.
- C3: the list of families that steer (with sign and MN types), and the families that do not, are reported. Which
  families go into F3b is the project owner's decision; nothing is chosen automatically here.
- No threshold, window, seed, test or protocol is changed afterwards. Negative results are reported with equal weight.

## 6. Origin classes (CLAUDE.md rule 2)
No new model constant. The 100 Hz baseline and steering rates are protocol settings of this document (E, engineering
scaffold; the baseline is labelled `HAND`). The 1 Hz silence threshold and the recruitment bins (20/50/100 Hz) are
reporting thresholds of this document (E); the bins are read from F3a.
