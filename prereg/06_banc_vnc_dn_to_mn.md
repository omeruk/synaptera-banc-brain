# Pre-registration 06 — F3a: BANC VNC open loop, descending neurons -> motor neurons (`banc888_full_v3`)

Status: committed **before any run of this protocol** (CLAUDE.md rule 6). Not to be edited afterwards; additions go in
a new file and are labelled post-hoc. This is a new registration for a new protocol; it does not change
`prereg/05_banc_v3_and_innorm.md` or any earlier one. Decision context (project owner, 2026-10-13, after PR #9): the
BANC whole-brain track stops as a reported result; the main track is the hybrid FlyWire v783 brain -> DNs matched by
type -> BANC v3 VNC. This file tests only the BANC VNC half, open loop. No brain input, no body, no F2 test.

## 1. Data and engine (fixed)
- **Graph:** `banc888_full_v3` (`configs/data_sources.yaml`; v888 edge list `size >= 10`, pair-level, no synapse-count
  threshold, autapses excluded), Codex neuron set (158,262 neurons, engine index order of the Codex table), reference
  sign rule of `prereg/01_banc_sign_rule.md`. Loaded exactly as arm R3 of `prereg/05` (`scripts/f2_innorm_run.py:setup`).
- **Engine:** reference Shiu LIF model, hash-pinned, `w_syn` = 0.275 mV, weight scale 1.0, no per-neuron scaling, Poisson
  input weight unscaled (arm `R` of every earlier prereg). Nothing is tuned.
- **Input:** Poisson drive applied **directly to the DN neurons named below; nothing else is driven.**

## 2. Neuron sets (fixed by cell-type name in the Codex table; counts are reported, not selected)
- **DN groups (`super_class == descending`):** `DNg02` = every neuron whose `cell_type` starts with `DNg02` (types
  `DNg02`, `DNg02_d`, `DNg02_h`), by `side`; `DNa02` = `cell_type == DNa02`; `GF` = `cell_type == DNp01`.
- **Motor-neuron readout types (`super_class == motor` unless stated; one row per `cell_type` and `side`):**
  - *wing* = `DLM1-4, DLM5, DVM1a-c, DVM2a-b, DVM3a-b, b1, b2, b3, i1, i2, iii1, iii3, iii4, hg1, hg2, hg3, hg4, tp1, tp2,
    tpn, ps1, ps2, PSI` (the registered wing-MN list; `MNxm01` and `PSn_u`, the remaining wing-class neurons, are
    reported as "other wing" and are not used in any expectation). *Wing steering* = the subset of these with
    `sub_class == wing_steering_motor_neuron`.
  - *haltere* = every cell type with `cell_class == haltere_motor_neuron`; *neck* = every cell type with
    `cell_class == neck_motor_neuron`; *TTMn* = `cell_type == TTMn`.
- **Symmetry gate (CLAUDE.md rule 5):** for every readout type, `connectome.qa_report.type_symmetry` on this graph
  (input synapses, total and excitatory, `|log2(R/L)| < 0.5` for both). Types present on one side only are reported as
  "not testable". The gate result is printed next to every rate row; no left/right claim is made for a type that fails.

## 3. Drives (1 s of drive, then a 200 ms cut with all inputs 0; seeds 0-9 for every drive)
Time step of the protocol 25 ms: 40 steps drive, 8 steps cut.

| id | driven | rate |
|---|---|---|
| D0 | none | 0 |
| G20 / G50 / G100 | DNg02, both sides | 20 / 50 / 100 Hz |
| GL100 / GR100 | DNg02 left only / right only | 100 Hz |
| AL100 / AR100 | DNa02 left only / right only | 100 Hz |
| F100 | GF (DNp01), both | 100 Hz |

## 4. Readouts (reported, not tuned)
- Per readout type and side: mean firing rate per neuron over the 1 s drive window (Hz), mean and s.d. over seeds, and
  the number of neurons; for every drive. Network rate (all neurons), non-driven rate, driven-DN own rate during drive;
  network rate over the 8 cut steps and in the last cut step (activity after the cut).
- Data only, same graph: direct synapse counts DNp01 -> TTMn and DNp01 -> DLM/DVM; DNg02 -> wing-MN types (reported to
  interpret the rates, not a criterion).

## 5. Gates
1. **No input (gate 1):** drive D0, seeds 0-4: 0 spikes (seeds 5-9 reported alongside).
2. **No ignition after the cut (gate 2):** all six DN groups (DNg02, DNa02, GF; both sides) at 100 Hz for 1 s, cut 200 ms,
   50 seeds 200-249; ignition = network rate > 1 Hz in the last cut step (as `prereg/05`); passes at 0 / 50; the
   Clopper-Pearson 95 % CI is reported.
3. **Runaway bound (gate 3', replaces gate 3 of `prereg/05` for this protocol):** network rate (all neurons, 1 s drive
   window, seed mean) < 5 Hz in every drive of section 3. Reported as the maximum over drives.

## 6. Pre-registered expectations (hit or miss, reported with equal weight)
- **(a)** Bilateral DNg02 drive makes at least one wing-MN type fire >= 5 Hz at G100 (mean over both sides' neurons and 10
  seeds) **and** that type's rate increases with the drive rate: mean(G20) < mean(G50) < mean(G100). Hit if at least one
  wing-MN type satisfies both.
- **(b)** Symmetric drive gives `|log2(R/L)| < 0.5` in the wing-MN types that pass the symmetry gate: evaluated at G100 on
  types with a both-sides mean rate >= 1 Hz (quieter types are listed as "not evaluated"); hit if every evaluated type
  satisfies it and at least one is evaluated.
- **(c)** Unilateral DNa02 drive gives a left/right difference in the wing steering MNs: D = rate(left type) - rate(right
  type); hit if at least one wing-steering type has |D| >= 1 Hz under both AL100 and AR100. The sign of D under each drive
  (ipsilateral > contralateral or the reverse) is reported per type and is not required.
- **(d)** GF -> TTMn **may fail** (the connection is largely electrical and the model has no gap junctions). Expectation:
  TTMn mean rate (both sides) < 5 Hz at F100. Reported; a failure of the underlying pathway here is expected and is not a
  model bug. The DLM/DVM response to F100 is reported next to it.

## 7. Decision rules
- If (a) holds **and** gates 1, 2 and 3' pass: the BANC v3 VNC is usable for the hybrid (interface design `docs/07`).
- Otherwise stop and report; a further route needs a new pre-registration and the project owner's decision.
- No threshold, window, seed, neuron set or protocol is changed afterwards. Expectations (b)-(d) do not enter the
  decision; they are reported for the record. Gate 3' is read as a gate of the decision (the owner's rule names "gates
  1-2"; the stricter reading is registered here).

## 8. Origin classes (CLAUDE.md rule 2)
No new model constant: `w_syn` stays 0.275 mV (M, Shiu et al.). The 5 Hz bounds, the 1 Hz evaluation floor, the 1 Hz
steering difference and the 0.5 symmetry bound are reporting thresholds of this document (E), not model parameters.
