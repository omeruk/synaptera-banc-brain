# Pre-registration 10 — F3b-2: DN ensemble bridge, FlyWire brain -> BANC VNC (open loop, no body)

Status: committed **before any run of this experiment** (CLAUDE.md rule 6). It is not edited afterwards; additions go in
a new file and are labelled post-hoc. Decision context: the project owner's decision after PR #13 (2026-10-13, recorded
in `docs/05_ARCHITECTURE_PLAN.md`). Earlier pre-registrations are not changed.

## 1. Question
F3b-1 (prereg/09) found direction-selective DNa04 and DNa10 in the FlyWire brain, but no direction-selective steering
in the BANC VNC with three bridged families. The owner listed three candidate explanations, to be tested in order:
1. too few bridged DNs;
2. no flight-state gain;
3. the neuron model.

F3b-2 tests explanation 1. When the bridge carries **all shared DN families that pass the symmetry gate in both
datasets**, at gain 1, does the BANC VNC show direction-selective steering under rotating gratings? Is the effect absent
in the controls?

## 2. Brain records: reused (no new brain runs)
- **Check (done before this file was committed, data only).** The F3b-1 brain records `runs/f3b1/brain/<stimulus>_s<seed>.npz`
  hold per-interval spike counts of **all 1,299 FlyWire DNs** (`super_class == descending`).
  - There are 40 files: 4 stimuli × seeds 0-9. Each has `dn_counts` of shape (300, 1299), with one weights hash and one
    model hash across all files.
- **Case that applies: reuse.** The brain stage is not re-run. The stimuli, the input check, the seeds and the FlyWire
  engine are therefore exactly those of prereg/09 sections 3-4.
- The VNC stage records the SHA-256 of every brain file it reads; `scripts/verify_report.py` checks the hashes against
  the files.

## 3. The ensemble (owner's list: the 112 of docs/07)
- **Families.** The ensemble is the 112 shared DN families that pass the symmetry gate in both datasets
  (`docs/results/f3_dn_crosswalk.json` `summary.shared_gate_both`; docs/07 section 2). The list, in the order used
  below:

  DNa02 DNa03 DNa04 DNa05 DNa06 DNa07 DNa08 DNa09 DNa10 DNa11 DNa13 DNa14 DNa15 DNa16 DNae002 DNae003 DNb01 DNb02 DNb05
  DNb06 DNb07 DNbe002 DNg04 DNg06 DNg07 DNg09 DNg100 DNg103 DNg104 DNg106 DNg111 DNg13 DNg14 DNg18 DNg19 DNg20 DNg33 DNg37
  DNg41 DNg43 DNg45 DNg46 DNg51 DNg52 DNg54 DNg56 DNg59 DNg65 DNg67 DNg68 DNg69 DNg70 DNg72 DNg75 DNg79 DNg80 DNg92 DNg93
  DNg97 DNg98 DNge024 DNge051 DNge053 DNge062 DNge084 DNge122 DNge142 DNge148 DNp01 DNp04 DNp05 DNp06 DNp07 DNp08 DNp103
  DNp12 DNp13 DNp14 DNp18 DNp19 DNp20 DNp22 DNp25 DNp31 DNp35 DNp38 DNp39 DNp43 DNp44 DNp47 DNp49 DNp54 DNp55 DNp58 DNp59
  DNp60 DNp62 DNp64 DNp66 DNp67 DNp69 DNp70 DNp73 DNpe001 DNpe003 DNpe008 DNpe014 DNpe031 DNpe041 DNpe048 MDN aSP22

- **DNg02.** DNg02 is not among the 112 (it fails the gate in FlyWire). It stays the `HAND` baseline at 100 Hz on both
  sides, as in prereg/08-09, and is excluded from the brain-driven set.
- **Every other BANC DN** is silent.
- **DNp31** (the F3b-1 modulator) is part of the ensemble, so it is driven here.
- The ensemble includes families that are not flight-related, e.g. MDN (backward walking) and DNp01 (giant fibre).
  The owner's list is used as given.
- Code: `scripts/f3b2_common.py` `ENSEMBLE`. A unit test checks it against the crosswalk JSON.

## 4. The DN bridge (unchanged; `coupling/dn_bridge.py`)
- **Rules.** The same rules as prereg/09 section 5:
  - matching by exact `cell_type` first, then family, else silent; side to side;
  - per-neuron mean rate of the matched FlyWire neurons over a trailing window W = 5 intervals (25 ms), ending at
    interval k - L, with L = 1 (5 ms latency);
  - an independent Poisson process per BANC neuron;
  - **gain 1**.
- **No new parameter.** W and L are the registered `f3b1.bridge_window_intervals` and `f3b1.bridge_latency_intervals`.
- **Mapping on the real tables** (data only, computed before this file was committed, from the F3b-1 brain record
  table and the prereg/08 subgraph):
  - **`ensemble`:** 225 units (221 by exact type, 4 by family). 305 BANC neurons (152 left, 153 right) receive a rate,
    from 321 FlyWire neurons (159 left, 162 right). 0 BANC and 0 FlyWire neurons of these families are unmatched.
  - **`ensemble_shuffled`:** 224 units (matched by source family). The same 305 BANC targets and the same 321 FlyWire
    sources.
- **Record.** The bridge record (units, unmatched neurons, configuration) and the spike accounting per run go to the
  result file, as in prereg/09.

## 5. Arms (VNC stage)
| arm | ensemble BANC DNs receive | gain |
|---|---|---|
| `ensemble` | their own family's FlyWire rate | 1 |
| `ensemble_shuffled` (control) | the FlyWire rate of another ensemble family (fixed permutation below) | 1 |
| `bridge_off` (control, ablation) | 0 Hz (only the `HAND` DNg02 baseline) | - |

- **Fixed permutation.** A single random cycle over the 112 families (a derangement). It is drawn with Sattolo's
  algorithm, `numpy.random.default_rng(20261013)`, over the list order of section 3, and stored literally in
  `scripts/f3b2_common.py` `PERMUTATION`. Side is kept.
- **What the shuffle keeps.** The set of FlyWire rate streams is the same as in `ensemble`. The total applied input is
  not exactly matched, because BANC families differ in neuron count; the applied rate per BANC family and side is
  recorded in both arms.
- **The pairs, as "BANC target <- FlyWire source":**
  DNa02 <- DNp54; DNa03 <- DNpe008; DNa04 <- DNa06; DNa05 <- DNp64; DNa06 <- DNp01; DNa07 <- DNg67; DNa08 <- DNg69;
  DNa09 <- DNg75; DNa10 <- DNge084; DNa11 <- DNbe002; DNa13 <- DNp103; DNa14 <- DNg33; DNa15 <- DNpe014; DNa16 <- DNg06;
  DNae002 <- DNp67; DNae003 <- DNge148; DNb01 <- DNg51; DNb02 <- DNg37; DNb05 <- DNge062; DNb06 <- DNg98; DNb07 <- DNa13;
  DNbe002 <- DNg45; DNg04 <- DNg68; DNg06 <- DNg56; DNg07 <- DNp69; DNg09 <- DNge053; DNg100 <- DNa03; DNg103 <- DNg93;
  DNg104 <- DNp62; DNg106 <- DNge122; DNg111 <- DNge051; DNg13 <- DNa15; DNg14 <- MDN; DNg18 <- DNb05; DNg19 <- DNb06;
  DNg20 <- DNae002; DNg33 <- DNg18; DNg37 <- DNp06; DNg41 <- DNg72; DNg43 <- DNg19; DNg45 <- DNp08; DNg46 <- DNg20;
  DNg51 <- DNg04; DNg52 <- DNge024; DNg54 <- DNpe031; DNg56 <- DNp22; DNg59 <- DNa04; DNg65 <- DNp44; DNg67 <- DNp13;
  DNg68 <- DNp43; DNg69 <- aSP22; DNg70 <- DNa02; DNg72 <- DNp38; DNg75 <- DNg103; DNg79 <- DNg07; DNg80 <- DNp49;
  DNg92 <- DNge142; DNg93 <- DNp47; DNg97 <- DNp12; DNg98 <- DNg46; DNge024 <- DNg14; DNge051 <- DNa16; DNge053 <- DNb02;
  DNge062 <- DNa05; DNge084 <- DNg13; DNge122 <- DNg54; DNge142 <- DNb07; DNge148 <- DNg97; DNp01 <- DNg79;
  DNp04 <- DNg80; DNp05 <- DNpe048; DNp06 <- DNp20; DNp07 <- DNp31; DNp08 <- DNg59; DNp103 <- DNpe001; DNp12 <- DNg106;
  DNp13 <- DNg111; DNp14 <- DNpe003; DNp18 <- DNp19; DNp19 <- DNpe041; DNp20 <- DNp04; DNp22 <- DNg104; DNp25 <- DNg70;
  DNp31 <- DNp39; DNp35 <- DNp59; DNp38 <- DNp66; DNp39 <- DNp05; DNp43 <- DNp14; DNp44 <- DNae003; DNp47 <- DNp58;
  DNp49 <- DNb01; DNp54 <- DNp70; DNp55 <- DNg52; DNp58 <- DNa08; DNp59 <- DNa10; DNp60 <- DNp35; DNp62 <- DNa09;
  DNp64 <- DNp60; DNp66 <- DNg100; DNp67 <- DNa07; DNp69 <- DNg41; DNp70 <- DNp73; DNp73 <- DNp18; DNpe001 <- DNp25;
  DNpe003 <- DNa14; DNpe008 <- DNp55; DNpe014 <- DNp07; DNpe031 <- DNg92; DNpe041 <- DNg09; DNpe048 <- DNa11;
  MDN <- DNg43; aSP22 <- DNg65.
- **Everything else as prereg/09 section 6:**
  - the reference Shiu arm on the prereg/08 subgraph (`w_syn` 0.275 mV, scale 1.0);
  - input groups `DNg02_L`, `DNg02_R` (100 Hz throughout) and one per-neuron group for the 305 bridge targets, with
    rates set at every 5 ms boundary;
  - the full 1.5 s per run;
  - VNC seed = 100 × stimulus code + seed (common noise across arms);
  - 4 stimuli × 10 seeds per arm.
- `bridge_off` is re-run here (the engine's input groups differ from F3b-1). Its result is shown next to the F3b-1
  `bridge_off` for the five primary MN types. This comparison is descriptive.

## 6. Readouts
As prereg/09 section 7:
- every rate is the per-neuron mean over the 1 s stimulus window (intervals 100-299);
- BANC rate per readout type and side; the network rate, the non-driven rate, and the realised rates of the bridge
  targets and of DNg02;
- the 25 ms-binned network rate.

In addition:
- **Applied DN rates per family (owner's item 5).** For every ensemble family and side, the mean applied rate over the
  stimulus window, per arm and stimulus (mean and s.d. over seeds), and its maximum over stimuli. Next to it: the
  realised BANC rate of those neurons.
- **Brain rates per family.** The FlyWire per-neuron rate per ensemble family and side, per stimulus and seed (from the
  reused records).

## 7. Primary test (as prereg/09 section 8)
- **MN types.** b3, i1, i2, hg1, hg4.
- **Statistic.** Per arm, MN type m and seed: D = (R - L)(`rot_right`) - (R - L)(`rot_left`), paired over seeds 0-9.
  - Two-sided one-sample t-test against 0. If D is identically 0, p = 1. If D has zero variance and a non-zero mean,
    p = 0.
  - **Holm over the 5 types, per arm** (alpha 0.05).
- **Decision.** An arm shows **"direction-selective steering"** if at least one type is significant after Holm. The sign
  of the mean D is reported but not required.
- **Specificity.** The controls of `ensemble` are `bridge_off` and `ensemble_shuffled`.
  - `ensemble`'s direction-selective result is **specific** if neither control is direction-selective under the same
    test.
  - Otherwise it is reported as **not specific**.
- **Baseline and sensitivity.** (R - L) under `static` and `grey`, per arm (descriptive). Wilcoxon signed-rank with Holm
  as a sensitivity analysis, not used for the decision.

## 8. Secondary test (owner's item 4; reported separately, not used for the decision)
- **MN types.** Every type of the owner's steering MN list of prereg/08 R3 (`b1 b2 b3 i1 i2 iii1 iii3 iii4 hg1 hg2 hg3
  hg4 tp1 tp2 tpn ps1 ps2`) that passes the symmetry gate on the subgraph. The gate is recomputed in the run with
  `connectome.qa_report.type_symmetry`; prereg/08 found 15 passing (b1 and iii3 fail).
- **Statistic.** The same D and t-test as section 7, with **Holm over the gate-passing types, per arm**.
- **Reported.** Per arm: the significant types and their signs. Also whether each control is direction-selective on
  this set.

## 9. Brain-side check (owner's item 5)
- **Statistic.** For every ensemble family: D_brain = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the FlyWire
  per-neuron rate per side, paired over seeds 0-9.
  - Same t-test. **Holm over the 112 families.**
  - A family **"carries a direction-selective brain signal"** if it is significant after Holm. The list of such families
    and their signs is reported.
- **Descriptive.** Rates per stimulus and family.
- **Consistency check.** For DNa04, DNa10 and DNa15 the uncorrected p-values must equal those of F3b-1. The Holm values
  differ, because the family of tests is larger. This check is descriptive.

## 10. Gates (VNC side, as prereg/09 section 10)
1. **No input.** No drive, seeds 0-9: 0 spikes (25 ms steps, 1 s + 200 ms).
2. **No ignition after the cut.** All bridge input groups at 100 Hz for 1 s, then a 200 ms cut: DNg02 both sides and
   all 305 ensemble BANC neurons.
   - 50 seeds, 200-249.
   - Ignition means a network rate > 1 Hz in the last 25 ms cut step. The gate passes at 0 / 50; the Clopper-Pearson
     95 % CI is reported.
3. **Runaway bound (gate 3').** The network rate (stimulus window, mean over seeds) must be < 5 Hz for every
   arm × stimulus and for the gate-2 condition. The maximum is reported.

If gate 1, 2 or 3' fails: stop and report. The tests are then reported but not used.

## 11. Decision rules
- **Outcome A.** `ensemble` is direction-selective (section 7), and specific. This supports explanation 1 at gain 1.
- **Outcome B.** `ensemble` is direction-selective but not specific. It is reported as such; it does not support
  explanation 1.
- **Outcome C.** `ensemble` is not direction-selective. Explanation 1 is not supported at gain 1, with these bridge
  rules. As the owner instructed, the work stops and is reported; the next step (neuron model v2) is the owner's
  decision.
- No threshold, window, seed, test, gain, family list or permutation is changed afterwards.
- The secondary test and the brain-side check are reported with every outcome. Negative results are reported with
  equal weight.
- **Controls (rule 7).** Here they are `ensemble_shuffled` (activity-matched in the sense of section 5) and
  `bridge_off` (ablation). A degree-matched random VNC is not part of this open-loop step, as in prereg/06-09; this is
  stated in the report.

## 12. Origin classes (rule 2)
- **No new constant.** Gain 1 is the identity of the bridge; W and L are registered (`f3b1.*`, E).
- **Protocol settings of this document (E):** the permutation and its seed (20261013), and the reuse of the prereg/09
  stimuli, seeds and VNC seed rule.
- **Unchanged:** every engine, vision and transduction constant.
