# Pre-registration 09 — F3b-1: hybrid optomotor test, FlyWire brain -> BANC VNC (open loop, no body)

Status: committed **before any run of this experiment** (CLAUDE.md rule 6), including the stimulus render. It is not
edited afterwards; additions go in a new file and are labelled post-hoc. Decision context: the project owner's decision
after PR #12 (2026-10-13, recorded in `docs/05_ARCHITECTURE_PLAN.md`). The VNC subgraph of prereg/08 is adopted; DNa04 is
the primary steering family; DN set DNa04, DNa10, DNa15 driven from the FlyWire brain, DNp31 recorded as a modulator,
DNg02 the `HAND` baseline at 100 Hz. This is the first coupled experiment: open loop, without a body and without
VNC -> brain feedback. Earlier pre-registrations are not changed.

## 1. Question
When the FlyWire v783 brain sees a grating rotating left or right, do its DNa04 / DNa10 / DNa15 rates, passed through
the DN bridge, produce direction-selective left/right steering in the BANC v3 VNC subgraph? Is the effect absent in the
controls? Are the FlyWire DN rates direction-selective themselves?

## 2. Design (feed-forward replay)
There is no VNC -> brain feedback and no body, so the brain's output does not depend on the VNC. Therefore:
1. **Brain stage, once per stimulus and seed.** The FlyWire brain runs once per (stimulus, seed). The per-interval spike
   counts of all FlyWire DNs are recorded (5 ms intervals; 1,299 DNs).
2. **VNC stage, once per arm, stimulus and seed.** The recorded counts pass through `coupling/dn_bridge.py` and drive the
   BANC VNC subgraph, separately for every arm.
3. **Equivalence.** The bridge applies the one-interval latency during the replay: the rate applied in VNC interval k
   uses FlyWire spikes up to interval k - 1 only. A coupled run with this latency would give the VNC the same inputs.
   The replay is therefore the coupled experiment, computed in two passes.

Labels (rule 1): FlyWire DN rates `BRAIN`; BANC MN rates `VNC`; the DNg02 baseline `HAND`. The vision path is the
existing `FLYVIS` boundary layer, which enters the brain only.

## 3. Stimuli (the v1 visual screen protocol, `tasks/t0_stimuli.py`)
- **Conditions.** Four body-frame conditions from `tasks/t0_stimuli.py`. They are ray-cast to the raw FlyGym eye images,
  then passed through `Retina.raw_image_to_hex_pxls` -> FlyVis -> boundary layer -> FlyWire input groups `vbnd_L` and
  `vbnd_R`, exactly as the v1 screen (`NeuroFly/scripts/diag/vd_render.py`).
  - `rot_left` = t0 condition 2 (`yaw_ccw`): the grating moves toward positive azimuth, i.e. to the fly's left;
    counter-clockwise seen from above.
  - `rot_right` = t0 condition 1 (`yaw_cw`): the grating moves toward the fly's right; clockwise seen from above.
  - `static` = condition 3 (the same grating, not moving).
  - `grey` = condition 10 (uniform grey).

  Grating parameters are those of the v1 screen: wavelength 30 deg, temporal frequency 2 Hz, contrast 0.8.
- **Timing (the 5 ms stack of the F0 baseline, `docs/results/f0_acceptance.md`).**
  - Coupling interval 5 ms. One FlyVis render per interval (`subframes = 1`, FlyVis dt 5 ms), with the transduction
    calibrated at that step (`data/visual_transduction_5ms.json`).
  - Each run is a **0.5 s grey lead-in** (100 intervals) followed by the **1 s stimulus** (200 intervals). FlyVis is
    reset before every condition, as in v1.
  - The frame of stimulus interval j (0-based) shows the stimulus at time (j + 1) x 5 ms after onset (v1 convention).
    Its boundary-layer rates drive the brain in that interval (v1 screen convention; there is no body, so rule 4 does
    not apply to this sensory input).
  - Rendering is deterministic and is done once, for all seeds.
- **Input check (as v1, criterion fixed here).** Over the stimulus window, A_in = (L - R) / (L + R + 1) of the mean
  T4a + T5a boundary rates.
  - It must have opposite signs under `rot_left` and `rot_right`, with |A_in| >= 0.05 in both.
  - If the check fails, nothing further runs, and the failure is reported.
  - The expected signs are recorded but are not criteria: `rot_right` negative, `rot_left` positive.
- **Seeds 0-9** for every stimulus. The pairing unit is the seed.

## 4. Brain stage (FlyWire v783)
- **Engine.** Reference Shiu arm on FlyWire v783 (`connectome.adapters.flywire783`, Shiu sign rule, `w_syn` and scale as
  F0).
  - Input groups as F0: `sugar` (held at 0 Hz), `vbnd_L` and `vbnd_R` (boundary-layer rates).
  - No other input. Fresh state per run (`reset(seed)`), with brain seed = the seed.
- **Record per run.** Per-interval spike counts of all 1,299 FlyWire DNs (`super_class == descending`), the network
  spike count per interval, and the mean boundary input rate per side.
  - Files: `runs/f3b1/brain/<stimulus>_s<seed>.npz`. These are never overwritten; a run whose file exists is skipped.
- **Memory.** `free -h` is checked before the brain runs. At least 6 GB must be available, or the experiment stops and
  the owner is told.

## 5. The DN bridge (`coupling/dn_bridge.py`, unit tests `tests/test_dn_bridge.py`)
- **Matching.** Exact `cell_type` first, then family (type without a single trailing `_<letter>`), else silent.
  - Side maps to the same side; `center` maps only to `center`.
  - Mapping on the real tables (data only, computed before this file was committed): DNa04, DNa10 and DNa15 match by
    exact type, 1 FlyWire -> 1 BANC neuron per side. That gives 6 units, with 0 unmatched BANC and 0 unmatched FlyWire
    neurons of these families.
  - The FlyWire spikes of the source families that find no unit are recorded per run (here expected 0).
- **Rate.** The rate applied in interval k is the per-neuron mean rate of the matched FlyWire neurons over a trailing
  window of **W = 5 intervals (25 ms)**, ending at interval k - **L**, with **L = 1** (5 ms latency). The window is
  zero-padded before the first interval.
  - W is an engineering choice (E). It was chosen to equal the 25 ms readout bin used since v1, and is fixed here.
  - The time-averaged input rate over a window much longer than W does not depend on W.
- **Re-synthesis.** Each matched BANC neuron gets an independent Poisson process of rate g x rho. The 1 s means are
  unaffected, but spike-timing correlations are lost; this is stated in the report.
- **What the BANC side receives.** Only the DNa04, DNa10 and DNa15 neurons of the subgraph receive a brain rate.
  - DNg02 (`cell_type` starts with `DNg02`, both sides) receives the `HAND` baseline of 100 Hz in every arm, as in
    prereg/08.
  - DNp31 is **recorded on the brain side only** (modulator); its BANC neurons are not driven.
  - Every other BANC DN is silent.
- **Record.** The bridge configuration, the mapping table, the applied rates (1 s mean per unit and arm) and the spike
  accounting are written to the result file.

## 6. Arms (VNC stage)
**Gains of the `rate_gain` arm.** These are computed from existing numbers before any run.
- The formula is g_f = (lowest steering rate of family f in F3a-3) / (median FlyWire F0 flight-window per-neuron rate
  of family f).
  - The numerator comes from `docs/results/f3a3_vnc_subgraph.json` `r3.lowest_rate`.
  - The denominator comes from `docs/results/f3a2_checks.json` `c1.f0.<f>.flight.median`: the median over the family's
    neurons of each neuron's seed-mean rate over the 5 F0 seeds.
- Origin F; `fitted_to`: "open-loop VNC dose-response (F3a-3) and FlyWire F0 rates (C1)". No behaviour is used.

| family | lowest steering rate (Hz) | FlyWire flight median (Hz) | g_f |
|---|---|---|---|
| DNa04 | 25 | 21.459369817578768 | 1.1649922720247299 |
| DNa10 | 100 | 13.300165837479272 | 7.5187032418952615 |
| DNa15 | 50 | 5.406301824212271 | 9.248466257668714 |

`scripts/verify_report.py` recomputes these values from the two JSON files.

| arm | steering families receive | gain |
|---|---|---|
| `ref` | their own FlyWire rate | 1 |
| `rate_gain` | their own FlyWire rate | g_f of the family |
| `bridge_off` (control) | 0 Hz (HAND DNg02 baseline only) | - |
| `shuffled` (control of `ref`) | the rate of a wrong family (fixed permutation below) | 1 |
| `shuffled_gain` (control of `rate_gain`) | the rate of a wrong family (same permutation) | g of the **source** family |

- **Fixed permutation (a derangement, chosen here).** Side is kept.
  - BANC DNa04 receives FlyWire DNa10.
  - BANC DNa10 receives FlyWire DNa15.
  - BANC DNa15 receives FlyWire DNa04.
- **Why `shuffled_gain` uses the source gain.** The three scaled rate streams of `rate_gain` are moved unchanged to the
  wrong targets, so the total scaled input is matched.
- The owner's list names one shuffled control. It is run at both gain settings, so that each test arm has a control at
  its own gain.
- **Engine.** Reference Shiu arm on the prereg/08 subgraph (`w_syn` 0.275 mV, scale 1.0).
  - Input groups: `DNg02_L`, `DNg02_R`, and one per-neuron group for the bridge targets.
  - Rates are set at every 5 ms interval boundary.
- **Duration.** Each VNC run covers the full 1.5 s (grey lead-in + stimulus), with DNg02 at 100 Hz throughout.
- **VNC seeds.** VNC seed = 100 x c + seed, with stimulus code c: `rot_left` 1, `rot_right` 2, `static` 3, `grey` 4.
  - This gives independent VNC noise per stimulus, so `bridge_off` measures how the test behaves under VNC noise alone.
  - The VNC seed is the same in every arm (common noise across arms).

## 7. Readouts
- **Window.** Every rate is the per-neuron mean over the **1 s stimulus window** (intervals 100-299). The same window is
  used on both sides.
- **Brain.** FlyWire DN rate per family (DNa04, DNa10, DNa15, DNp31) and side, per stimulus and seed. Also the FlyWire
  network rate.
- **VNC.** BANC rate per readout type and side (`readout_types` of prereg/06), per arm, stimulus and seed.
  - Also: the realised rate of the bridge targets and of DNg02, the network rate, and the non-driven rate.
  - The 25 ms-binned network rate and the rates of the primary MN types.

## 8. Primary test (fixed by the owner)
- **MN types.** The steering MN types that steered in F3a-3 at any rate for any family: **b3, i1, i2, hg1, hg4**. All
  five pass the symmetry gate on the subgraph (prereg/08 R3).
- **Statistic.** Per arm, MN type m and seed: D = (R - L)(`rot_right`) - (R - L)(`rot_left`), paired over the 10 seeds.
  - Two-sided one-sample t-test of D against 0. If D is identically 0, p = 1. If D has zero variance and a non-zero
    mean, p = 0.
  - **Holm over the 5 MN types, per arm** (alpha 0.05).
- **Decision.** An arm shows **"direction-selective steering"** if at least one MN type is significant after Holm. The
  sign of the mean D is reported but not required.
- **Specificity.** For `ref` the controls are `bridge_off` and `shuffled`; for `rate_gain` they are `bridge_off` and
  `shuffled_gain`.
  - An arm's direction-selective result is **specific** if neither of its controls is direction-selective (same test).
  - Otherwise it is reported as **not specific**.
- **Baseline.** (R - L) per MN type under `static` and under `grey`, per arm, mean and s.d. over seeds. Descriptive.
- **Sensitivity (not used for the decision).** Wilcoxon signed-rank, Holm over the 5 types.

## 9. Brain-side check
- **Main check.** D_brain = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the FlyWire rate, per family DNa04, DNa10,
  DNa15.
  - Same t-test as section 8, Holm over these 3 families.
  - A family is "direction-selective in the brain" if it is significant after Holm. The sign is reported.
- **DNp31.** The same statistic is reported uncorrected (modulator; not part of the Holm family).
- **Descriptive.** Rates per stimulus.

## 10. Gates (VNC side, as prereg/06-08)
1. **No input.** No drive, seeds 0-9: 0 spikes (25 ms steps, 1 s + 200 ms).
2. **No ignition after the cut.** DNg02, DNa04, DNa10 and DNa15, both sides, at 100 Hz for 1 s, then a 200 ms cut.
   - 50 seeds, 200-249.
   - Ignition means a network rate > 1 Hz in the last 25 ms cut step. The gate passes at 0 / 50; the Clopper-Pearson
     95 % CI is reported.
3. **Runaway bound (gate 3').** The network rate (stimulus window, mean over seeds) must be < 5 Hz for every
   arm × stimulus and for the gate-2 condition. The maximum is reported.

If gate 1, 2 or 3' fails: stop and report. The tests are then reported but not used.

## 11. Decision rules and what is not decided here
- The decisions are those of sections 8 and 9. No threshold, window, seed, test, gain or stimulus is changed afterwards.
- Negative results (e.g. no direction selectivity in `ref`, or none in the brain) are reported with equal weight.
- What follows (closed loop, a body, other DN families, the neck MNs) is the project owner's decision.
- Controls (rule 7) here are `bridge_off` and `shuffled`. A degree-matched random VNC is not part of this open-loop step,
  as in prereg/06-08; this is stated in the report.

## 12. Origin classes (rule 2)
- **F:** g_f (`configs/assumptions.json` `f3b1.rate_gain`, fitted_to as section 6).
- **E:** the bridge window W = 5 intervals and the latency L = 1 interval (registered as `f3b1.bridge_window_intervals`
  and `f3b1.bridge_latency_intervals`). Also E: the grey lead-in, the stimulus length, the seeds, the VNC seed rule and
  the permutation (protocol settings of this document).
- **Unchanged:** every engine, vision and transduction constant (registered earlier).
