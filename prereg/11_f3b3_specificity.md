# Pre-registration 11 — F3b-3: confirmatory specificity test of the DN ensemble (open loop, no body)

Status: committed **before any run of this experiment** (CLAUDE.md rule 6). It is not edited afterwards; additions go in
a new file and are labelled post-hoc. Decision context: the project owner's decision after PR #14 (2026-10-13, recorded
in `docs/05_ARCHITECTURE_PLAN.md`). Earlier pre-registrations are not changed.

## 1. Question
F3b-2 (prereg/10) found outcome B:
- `ensemble` and its single shuffled control were both direction-selective, so the result was not specific.
- Post-hoc observation (not a claim): on b3 and hg1 the two arms had opposite signs. The family wiring may set the
  sign of the steering response.

F3b-3 asks, on **fresh data**:
1. Does the true wiring (`ensemble`) replicate a direction-selective steering response?
2. Is that response larger in magnitude than under **random family wirings** of the same rate streams?

## 2. Brain stage: fresh runs (no F3b-1 data used for any decision)
- **Protocol.** As prereg/09 sections 3-4: the same four stimuli (`rot_left`, `rot_right`, `static`, `grey`; t0
  conditions 2, 1, 3, 10), 100 grey + 200 stimulus intervals of 5 ms, the same render chain and input check, and the
  Shiu reference arm on FlyWire v783 with inputs sugar (0 Hz), `vbnd_L` and `vbnd_R`.
- **Seeds.** New brain seeds **10-29** (20 seeds). Per (stimulus, seed) the engine state is fresh.
- **Recorded.** Per-interval spike counts of **all FlyWire DNs** (`super_class == descending`), to
  `runs/f3b3/brain/<stimulus>_s<seed>.npz`.
- **Render.** The stimulus is rendered again (it is deterministic). Whether it equals the F3b-1 render bit for bit is
  recorded (descriptive). The input check of prereg/09 section 3 must pass; otherwise the work stops.
- **No reuse.** The F3b-1 brain records (seeds 0-9) and the F3b-2 VNC results are not used for any decision here.
- Code: `scripts/f3b3_brain.py`, which reuses the render, input-check and engine code of `scripts/f3b1_brain.py`.

## 3. The ensemble and the bridge (unchanged)
- **Ensemble.** The 112 families of prereg/10 section 3 (`f3b2_common.ENSEMBLE`). DNg02 stays the `HAND` baseline at
  100 Hz on both sides. Every other BANC DN is silent.
- **Bridge rules** (`coupling/dn_bridge.py`), as prereg/09 section 5 and prereg/10 section 4:
  - matching by exact `cell_type` first, then family; side to side;
  - per-neuron mean rate over a trailing window W = 5 intervals, ending at interval k - L, with L = 1;
  - an independent Poisson process per BANC neuron;
  - **gain 1**.
- **Mapping on the real tables** (data only, computed before this file was committed):
  - `ensemble`: 225 units, 305 BANC targets.
  - Each `random_k`: 224 units (matched by source family), the same 305 BANC targets.

## 4. VNC stage
- **Model.** The prereg/08 subgraph (25,818 neurons), reference arm (`w_syn` 0.275 mV, scale 1.0). Input groups
  `DNg02_L`, `DNg02_R` (100 Hz throughout) and one per-neuron group for the 305 bridge targets, with rates set at every
  5 ms boundary. Each run lasts the full 1.5 s.
- **VNC seed** = 1000 × stimulus code + brain seed (codes: `rot_left` 1, `rot_right` 2, `static` 3, `grey` 4). The noise
  is common across arms, and these seeds are disjoint from the gate seeds.
- Every arm replays the **same** fresh brain records: 4 stimuli × 20 seeds per arm.

## 5. Arms
| arm | ensemble BANC DNs receive | gain |
|---|---|---|
| `ensemble` | their own family's FlyWire rate (the true wiring) | 1 |
| `bridge_off` (ablation) | 0 Hz (only the `HAND` DNg02 baseline) | - |
| `random_k`, k = 1..39 | the FlyWire rate of another ensemble family, under the k-th random derangement | 1 |

- **Generator.** For k = 1..39: `numpy.random.default_rng(seed_k).permutation(112)` over the list order of
  prereg/10 section 3. It is redrawn from the same generator until it has no fixed point, which gives a uniform random
  derangement. Mapping: BANC target family i <- FlyWire source family p[i]. Side is kept.
- **Permutation seeds.** seed_k = 20261100 + k, i.e. random_1 20261101, random_2 20261102, random_3 20261103, random_4
  20261104, random_5 20261105, random_6 20261106, random_7 20261107, random_8 20261108, random_9 20261109, random_10
  20261110, random_11 20261111, random_12 20261112, random_13 20261113, random_14 20261114, random_15 20261115,
  random_16 20261116, random_17 20261117, random_18 20261118, random_19 20261119, random_20 20261120, random_21
  20261121, random_22 20261122, random_23 20261123, random_24 20261124, random_25 20261125, random_26 20261126,
  random_27 20261127, random_28 20261128, random_29 20261129, random_30 20261130, random_31 20261131, random_32
  20261132, random_33 20261133, random_34 20261134, random_35 20261135, random_36 20261136, random_37 20261137,
  random_38 20261138, random_39 20261139.
- **Fixed now.** The 39 maps are fixed by the seeds and the generator (`scripts/f3b3_common.py` `derangement`,
  `permutations`). The SHA-256 of their canonical JSON (`permutations_json()`) is
  `cf26524b6c1f32d6ee57b9d7ef13858e80d905c8ceab5daee30f99e8fac84f1f`. A unit test checks this hash.
- **Checked before commit.** All 39 are derangements and pairwise distinct. None equals the F3b-2 permutation; on
  average 1.15 of the 112 pairs coincide with it.
- **What a random wiring keeps.** The same set of FlyWire rate streams and the same BANC targets. Only which family
  receives which stream changes.

## 6. Primary statistic
- **MN types.** b3, i1, i2, hg1, hg4 (prereg/09 section 8).
- Per arm, MN type m and seed s: D_m(s) = (R - L)(`rot_right`, s) - (R - L)(`rot_left`, s). Here R and L are the
  per-neuron BANC rates of type m on the right and left side in the 1 s stimulus window (intervals 100-299).
- **T_m(arm)** = the mean of D_m over the 20 seeds.

## 7. Decision rules (fixed now)
**(a) Replication.** In `ensemble`, at least one of the 5 MN types has D significant.
- Test: two-sided one-sample t-test of D_m against 0 over the 20 seeds, paired by seed, with Holm over the 5 types,
  alpha 0.05.
- If D is identically 0, p = 1. If D has zero variance and a non-zero mean, p = 0.

**(b) Specificity (magnitude).** For at least one MN type m that is significant in (a), |T_m(`ensemble`)| is larger
than |T_m(`random_k`)| in at least 38 of the 39 random wirings.
- Permutation p: p_m = (1 + #{k : |T_m(random_k)| >= |T_m(ensemble)|}) / 40. Criterion p_m <= 0.05. Ties count
  against `ensemble`.
- As the owner specified, there is no correction across the types tested in (b). p_m is reported for all 5 types,
  including those not significant in (a); only types significant in (a) count toward the decision.

**(c) Sign specificity (secondary; reported, not used for the decision).** For each of the 5 types: the fraction of the
39 random wirings whose T_m has the same sign as T_m(`ensemble`). T_m = 0 counts as a different sign.

**bridge_off must stay silent.** `bridge_off` must not be direction-selective under the test of (a) (Holm over the 5
types). Its MN rates are reported.

**Outcome.**
- **"Specific"** only if (a) and (b) hold, `bridge_off` is silent and the gates pass.
- **"Replicated, not specific"** if (a) holds but (b) does not. Also if (a) and (b) hold but `bridge_off` is
  direction-selective; in that case the reason is stated.
- **"Not replicated"** if (a) does not hold.

No threshold, window, seed, test, gain, family list, generator, permutation seed or arm is changed afterwards.

## 8. Reported (descriptive, not used for the decision)
1. **Null distribution.** Per primary type: the 39 values of T_m(random_k), sorted; their quantiles; the position of
   T_m(`ensemble`); and p_m.
2. **Secondary set.** The same D, T and permutation p for the owner's prereg/08 steering MN list
   (`f3b2_common.STEER_MN`), restricted to types that pass the symmetry gate on the subgraph. Holm over those types in
   `ensemble`. Labelled secondary.
3. **Brain side.** On the fresh seeds, D_brain per ensemble family (FlyWire per-neuron rate per side), with a t-test
   and Holm over the 112 families. The direction-selective families and their signs are listed.
4. **Applied DN rates.** Per family and side in `ensemble` (mean over seeds, per stimulus), and their maximum.
5. Network and non-driven rates per arm and stimulus.

## 9. Exploratory: expected sign of syndirectional steering (labelled exploratory; not used for any decision)
- **Assumptions.**
  - Under a rotating grating the fly turns with the motion (syndirectional optomotor response): `rot_right` (grating
    moves to the fly's right; prereg/09 section 3) → right turn, right side = inner side of the turn; `rot_left` the
    reverse.
  - BANC and FlyWire `side` are the fly's own side.
  - MN spike rate is a proxy for muscle activation. This proxy fails for a muscle controlled by firing phase
    (e.g. b1).
- **Expected sign.** A muscle more active on the inner side of a turn predicts D > 0; one more active on the outer
  side predicts D < 0.
- **Source.** Table 1 of Cheong et al., "Organization of circuits linking descending input to motor output in the
  *Drosophila* male adult nerve cord", eLife (doi 10.7554/eLife.96084). The table compiles Egelhaaf 1989; Heide 1975;
  Heide & Götz 1996 (J Exp Biol 199:1711); Lehmann & Götz 1996; Lindsay, Sustar & Dickinson 2017 (Curr Biol 27:345);
  Nachtigall & Wilson 1967.
  It lists activity on the "ipsilateral (inner side of turn)" vs "contralateral (outer side of turn)".

| MN type | Table 1 (inner / outer side of turn) | expected sign of D |
|---|---|---|
| b3 | activity increased / decreased | + |
| i1 | active / inactive | + |
| i2 | active, minor change / active, minor change | not fixed by the literature |
| hg1 | active / inactive | + |
| hg4 | activity increased / decreased | + |
| b1 | phase delay / phase advance (a timing effect, not a rate effect) | not fixed for a rate readout |
| b2 | inactive / active | - |
| iii1 | inactive / active | - |
| iii3 | activity decreased / increased | - |
| iii4 | minor or inactive / minor or inactive | not fixed |
| hg2 | minor or inactive / minor or inactive | not fixed |
| hg3 | active / inactive | + |
| tp1, tp2, tpn, ps1, ps2 | not in Table 1 (indirect control muscles) | not fixed |

- **Comparison.** The observed sign of T_m(`ensemble`) is compared with this column for every type with a fixed
  expected sign that passes the symmetry gate.
  - prereg/08 found that b1 and iii3 fail the gate on the subgraph; if they still fail, they are not compared.
  - Whether T_m is significant is shown next to each sign.
- **Limits.**
  - Most of these sources do not test a rotating grating in an open-loop flying *Drosophila* with these exact stimuli.
    The table is a compilation across dipterans and methods.
  - This comparison is a plausibility check only.

## 10. Gates (VNC side, as prereg/10 section 10)
1. **No input.** No drive, seeds 10-19: 0 spikes (25 ms steps, 1 s + 200 ms).
2. **No ignition after the cut.** DNg02 (both sides) and all 305 ensemble BANC neurons at 100 Hz for 1 s, then a 200 ms
   cut.
   - Seeds 250-299 (50; fresh).
   - Ignition means a network rate > 1 Hz in the last 25 ms cut step. The gate passes at 0 / 50; the Clopper-Pearson
     95 % CI is reported.
   - All arms drive the same input groups, so one gate-2 run covers every arm.
3. **Runaway bound (gate 3').** The network rate (stimulus window, mean over seeds) must be < 5 Hz for every arm ×
   stimulus (41 × 4) and for the gate-2 condition. The maximum is reported.

If gate 1, 2 or 3' fails: stop and report. The tests are then reported but not used.

## 11. Controls (rule 7)
- `bridge_off` is the ablation.
- The 39 `random_k` wirings are activity-matched shuffles: the same rate streams on the same targets. Together they
  form the null distribution of the magnitude test.
- As in prereg/06-10, a degree-matched random VNC graph is not part of this open-loop step. This is stated in the
  report.

## 12. Run-time estimate (before the run)
- **Brain stage:** 80 FlyWire runs at about 9 s each, plus about 3 min for setup and render: about 15 min.
- **VNC stage:** 41 arms × 80 replays at about 1.4 s each (F3b-2: 10 replays per 14 s), plus gates and setup: about
  85 min.
- **Total:** about 1.7 h, below the owner's 5 h threshold.

## 13. Origin classes (rule 2)
- **No new model constant.** Gain 1 is the identity; W and L are registered (`f3b1.*`, E).
- **Protocol settings of this document (E):**
  - brain seeds 10-29;
  - the VNC seed rule (1000 × code + seed);
  - gate seeds 10-19 and 250-299;
  - the derangement generator and the 39 permutation seeds;
  - the permutation alpha 0.05.
- **Unchanged:** every engine, vision and transduction constant.
