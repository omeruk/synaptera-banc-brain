# 08. Closed-loop optomotor yaw (T2): design and data-only prediction

Status: **design only** (2026-10-14, for the project owner's review). No code, no simulation and no pre-registration
exist for this task yet. Sections 4 and 8 contain numbers; those are produced by `scripts/f4_design_numbers.py` from
the F3b-3 records (`runs/f3b3/f3b3_v3.json`, not committed) and checked by `scripts/verify_report.py`. They are inside
`GEN` blocks and are regenerated, not edited. Section 8 is a **design analysis, not a test**: nothing in it enters a
pre-registered criterion.

Why this task. F3b-3 (`docs/results/f3b3_specificity.md`) showed that direction information passes through the hybrid
(brain DN activity -> BANC VNC subgraph -> steering MN asymmetry) reliably, and that the size and sign of the MN
asymmetry do not depend on which DN families carry it. T2 asks the next question: when the MN asymmetry is turned into
wing kinematics and fed back to the eyes, does the fly turn with the drum, against it, or not at all?

## 1. Task

- The fly hovers in place. Only body yaw is free; position, roll and pitch are held.
- Wing power: `HAND` DNg02 baseline at 100 Hz on both sides (as in F3b). The existing haltere reflex
  (`motor/legacy_reflexes.py`, yaw rate damping) stays as a labelled `REFLEX` scaffold.
- Stimulus: a grating drum around the fly rotates at a constant speed to the left or to the right. Controls: a static
  drum and a grey field.
- Question: does body yaw velocity follow the drum (syndirectional), oppose it, or stay unrelated?

## 2. Stimulus: virtual-reality drum, no 3D rendering

- The world pattern is the body-frame grating of `tasks/t0_stimuli.py` (wavelength and temporal frequency as in the
  open-loop screen, section 4 constants). With world azimuth a_w = a + psi (psi = body yaw, positive to the left,
  azimuth positive to the left) and drum angle phi_d(t) (positive to the left):

  `luminance(a, t) = grey + 0.5 * contrast * sin( 2 pi (a + psi(t) - phi_d(t)) / wavelength )`

  So the image depends only on the **relative angle** rho = phi_d - psi. In open loop (psi = 0, phi_d = -TF * wavelength * t)
  this is exactly condition 1 (`rot_right`, grating moves to the fly's right); `rot_left` is condition 2.
- The raw 512 x 450 eye images are filled from this formula with the camera rays of `tasks/t0_stimuli.camera_rays`
  (ray directions are constant for a fixed body, so they are computed once), then the existing path is used unchanged:
  Retina hex mapping -> FlyVis -> boundary layer (`sense/vision_boundary.py`, 5 ms transduction). No MuJoCo rendering.
- Sequence per flight: grey lead-in, then the drum rotates at constant speed. The analysis window excludes the first
  part of the rotation (brain onset transient).

## 3. Loop, causal, 5 ms

```
eyes(k) -> FlyVis -> FlyWire brain, interval k -> DN spike counts at boundary k+1
        -> DN bridge (ensemble, 112 families; window W intervals, latency 1, Poisson re-synthesis, gain 1)
        -> BANC VNC subgraph, interval k+1 -> steering-MN spike counts at boundary k+2
        -> MN rate estimate -> wing amplitudes A_L, A_R (command queue, applied one interval later)
        -> yaw plant -> psi(t) -> eyes
```

- Each hand-over is at an interval boundary (CLAUDE.md rule 4); the latency of every stage is written into the
  configuration and the record. Nothing is driven by spikes produced in the same interval.
- The bridge is the F3b-2 / F3b-3 bridge unchanged. DNg02 is `HAND` at 100 Hz; every other BANC DN is silent unless
  driven by the bridge.
- Plant (proposal, question Q6): a one-degree-of-freedom yaw model made of the existing stroke-averaged wing model
  (`body/quasi_steady.py`, yaw term of `aero_wrench`) and the `HaltereReflex` yaw damping. It is the same physics as the
  v1 hover flights, with the other degrees of freedom held. This keeps the cost of a flight at the engines.
- Source labels recorded separately every interval: `turn_vnc` (VNC), `turn_reflex` (REFLEX), DNg02 baseline (HAND),
  visual input (FLYVIS). The MN -> amplitude mapping and the gain G are `E`.

## 4. MN -> wing mapping

### 4.1 What the literature fixes

> **Superseded (2026-10-14):** the signs below are the v1 design. Section 10 replaces them with the signs stated in
> Lindsay et al. 2017 (read in full); this section is kept as the record of the first design.

Table 1 of Cheong et al. 2024 (eLife, doi 10.7554/eLife.96084) lists, per steering muscle, activity on the inner
(ipsilateral) and outer (contralateral) side of a turn. It has **no column for the kinematic effect**. A statement that
relates a muscle's activity to stroke amplitude exists for only some muscles, and the primary sources are not all open
access. What was read and where it was read:

| MN type | Table 1: inner / outer side of a turn | statement on stroke amplitude found | s_m used | basis |
|---|---|---|---|---|
| b2 | inactive / active (phasic) | phasic bursts correlate with rapid changes of wing-beat amplitude in body saccades (Heide & Goetz 1996, as summarised by Ehweiner et al. 2024); activity increases forward stroke angle and downstroke deviation (Whitehead et al. 2022); b1 and b2 pull the basalar apophysis forward (Nachtigall & Wilson 1967, as cited by Cheong et al. 2024) | + | direct |
| b3 | activity increased / decreased | activity increases when the ipsilateral wing decreases its amplitude, antagonist of b1 (Lindsay et al. 2017, as summarised by Ehweiner et al. 2024) | - | direct |
| i1 | active / inactive | responds to optomotor stimulation; specific role unknown (Ehweiner et al. 2024, citing Heide & Goetz 1996) | - | lateralisation |
| iii1 | inactive / active | reported not to take part in optomotor responses (Ehweiner et al. 2024, citing Heide & Goetz 1996) | + | lateralisation |
| hg1, hg3 | active / inactive | none found | - | lateralisation |
| hg4 | activity increased / decreased (tonic) | none found | - | lateralisation |

- **Direct** = a published relation between the muscle's activity and the amplitude of the ipsilateral wing.
- **Lateralisation** = the sign is inferred, not stated: a muscle more active on the inner side of a turn is taken to
  lower the ipsilateral amplitude, one more active on the outer side to raise it. This uses the convention of
  `body/quasi_steady.turn_to_amplitudes` (turn right -> larger left amplitude; origin `E`, ported from v1) and is
  consistent with the b3 statement. It is an inference from a correlation and is the weakest link of the design.
- **Excluded.** b1 and iii3 fail the symmetry gate on the subgraph (F3a-3 section 5.1). b1 acts through firing phase,
  not rate. i2, iii4 and hg2 are minor or inactive on both sides, so no sign is fixed. tp1, tp2, tpn, ps1, ps2 are
  indirect control muscles (Dickinson & Tu 1997, as cited by Cheong et al. 2024) and not in Table 1.
- **Conflict in the sources.** Ehweiner et al. 2024 write that the left i1 is active during right turns, which is the
  outer side; Table 1 gives i1 active on the inner side. The mapping set `lateral_i1_flipped` below tests this.
- **Relative effect sizes per muscle.** Lindsay et al. 2017 fitted wing kinematics to muscle activity, but the paper is
  not open access and its coefficients were not read. **A literature-weighted prediction is therefore not computed.**
  The only weights available are the equal per-Hz weights (s_m = +-1) and a data-defined variant (section 8).
- The mapping uses the MN rate as a proxy for muscle activation (same limit as `docs/results/f3b3_specificity.md`
  section 7).

### 4.2 Constants and gain

<!-- GEN:constants -->
| constant | value | source |
|---|---|---|
| amplitude asymmetry per unit turn_bias (each wing) | 5 deg | `configs/body_flight.yaml` `turn_dphi_deg` (E) |
| steady yaw rate per unit turn_bias | 5 rad/s | `yaw_rate_per_turn` (E) |
| yaw damping time constant | 20 ms | `yaw_damping_tau` (E) |
| turn_bias limit | 2.5 | `configs/readouts.yaml` `turn_bias_max` (E) |
| v1 range of A_L - A_R | 25 deg | 2 x turn_dphi_deg x turn_bias_max |
| yaw rate per degree of A_L - A_R | 28.6 deg/s | yaw_rate_per_turn / (2 turn_dphi_deg) |
| grating wavelength / temporal frequency | 30 deg / 2 Hz | `tasks/t0_stimuli.py` |
| drum speed in the open-loop data (F3b) | 60 deg/s | wavelength x frequency |
<!-- /GEN:constants -->

The mapping is `A_side = A_hover + G * sum_m s_m * r_(m, side)` with the MN rate r in Hz (per-neuron mean), so
`A_L - A_R = G * sum_m s_m (r_L - r_R)`. One global gain G (deg per Hz) for all types, fixed by an open-loop rule before
any closed-loop run and never tuned on behaviour. Two candidate rules, both from existing records:

<!-- GEN:gain -->
| candidate | rule | G (deg per Hz) | predicted A_L - A_R for `ensemble` (deg) | predicted steady yaw rate (deg/s) | ratio to the drum speed |
|---|---|---|---|---|---|
| G1 | largest F3a-3 asymmetry (DNa04 at 100 Hz, left drive, 63.3 Hz summed with the signs) = v1 range 25 deg | 0.395 | -0.39 | -11.3 | -0.19 |
| G2 | F3a-3 DNa04 at 25 Hz (7.6 Hz) = typical v1 hover turn (median peak turn_dn 0.91 -> 9.1 deg) | 1.197 | -1.20 | -34.3 | -0.57 |
<!-- /GEN:gain -->

- G1 uses the largest MN asymmetry in the F3a-3 dose-response (unilateral steering DN drive) and sets it to the full v1
  range, so G1 is the most conservative rule (smallest G). G2 matches the primary steering family at its lowest
  steering rate (25 Hz, inside its FlyWire flight range) to a typical v1 hover turn.
- Rejected: a G that makes the F3b-3 optomotor asymmetry reproduce the drum speed. It fits G to the very response under
  test and makes a gain near 1 true by construction.
- The predicted steady yaw rates above are open-loop (no visual feedback). The v1 constants make 1 deg of
  `A_L - A_R` worth about the yaw rate in the constants table; the drum speed is the one of the F3b screen.

## 5. Arms

| arm | MN -> wing signs | brain | VNC | notes |
|---|---|---|---|---|
| `ensemble` | literature signs | FlyWire, 112-family bridge | BANC subgraph | the hybrid |
| `bridge_off` | literature signs | FlyWire | DNg02 baseline only | DN bridge cut: no direction information reaches the VNC; control for everything downstream |
| `sign_flip` | all signs inverted | as `ensemble` | as `ensemble` | plant / sign check; turns negative into positive visual feedback (section 9) |
| `random_1` ... `random_5` | literature signs | as `ensemble` | F3b-3 derangements 1-5 (same SHA-256 list) | wiring control; chosen by index, not by outcome |
| `v1_hand` | v1 hand-made bridge and turn gain | FlyWire | none | reference; `legacy_bridge.Bridge`, DNp15 drive which fails the symmetry gate (R/L 1.66x), so run as a labelled `--waive DNp15` exception (Q4) |

Every arm records the HAND/REFLEX terms separately, and `zero_turn` (a constant zero yaw command) is available as the
brainless plant check, as decided after F0.

## 6. Primary readout and proposed criterion (to be pre-registered after the owner's answers)

- Per flight: mean body yaw velocity psi_dot over the analysis window. Per seed and arm: syndirectional gain

  `g = (psi_dot(rot_right) - psi_dot(rot_left)) / (2 * drum speed)`, and the offset `b = (psi_dot(rot_right) + psi_dot(rot_left)) / (2 * drum speed)`.

  g = 1 would cancel the retinal slip, g > 0 is syndirectional, g < 0 anti-directional. The static and grey flights give
  the yaw drift without any drum rotation.
- Proposed criteria (the thresholds are open questions, Q3 and Q8):
  1. **Response.** `ensemble`: g differs from 0 (two-sided one-sample t-test over seeds, Holm over the arms that carry
     the claim); the sign is reported with it. This is the replication-style test.
  2. **Brain dependence.** g(`ensemble`) against g(`bridge_off`) (paired by seed); `bridge_off` must have g = 0 within
     noise.
  3. **Wiring.** Rank of |g(`ensemble`)| among the five random wirings. With five random wirings the smallest possible
     permutation p is 1/6, so this is **descriptive**, not a test. F3b-3 already found no wiring specificity of the
     open-loop MN asymmetry, and this design does not try to re-test it.
  4. **Reference.** g(`v1_hand`) shows what the hand-made controller achieves in the same loop.
- Secondary: mean retinal slip speed (|rho_dot|) against the open-loop drum speed; runaway events (clipped amplitude,
  turn_bias at its limit); the yaw drift in the static and grey flights.
- Rotation to the left and to the right, at least five seeds (CLAUDE.md rule 7); seed changes brain noise, VNC noise
  and the Poisson re-synthesis. Brain seeds distinct from F3b-3 (10-29).

## 7. Compute, RAM, risks

<!-- GEN:timing -->
| stage | measured in F3b-3 | per 5 ms interval |
|---|---|---|
| FlyVis + boundary transduction | 7 s per 300 intervals | 23.3 ms |
| FlyWire brain | 9 s per 300 intervals | 30.0 ms |
| BANC VNC subgraph | 28 s per 20 replays of 300 intervals | 4.7 ms |
| total | | 58.0 ms |
| one-off engine build | brain 134 s, VNC 27 s | |
<!-- /GEN:timing -->

<!-- GEN:compute -->
| seeds per cell | flights (9 arms x 4 stimuli) | wall time (h, serial) | over the 5 h limit |
|---|---|---|---|
| 5 | 180 | 1.4 | no |
| 10 | 360 | 2.9 | no |
| 20 | 720 | 5.7 | yes |

One flight = 0.5 s grey + 2 s drum = 500 intervals = 29 s of wall time with all three engines. RAM: FlyVis + brain engine about 3.6 GB (free memory before / after the build in the F3b-3 log), VNC engine peak 2.8 GB (F3b-3 record), so one flight needs about 6.4 GB and flights run serially.
<!-- /GEN:compute -->

- The estimate uses the per-interval costs measured in the F3b-3 logs and does not include the engine builds
  (one-off per process) or the yaw plant (negligible). The brain cannot be shared between arms, because the visual
  input depends on the arm's own yaw.
- Risks:
  1. **Noise.** Single steering MNs fire a few spikes per 25 ms; the F3b-3 25 ms series has a signal-to-noise ratio far
     below 1 (section 8). The muscle low-pass and the rate estimator are new constants (Q7).
  2. **Constant bias.** The static grating already produces an MN asymmetry (section 8); a constant turn bias rotates
     the fly and changes the visual input even without a drum.
  3. **Sign.** If the loop sign is positive visual feedback (anti-directional response, `sign_flip`) the yaw runs
     away to the limits; the readout is then dominated by saturation and needs the runaway record.
  4. **Mapping evidence.** Only two of the seven mapped types have a direct statement (section 4.1).
  5. **Gate.** All seven mapped types pass the symmetry gate, but each is a single neuron per side; the left/right
     proofreading asymmetry of BANC (docs/05, decision after prereg/05) remains a property of the data.
  6. **Pipeline latency.** Several intervals plus the window average and the yaw damping time constant; the loop
     is untested for stability.

## 8. Open-loop prediction (data only; design analysis, not a test)

Source: the 3,280 F3b-3 replays (20 seeds x 4 stimuli x 41 arms). Steps: for every arm, stimulus and seed,
`A_L - A_R = G * sum_m s_m (r_L - r_R)` with G = 1 (so the unit is Hz; G only scales); the syndirectional index
`S = (dA(rot_right) - dA(rot_left)) / 2` per seed; S > 0 is syndirectional. `bridge_off` has S = 0 by construction
(identically silent steering MNs), and `sign_flip` has S(`ensemble`) with the sign inverted. The seed index is the same
across arms, so seeds are paired.

### 8.1 `ensemble`, primary mapping set `lateral`

<!-- GEN:ensemble -->
S(`ensemble`) = -1.00 Hz (s.d. over seeds 3.39; positive in 6 of 20 seeds, negative in 14; one-sample t-test p = 0.203, descriptive). S(`sign_flip`) = +1.00 Hz by construction; S(`bridge_off`) = -0.03 Hz.
<!-- /GEN:ensemble -->

<!-- GEN:per_stimulus -->
| stimulus | A_L - A_R (Hz units, G = 1), mean +- s.d. over seeds | bridge_off |
|---|---|---|
| rot_left | -11.00 +- 4.08 | +0.05 +- 0.22 |
| rot_right | -13.00 +- 5.05 | +0.00 +- 0.00 |
| static | -5.20 +- 3.58 | +0.05 +- 0.22 |
| grey | +0.00 +- 0.00 | +0.00 +- 0.00 |
<!-- /GEN:per_stimulus -->

<!-- GEN:bias -->
Under the static grating `ensemble` has A_L - A_R = -5.20 Hz units (a constant turn bias before any drum rotation); random wirings: median +2.70, 5-95 % -9.33 to +27.42.
<!-- /GEN:bias -->

### 8.2 What each MN type contributes

S = sum over types of the contribution `-s_m * D_m / 2`, with D_m = (R - L)(rot_right) - (R - L)(rot_left) (T = mean D, as
in F3b-3). The last column divides T by the standard deviation of T over the 39 random wirings, which puts the types on a
common scale.

<!-- GEN:contrib -->
| MN type | s_m | basis | T (Hz) | contribution to S (Hz) | seeds syn / anti | T relative to the null s.d. (z) |
|---|---|---|---|---|---|---|
| b2 | +1 | direct | -1.55 | +0.78 | 14 / 5 | -0.83 |
| b3 | -1 | direct | +2.80 | +1.40 | 18 / 1 | +0.64 |
| i1 | -1 | lateralisation | +2.75 | +1.38 | 18 / 1 | +0.98 |
| iii1 | +1 | lateralisation | +0.00 | +0.00 | 0 / 0 | +0.00 |
| hg1 | -1 | lateralisation | +2.75 | +1.38 | 16 / 2 | +0.37 |
| hg3 | -1 | lateralisation | -0.20 | -0.10 | 1 / 4 | -0.58 |
| hg4 | -1 | lateralisation | -11.65 | -5.83 | 0 / 20 | -1.83 |
<!-- /GEN:contrib -->

<!-- GEN:hg4 -->
Ratio of the absolute T of hg4 to that of b3, i1 and hg1: 4.2, 4.2, 4.2 (F3b-3 values).
<!-- /GEN:hg4 -->

hg4 carries the largest response and, under the lateralisation sign, pulls S towards anti-directional. Like hg1 and
hg3, it has no stated kinematic effect in the sources read, so its sign rests on the lateralisation inference alone.

### 8.3 Dependence on the mapping set, and the random wirings

<!-- GEN:variants -->
| mapping set | types (sign) | S(`ensemble`) Hz | offset B (Hz) | seeds syn / anti | seeds for 80 % power (alpha 0.05) | random wirings syn / anti (of 39) | random significantly syn / anti (p < 0.05) | random wirings below `ensemble` |
|---|---|---|---|---|---|---|---|---|
| lateral | b2(+), b3(-), i1(-), iii1(+), hg1(-), hg3(-), hg4(-) | -1.00 | -12.00 | 6 / 14 | 91 | 17 / 21 | 12 / 15 | 18 |
| direct | b2(+), b3(-) | +2.17 | +7.53 | 17 / 3 | 7 | 21 / 18 | 14 / 7 | 32 |
| lateral_i1_flipped | b2(+), b3(-), i1(+), iii1(+), hg1(-), hg3(-), hg4(-) | -3.75 | -13.75 | 2 / 17 | 8 | 17 / 21 | 12 / 15 | 6 |
| lateral_no_hg | b2(+), b3(-), i1(-), iii1(+) | +3.55 | +8.40 | 17 / 2 | 5 | 25 / 14 | 16 / 7 | 33 |
| lateral_no_hg4 | b2(+), b3(-), i1(-), iii1(+), hg1(-), hg3(-) | +4.83 | -0.88 | 19 / 1 | 4 | 24 / 15 | 17 / 8 | 34 |
| lateral, null-z weights (data-defined, not literature) | as lateral | +0.20 (z units) | - | - | - | 20 / 19 | - | 23 |
<!-- /GEN:variants -->

<!-- GEN:random -->
Over the 39 random wirings (set `lateral`): S has quantiles min -19.62, 5 % -10.99, 25 % -2.66, median -0.23, 75 % +1.84, 95 % +9.25, max +23.52 Hz. 17 of 39 are syndirectional (12 significantly, p < 0.05, descriptive), 21 anti-directional (15 significantly) and 1 exactly zero. 18 of 39 random wirings have S below `ensemble`.
<!-- /GEN:random -->

- `direct` keeps only b2 and b3; `lateral_no_hg` and `lateral_no_hg4` drop the hg types, which have no stated kinematic effect.
  `lateral_i1_flipped` follows the conflicting i1 statement. The null-z variant weights each type by the inverse of its
  spread over the random wirings; it is data-defined, not literature.
- "Seeds for 80 % power" is the number of paired seeds a one-sample test of S at alpha 0.05 would need if the closed
  loop reproduced the open-loop S and its spread (a rough guide, because the closed loop adds feedback and a
  new noise path).

### 8.4 Time resolution (25 ms bins, `ensemble`, types b3, i1, hg1, hg4)

<!-- GEN:snr -->
| averaging bin | mean S (Hz) | s.d. over seeds x bins (Hz) | mean / s.d. | bins with S > 0 | bins with S < 0 |
|---|---|---|---|---|---|
| 25 ms | -1.13 | 23.80 | -0.05 | 0.18 | 0.25 |
| 100 ms | -1.13 | 10.77 | -0.11 | 0.24 | 0.30 |
<!-- /GEN:snr -->

### 8.5 Conclusion

**The closed-loop test is expected to show nothing reliable: under the primary (lateralisation) mapping the predicted
turn is weakly anti-directional and not distinguishable from zero, its sign flips with the mapping of the single type
hg4, the random wirings are as likely to predict either direction, and a constant yaw bias is larger than the
direction-dependent term.** The numbers behind each clause are in sections 8.1 and 8.3. This is a design analysis made
after seeing the F3b-3 results and is not a registered expectation.

## 9. Questions the owner must answer before implementation

1. **Q1. Primary mapping set.** Recommendation: `lateral` as the primary set, because it is the only rule that does not
   pick types by outcome (every gate-passing type with a sign in Table 1), with `direct`, `lateral_i1_flipped` and
   `lateral_no_hg4` reported as sensitivity sets. The alternative (`direct`) rests on two types.
2. **Q2. Gain.** G1 (conservative) or G2? Recommendation: G1, because it is bounded by the v1 range and by an observed
   F3a-3 response, and only scales the loop.
3. **Q3. Seeds.** The table in section 7 gives cost; section 8.3 gives the power guide. Recommendation: at least 10 seeds
   per cell, more for `ensemble` and `bridge_off` if the budget allows.
4. **Q4. Reference arm.** v1's own controller (DNp15, symmetry gate waived, labelled exception) or a hand-made readout on
   a gate-passing DN family (new, would need its own justification)?
5. **Q5. Drum speed.** Keep the F3b screen speed (the brain responses are known there) or add a second speed?
6. **Q6. Plant.** One-degree-of-freedom analytic yaw (recommended) or the MuJoCo `FlightBody` with the other degrees
   locked?
7. **Q7. New constants.** MN rate estimator window and muscle low-pass, and a limit for the amplitude and yaw rate.
   They are `E` constants without a literature value here. An open-loop rule would be, for example, the smallest
   window whose F3b-3 signal-to-noise ratio is at least 1; this is a choice made on the very response under test, so
   the owner may prefer a fixed value from the muscle literature.
8. **Q8. Effect-size criterion.** A minimum |g| for the response criterion (for example a multiple of the seed-to-seed
   spread of `bridge_off`), or only the sign and significance.
9. **Q9. Randoms.** Five random wirings by index (random_1 ... random_5), or more (needed for a permutation test at
   alpha 0.05: at least 19)?
10. **Q10. Stop rule.** If `sign_flip` or the offset makes the yaw run away, are runaway flights excluded, or analysed
    with the saturation recorded?

### 9.1 Answers (owner-approved 2026-10-14)

The F4-1 closed-loop run is postponed until the open-loop signal is syndirectional, significant and specific
(docs/05, decision after the T2 design). These answers fix the design for when it is run.

1. **Q1.** Primary mapping set `lateral`, redefined from Lindsay et al. 2017 in section 10 (v2). `direct` (b2, b3) is a
   registered secondary set. The iii1 / iii4 sets of section 10 are sensitivity sets.
2. **Q2.** G1, fixed at the v1 value (section 10.4); it only scales the loop.
3. **Q3.** At least 10 seeds per cell (section 7), more for `ensemble` and `bridge_off` if the budget allows.
4. **Q4.** Reference arm: v1's own controller (DNp15, symmetry gate waived, labelled exception), run unmodified
   (CLAUDE.md rule 8). A new hand-made readout on a gate-passing family would need its own justification and is not
   added.
5. **Q5.** One drum speed, the F3b screen speed, where the brain responses are known.
6. **Q6.** One-degree-of-freedom analytic yaw plant (section 3).
7. **Q7.** New `E` constants fixed without looking at the response under test: MN rate estimator = 25 ms boxcar (the
   bin width of the F3b-3 records, registered before them), no further muscle low-pass; amplitude asymmetry limited to
   the v1 range and yaw rate to the v1 value at `turn_bias_max`. All are registered in `configs/assumptions.json` with
   origin `E` when the run is implemented.
8. **Q8.** The criterion is sign, significance and specificity against the random wirings; |g| is reported
   descriptively, with no minimum effect size.
9. **Q9.** At least 19 random wirings (the minimum for a permutation test at alpha 0.05); the compute table of section 7
   is re-estimated for this arm count at pre-registration.
10. **Q10.** No flight is excluded. Saturation of the amplitude or yaw-rate limit is recorded per interval and analysed.
11. **Yaw trim (owner decision).** In the closed loop the per-type (R - L) under grey of the same arm is subtracted,
    calibrated on separate seeds; origin `E`. It cancels in D and S and changes only the offset (section 10.5).

## 10. Addendum (2026-10-14): mapping update from Lindsay et al. 2017 (v2)

Source: Lindsay T, Sustar A, Dickinson MH (2017) The function and organization of the motor system controlling flight
maneuvers in flies. Curr. Biol. 27, 345-358, doi:10.1016/j.cub.2016.12.018. Read in full from the PDF (not committed).
The muscles were imaged on the right side; "ipsilateral" is the imaged side. Sections 4.1 and 8 (v1) are kept
unchanged as the record of the first design; this section supersedes their signs.

### 10.1 What the paper states

| muscle | statement | location | sign (effect on ipsilateral amplitude) |
|---|---|---|---|
| b2 | rises in saccades in which the ipsilateral amplitude increases; rises in yaw turns in which it increases; "+" row of the model; b1 and b2 extend the ventral stroke extreme | p. 349; p. 353; Fig. 6; p. 354-355 | + |
| b3 | falls in saccades with ipsilateral amplitude increase, rises in those with decrease; recruited in yaw turns with decrease; "-" row of the model; antagonist of b1 / b2 | p. 349; p. 353; Fig. 6; Fig. 7 tuning (p. 353) | - |
| i1 | rises in saccades with ipsilateral amplitude decrease; in yaw it rises for both directions (saccade-linked, slightly directional); "-" row of the model; i1 / i2 depress the stroke plane with a drop in amplitude | p. 349; p. 351, Fig. 5; Fig. 6; p. 355 | - |
| i2 | tonic; activity correlated with decreases in stroke amplitude (regression); "-" row of the model | p. 349 (Fig. 4); Fig. 6 | - |
| hg1 | rises in saccades with ipsilateral amplitude decrease; recruited in yaw turns with decrease; "-" row of the model | p. 349; p. 353; Fig. 6 | - |
| hg2, hg3 | all four hg muscles recruited in yaw turns with ipsilateral amplitude decrease; tuned to yaw toward the ipsilateral side; "-" row of the model | p. 353; Fig. 6; Fig. 7 | - |
| hg4 | tonic; rises in saccades with ipsilateral amplitude decrease; falls in yaw turns with increase, recruited with decrease; "-" row of the model | p. 349; p. 353; Fig. 6 | - |
| iii1 | rises in saccades with ipsilateral amplitude increase (p. 349); iii muscles tuned toward the contralateral side and proposed to increase stroke amplitude (p. 353, p. 355); but placed in the "-" row of Fig. 6 | p. 349, 353, 355 vs Fig. 6 | conflict |
| iii4 | tuned with iii1 and iii3 toward the contralateral side (p. 353), proposed to increase amplitude (p. 355); "-" row of Fig. 6 | p. 353, 355 vs Fig. 6 | conflict |
| b1, iii3 | + (Fig. 6, text) | | fail the symmetry gate; not used |

- **Changes from v1.** i1, hg1, hg3 and hg4 keep their signs, now stated directly instead of inferred from
  lateralisation. i2 and hg2 gain a sign (-). iii1 leaves the primary set (conflict). The i1 statement of Ehweiner et
  al. 2024 (left i1 active in right turns) is not supported by Lindsay; `lateral_i1_flipped` is dropped.
- **Relative effect sizes: not given.** Fig. 4 reports only the variance explained by muscle combinations, not
  per-muscle coefficients; Fig. 5 traces are in calcium z-score units; Fig. 6 is a schematic. The per-Hz weights stay
  equal (s_m = +-1).
- **Tonic and phasic muscles.** The slow optomotor response is carried mainly by the tonic muscles (b1, b3, i2, iii3,
  hg4; p. 351-352), the phasic ones (b2, i1, iii1, hg1-hg3) follow saccades. hg4, the type that dominates the v1
  prediction, is the tonic hg muscle. (The text lists hg4 once among the "phasic" largest muscles, p. 355, which
  contradicts its tonic classification on p. 347 and in Fig. 6; the classification is not used here.)
- **Rule fixed before the v2 numbers:** primary `lateral` = muscles with a sign stated directly by Lindsay that pass
  the gate (b2, b3, i1, i2, hg1, hg2, hg3, hg4). iii1 and iii4 are left out and reported with both signs as
  sensitivity sets. Numbers: `docs/results/f4_design_numbers_v2.json` (`scripts/f4_design_numbers.py --version v2`);
  v1 is not overwritten.

### 10.2 `ensemble`, primary mapping set `lateral` (v2)

<!-- GEN:v2_ensemble -->
S(`ensemble`) = -1.02 Hz (s.d. over seeds 3.38; positive in 5 of 20 seeds, negative in 14; one-sample t-test p = 0.191, descriptive). S(`sign_flip`) = +1.02 Hz by construction; S(`bridge_off`) = -0.03 Hz.
<!-- /GEN:v2_ensemble -->

<!-- GEN:v2_per_stimulus -->
| stimulus | A_L - A_R (Hz units, G = 1), mean +- s.d. over seeds | bridge_off |
|---|---|---|
| rot_left | -11.00 +- 4.08 | +0.05 +- 0.22 |
| rot_right | -13.05 +- 5.05 | +0.00 +- 0.00 |
| static | -5.25 +- 3.54 | +0.05 +- 0.22 |
| grey | +0.00 +- 0.00 | +0.00 +- 0.00 |
<!-- /GEN:v2_per_stimulus -->

### 10.3 Contributions, mapping sets, random wirings (v2)

<!-- GEN:v2_contrib -->
| MN type | s_m | basis | T (Hz) | contribution to S (Hz) | seeds syn / anti | T relative to the null s.d. (z) |
|---|---|---|---|---|---|---|
| b2 | +1 | direct | -1.55 | +0.78 | 14 / 5 | -0.83 |
| b3 | -1 | direct | +2.80 | +1.40 | 18 / 1 | +0.64 |
| i1 | -1 | Lindsay 2017 | +2.75 | +1.38 | 18 / 1 | +0.98 |
| i2 | -1 | Lindsay 2017 | -0.05 | -0.03 | 0 / 1 | -0.02 |
| hg1 | -1 | Lindsay 2017 | +2.75 | +1.38 | 16 / 2 | +0.37 |
| hg2 | -1 | Lindsay 2017 | +0.00 | +0.00 | 0 / 0 | - |
| hg3 | -1 | Lindsay 2017 | -0.20 | -0.10 | 1 / 4 | -0.58 |
| hg4 | -1 | Lindsay 2017 | -11.65 | -5.83 | 0 / 20 | -1.83 |
<!-- /GEN:v2_contrib -->

<!-- GEN:v2_hg4 -->
Ratio of the absolute T of hg4 to that of b3, i1 and hg1: 4.2, 4.2, 4.2 (F3b-3 values).
<!-- /GEN:v2_hg4 -->

<!-- GEN:v2_variants -->
| mapping set | types (sign) | S(`ensemble`) Hz | offset B (Hz) | seeds syn / anti | seeds for 80 % power (alpha 0.05) | random wirings syn / anti (of 39) | random significantly syn / anti (p < 0.05) | random wirings below `ensemble` |
|---|---|---|---|---|---|---|---|---|
| lateral | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-) | -1.02 | -12.03 | 5 / 14 | 86 | 19 / 19 | 13 / 13 | 16 |
| direct | b2(+), b3(-) | +2.17 | +7.53 | 17 / 3 | 7 | 21 / 18 | 14 / 7 | 32 |
| lateral+iii1(+) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii1(+) | -1.02 | -12.03 | 5 / 14 | 86 | 19 / 19 | 14 / 13 | 16 |
| lateral+iii1(-) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii1(-) | -1.02 | -12.03 | 5 / 14 | 86 | 19 / 19 | 13 / 13 | 16 |
| lateral+iii4(+) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii4(+) | +0.10 | -1.45 | 8 / 11 | 10712 | 20 / 19 | 12 / 11 | 19 |
| lateral+iii4(-) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii4(-) | -2.15 | -22.60 | 4 / 15 | 26 | 21 / 18 | 14 / 14 | 13 |
| lateral+iii1(+),iii4(+) (text) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii1(+), iii4(+) | +0.10 | -1.45 | 8 / 11 | 10712 | 20 / 19 | 12 / 11 | 19 |
| lateral+iii1(-),iii4(-) (Fig. 6) | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-), hg4(-), iii1(-), iii4(-) | -2.15 | -22.60 | 4 / 15 | 26 | 21 / 18 | 14 / 14 | 13 |
| lateral_no_hg4 | b2(+), b3(-), i1(-), i2(-), hg1(-), hg2(-), hg3(-) | +4.80 | -0.90 | 19 / 1 | 4 | 26 / 12 | 19 / 7 | 33 |
| lateral, null-z weights (data-defined, not literature) | as lateral | +0.20 (z units) | - | - | - | 19 / 20 | - | 23 |
<!-- /GEN:v2_variants -->

<!-- GEN:v2_random -->
Over the 39 random wirings (set `lateral`): S has quantiles min -15.45, 5 % -11.02, 25 % -2.16, median +0.00, 75 % +1.86, 95 % +9.23, max +23.52 Hz. 19 of 39 are syndirectional (13 significantly, p < 0.05, descriptive), 19 anti-directional (13 significantly) and 1 exactly zero. 16 of 39 random wirings have S below `ensemble`.
<!-- /GEN:v2_random -->

### 10.4 Gain (v2)

<!-- GEN:v2_gain -->
| gain | G (deg per Hz) | predicted A_L - A_R for `ensemble` (deg) | predicted steady yaw rate (deg/s) | ratio to the drum speed |
|---|---|---|---|---|
| G1, fixed (v1 value, owner decision) | 0.395 | -0.40 | -11.6 | -0.19 |
| G1 rule re-applied with the v2 signs (not used) | 0.395 | - | - | - |
<!-- /GEN:v2_gain -->

### 10.5 Offset and yaw trim (v2)

The trim here is calibrated leave-one-seed-out on the F3b-3 `ensemble` seeds, as a stand-in for the separate seeds of
the closed loop.

<!-- GEN:v2_bias -->
Under the static grating `ensemble` has A_L - A_R = -5.25 Hz units (a constant turn bias before any drum rotation); random wirings: median +2.60, 5-95 % -9.52 to +27.34.
<!-- /GEN:v2_bias -->

<!-- GEN:v2_trim -->
| mapping set | A_L - A_R under grey (Hz) | offset B untrimmed (Hz) | offset B after trim (Hz) | abs S / abs B after trim |
|---|---|---|---|---|
| lateral | +0.00 | -12.03 | -12.03 | 0.09 |
| direct | +0.00 | +7.53 | +7.53 | 0.29 |
| lateral+iii1(+) | +0.00 | -12.03 | -12.03 | 0.09 |
| lateral+iii1(-) | +0.00 | -12.03 | -12.03 | 0.09 |
| lateral+iii4(+) | +0.00 | -1.45 | -1.45 | 0.07 |
| lateral+iii4(-) | +0.00 | -22.60 | -22.60 | 0.10 |
| lateral+iii1(+),iii4(+) (text) | +0.00 | -1.45 | -1.45 | 0.07 |
| lateral+iii1(-),iii4(-) (Fig. 6) | +0.00 | -22.60 | -22.60 | 0.10 |
| lateral_no_hg4 | +0.00 | -0.90 | -0.90 | 5.33 |
<!-- /GEN:v2_trim -->

### 10.6 Conclusion (v2)

The Lindsay update leaves the prediction where it was: **nothing reliable.** All signs that carry weight in v1 are
confirmed, and the types that enter (i2, hg2) or leave (iii1) have T near or at zero in `ensemble`. So S of the primary
set stays weakly anti-directional and not distinguishable from zero. hg4 still sets the net sign. iii4 is a second
deciding type: its sign is a conflict inside the paper, and with the text's sign S comes to about zero, with the
Fig. 6 sign it becomes more anti-directional. The random wirings remain split between the two directions.

**The owner's yaw trim does not remove the offset.** In the F3b-3 records every mapped MN of `ensemble` is silent under
grey, so A_L - A_R under grey is zero and the trim subtracts nothing. The offset is evoked by the patterned stimulus
itself (it is already present under the static grating). A trim that removes it would have to be calibrated under a
patterned stimulus, for example the static grating; that is a design change for the owner, not made here.
