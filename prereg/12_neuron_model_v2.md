# Pre-registration 12 — neuron model v2 (open loop): adaptation, conductance synapses, electrical synapses

Status: committed **before any run of this experiment** (CLAUDE.md rule 6). It is not edited afterwards; additions go in
a new file and are labelled post-hoc. Decision context: the project owner's instruction after PR #17 (2026-10-14,
recorded in `docs/05_ARCHITECTURE_PLAN.md`). Design and parameter sources: `docs/09_NEURON_MODEL_V2.md`. Earlier
pre-registrations are not changed.

Only a wall-time benchmark ran before this file was committed (section 9). It used a non-registered condition (seed
9999) and read nothing but wall time.

## 1. Model and arms
- **Reference arm R.** The Shiu et al. 2023 LIF network, unchanged (`engine/lif_brian2.py`, `model_variant=None`;
  pinned `brain_model/model.py`).
- **Named variants** (`engine/model_variants.py`; constants in `configs/assumptions.json` under `model_v2.*`; electrical
  pairs in `configs/gap_junctions.yaml`; all fixed now):

| switch | what | constants (origin) |
|---|---|---|
| `adapt` | AdEx adaptation current: tau_w dw/dt = a (v - v_0) - w, w += b per spike; subtracted in the membrane equation | a / g_L 0.1333, b / g_L 2.683 mV, tau_w 144 ms (P, Brette & Gerstner 2005, Table 1) |
| `cond` | conductance synapses g_e (E_exc - v) + g_i (E_inh - v), Shiu tau and delay | E_exc 0 mV (P), E_inh -74 mV (M, Ryglewski et al. 2017), dg_exc 0.0052912003, dg_inh 0.0125153142 (F: one synapse at rest gives the Shiu peak PSP) |
| `gap` | rectifying electrical synapses GF -> TTMn and GF -> PSI only, pairs placed by >= 1 chemical synapse; an instantaneous postsynaptic jump of 68.75 mV per GF spike | GAP_DV (E: the model's own suprathreshold kick; measured property: 1:1 sub-ms transmission, Phelan et al. 2008) |

- **Arms.**
  - R: none.
  - A: adapt.
  - AC: adapt + cond.
  - ACG: adapt + cond + gap.

  The same arm is used for the FlyWire v783 brain and the BANC v3 VNC subgraph of prereg/08. In the hybrid both halves
  use the same arm.
- **Unchanged in every arm:** v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly, w_syn, the sign rule, and the Poisson input
  path.
- **Integration.** R and A use Brian2's exact `linear` method. AC and ACG use `exponential_euler` (their v equation is
  not linear). The clock is 0.1 ms throughout.
- **The `gap` pairs on the real tables** (data only, computed before this file was committed by
  `model_variants.gap_pairs`):
  - BANC VNC subgraph: left GF -> left TTMn, right GF -> right TTMn, each GF -> both PSIs (6 pairs).
  - FlyWire: none. The GF has no VNC part there, so ACG equals AC in the brain.
- **GF in the ensemble.** DNp01 (GF) belongs to the 112-family ensemble. So in test (iii) under ACG, the brain's GF rate
  reaches TTMn and PSI through the junctions.

## 2. Which arms run (owner's run-time rule)
- **The rule.** The owner's instruction: "if over 6 hours, run only R and ACG and tell me".
- **The estimate exceeds 6 h** (section 9): about 8 h of compute for all four arms run one after another. So **only R and
  ACG run**, and the owner is told.
- **A and AC.** Their code, constants and criteria are registered here unchanged. They run only if the owner asks, with
  the same seeds and rules, and are then reported against this file.
- **Holm correction.** "Holm over arms" in test (iii) is over the arms that ran: R and ACG.

## 3. Order and dropping rule
The tests run in this order: (i), then (ii), (iii) and (iv).
- An arm that fails (i) is reported and **dropped**: it is not run in (ii)-(iv).
- R is the reference and is not tested against itself in (i).
- If ACG is dropped, R still runs (iii) and (ii), as a fresh replication on new seeds and as a determinism check of the
  engine.

## 4. Test (i): the FlyWire brain does not break
- **Drive.** The 36 labellar sugar GRNs (`data/sugar_grn_783.csv`) at 100 Hz (Poisson input path) for 1 s (40 steps of
  25 ms), then all input cut for 200 ms (8 steps). Fresh state per seed.
- **Seeds.** 300-349 (50; fresh).
- **No-input check.** Seeds 300-302 without drive must give 0 spikes; otherwise the arm fails (i).
- **MN9.** The two `CB0701` neurons. Per seed, their mean rate in the 1 s drive window; per arm, the mean over seeds.
- **Ignition.** Network rate > 1 Hz in the last cut step (as prereg/02 and 08).
- **Pass (owner's criterion).** Both must hold:
  1. MN9(arm) >= 0.10 x MN9(R).
  2. The number of ignited seeds in the arm is <= that in R.
- **Reported, descriptive.** The ignition proportions with Clopper-Pearson 95 % CIs; a two-sided Fisher exact test of the
  arm vs R; the network rate during the drive; active neurons; the driven GRN rate.
- Code: `scripts/mv2_flywire.py`.

## 5. Test (ii): BANC VNC subgraph, the prereg/08 protocol
- **Protocol.** prereg/08 sections 1-6 unchanged (`scripts/f3a3_vnc_run.py --arm <arm>`): R1 recruitment, R2
  persistence, R3 steering dose-response (DNa04, DNa10, DNa15 at 10/25/50/100 Hz on the HAND DNg02 baseline), gates 1,
  2 and 3'. Seeds 0-9; gate-2 seeds 200-249.
- **Analysis.** The same as prereg/08 (`scripts/f3a3_report.py` `r2_section` / `r3_section`): the gate-passing steering
  types, Holm per rate, the steering decision per family and rate.
- **Headline.** The **lowest steering rate per family** in each arm next to R, reported as lower / equal / higher /
  none.
- **Criterion for an arm.** Gates 1, 2 and 3' pass, and R2 passes (>= 9 of 10 seeds decay). This is what R achieved in
  prereg/08. A variant that fails it is reported; its test (iii) result is then reported but marked as not usable for a
  pass.
- **Engine check.** R re-runs prereg/08 exactly (same code path, seeds and conditions). Its rates must equal the stored
  `runs/f3a3/f3a3_v3.json` bit for bit; whether they do is reported. A mismatch stops the work.
- R1 is descriptive: rates per type next to R.

## 6. Test (iii): hybrid optomotor, the prereg/11 protocol
- **Brain stage** (`scripts/f3b3_brain.py --arm <arm> --seeds 30-39`):
  - stimuli `rot_left`, `rot_right`, `static`, `grey` with 100 grey + 200 stimulus intervals of 5 ms;
  - inputs `vbnd_L` / `vbnd_R`, sugar 0 Hz;
  - **fresh brain seeds 30-39** (10);
  - per-interval DN spike counts.
  - The render is reused: `runs/f3b3/stim_rates.npz` (SHA-256
    `9b89aa1167ab0b7db79ca43ec549db70fdf243197ae8e8af1678301ced84f6d8`) and its passed input check. The render is
    deterministic, and F3b-3 recorded that it
    equals the F3b-1 render bit for bit.
- **VNC stage** (`scripts/f3b3_vnc.py --arm <arm> --seeds 30-39 --n-random 19`):
  - the prereg/08 subgraph; the 112-family ensemble and bridge of prereg/10-11 (gain 1, W = 5, L = 1); DNg02 `HAND`
    100 Hz.
  - Arms `ensemble`, `bridge_off` and `random_1` ... `random_19`: the first 19 derangements of prereg/11, permutation
    seeds 20261101-20261119, unchanged.
  - VNC seed = 1000 x stimulus code + brain seed.
  - Gates as prereg/11 section 10: gate 1 seeds 10-19, gate 2 seeds 250-299, gate 3' runaway < 5 Hz.
- **Primary statistic.** The predicted turn S, as in docs/08 (`scripts/f4_design_numbers.py`).
  - Mapping: the v2 `lateral` set, from the Lindsay et al. 2017 signs: b2 +, b3 -, i1 -, i2 -, hg1 -, hg2 -, hg3 -,
    hg4 -.
  - Per seed s: dA(stim, s) = - sum_m s_m [(R - L)_m(stim, s)], using the per-neuron BANC rates in the 1 s stimulus
    window.
  - S(s) = [dA(`rot_right`, s) - dA(`rot_left`, s)] / 2, in Hz. S > 0 is syndirectional.
  - The turn in degrees is G1 x S with G1 = 0.3949 deg/Hz (`docs/results/f4_design_numbers.json`, fixed); G1 changes
    neither sign nor test.
- **Pass for an arm.** All of the following:
  1. **Syndirectional and significant.** One-sided paired t-test of `rot_right` vs `rot_left`, i.e. one-sample t of S(s)
     against 0 with H1 mean > 0, over the 10 seeds. Holm over the arms that ran (section 2), alpha 0.05. Zero variance
     follows prereg/11.
  2. **Specific.** |mean S(`ensemble`)| > |mean S(`random_k`)| for **all** 19 random wirings. Ties count against
     `ensemble`; permutation p = (1 + #{k : |S_k| >= |S_ens|}) / 20 <= 0.05.
  3. **`bridge_off` silent.** Its S is not significant (two-sided t, p >= 0.05).
  4. **Gates.** The VNC gates of this stage pass, and the arm passed test (ii)'s criterion.

  Outcome per arm:
  - "pass" when 1-4 hold;
  - "syndirectional, significant, not specific" when 1 holds but 2 does not;
  - "not significant";
  - "anti-directional" when the mean S < 0 and a two-sided t has p < 0.05 (reported as a negative finding).
- **Secondary (reported, not used for the decision).**
  - S under the `direct` set (b2 +, b3 -).
  - The iii4 sensitivity sets `lateral+iii4(+)` and `lateral+iii4(-)`, and the combined text / Fig. 6 sets.
  - Per MN type m, D_m(s) = (R - L)(`rot_right`) - (R - L)(`rot_left`) and T_m = mean D_m. For every gate-passing
    steering type, a two-sided t with Holm over the types, and the contribution -s_m T_m / 2 to S.
  - The static offset B = [dA(`rot_right`) + dA(`rot_left`)] / 2.
  - S(ACG) - S(R) paired by seed (descriptive).
  - The applied and realised DN rates.
  - The GF (DNp01) rates and the TTMn / PSI / DLM rates in `ensemble`.

## 7. Test (iv): GF -> TTMn
- **Arms.** R and ACG (owner).
- **Protocol** (`scripts/mv2_gf.py`): BANC VNC subgraph; both GFs (DNp01, input nodes) at 100 Hz for 1 s, then 200 ms
  without input; no other input. Seeds 0-9.
- **Readouts.**
  - Following: per side, TTMn spikes / GF spikes in the drive window.
  - Latency: per side, each TTMn spike minus the latest preceding spike of the same-side GF, if < 5 ms.
  - PSI, DLM and network rates.
- **Criterion for ACG (the literature property, Phelan et al. 2008).** On both sides, the seed-mean following is
  >= 0.9 and the median latency over all seeds is < 1 ms. R is reported with the same numbers; no criterion applies to
  R.
- **What this tests.** It checks that the `gap` implementation reproduces the property it was built from (docs/09
  section 2.3). It is not an independent finding.

## 8. Controls (rule 7)
- In (iii): `bridge_off` (the ablation) and the 19 random wirings (activity-matched shuffles of the rate streams).
- As in prereg/06-11, degree-matched random graphs are not part of this open-loop model comparison. The report says so.

## 9. Run time (benchmark before this file; wall time only)
- **Benchmark.** 1.5 s simulated per engine and arm, at seed 9999, on the drive of the respective test (sugar 100 Hz on
  FlyWire; DNg02 100 Hz on the VNC subgraph). Second repetition, after compilation:

| | R | A | AC | ACG |
|---|---|---|---|---|
| FlyWire (s per 1.5 s simulated) | 4.0 | 5.0 | 11.3 | 11.2 |
| VNC subgraph (s per 1.5 s simulated) | 1.2 | 1.4 | 2.6 | 2.7 |

- **Projection.** The measured R durations of prereg/08 (38 min), prereg/11's brain stage (10.3 s per run) and its VNC
  stage (1.45 s per replay), scaled by these ratios:

| test | R | A | AC | ACG |
|---|---|---|---|---|
| (i) 53 FlyWire runs | 5 min | 6 min | 10 min | 10 min |
| (ii) prereg/08 on the VNC | 38 min | 45 min | 84 min | 86 min |
| (iii) brain 40 runs | 7 min | 9 min | 20 min | 20 min |
| (iii) VNC 21 arms x 40 + gates | 22 min | 26 min | 48 min | 50 min |
| (iv) | 2 min | - | - | 4 min |

  - All four arms one after another: about **8.1 h** of compute, above the 6 h limit.
  - R and ACG only: about **4.1 h**.
- **Execution.** Two processes may run at the same time when `free -h` shows enough memory: FlyWire about 4 GB, VNC
  about 3 GB. `free -h` is checked before each stage.

## 10. What is reported
`scripts/mv2_report.py` writes `docs/results/mv2_neuron_model.{md,json}`. Every number comes from the run records, and
`scripts/verify_report.py` checks them. Negative results are reported with equal weight.

No threshold, window, seed, test, mapping, gain, arm list, permutation or constant is changed afterwards.

## 11. Origin classes (rule 2)
- The new model constants are registered in `configs/assumptions.json` (`model_v2.*`), with origin classes as in
  section 1.
- **Protocol settings of this document (E):**
  - seeds 300-349 for (i), 30-39 for (iii) and 0-9 for (iv);
  - the 10 % MN9 threshold and the ignition rule (owner);
  - the (iv) thresholds 0.9 and 1 ms (literature property);
  - the pairing rule of the junctions.
