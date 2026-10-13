# Summary F0 - F3b-3: what works, what does not, what we can claim

Written 2026-10-14 by `scripts/summary_f0_f3.py`. Analysis and writing only: nothing was simulated for this document.
Every number is read from the linked report by a regular expression (`scripts/summary_f0_f3.py`, table `FACTS`) and
re-checked against that report by `scripts/verify_report.py`; the facts are in `docs/results/summary_f0_f3.json`.
Plan and decisions: `docs/05_ARCHITECTURE_PLAN.md`. Next step (design only): `docs/08_CLOSED_LOOP_OPTOMOTOR.md`.

## 1. The path so far

| phase | question | result | report |
|---|---|---|---|
| F0 | does the ported v1 stack reproduce v1 (FlyWire brain + body)? | yes; one realisation shows a self-sustaining ignition | [f0_acceptance](f0_acceptance.md), [f0_multiseed](f0_multiseed.md), [f0_ignition_frequency](f0_ignition_frequency.md) |
| F1 | does BANC load, and does the reference arm run on it? | loads and runs; MN9 is silent | [f1_banc_qa](f1_banc_qa.md), [f1_banc_smoke](f1_banc_smoke.md) |
| F2 | can the BANC whole-brain graph replace the FlyWire brain? | no; the track stops and the hybrid starts | [f2_wsyn_calibration](f2_wsyn_calibration.md), [f2_full_graph](f2_full_graph.md), [f2_innorm](f2_innorm.md) |
| F3 prep | which DN families are shared and pass the symmetry gate? | 112 of 348 shared families | [f3_dn_crosswalk](f3_dn_crosswalk.md) |
| F3a, F3a-2, F3a-3 | does the BANC VNC (subgraph) turn DN activity into wing MN activity? | yes for wing power; steering by DNa04, DNa10, DNa15 | [f3a_banc_vnc](f3a_banc_vnc.md), [f3a2_checks](f3a2_checks.md), [f3a3_vnc_subgraph](f3a3_vnc_subgraph.md) |
| F3b-1 | does a rotating grating reach the VNC through three DN families? | no | [f3b1_optomotor](f3b1_optomotor.md) |
| F3b-2 | does a 112-family DN ensemble carry it? | direction-selective, not specific | [f3b2_dn_ensemble](f3b2_dn_ensemble.md) |
| F3b-3 | is that specific to the true wiring (fresh data, permutation test against 39 random wirings)? | replicated, not specific | [f3b3_specificity](f3b3_specificity.md) |

## 2. What works

- **The v1 flight stack is ported and reproduces v1 at the acceptance criteria.** In the multi-seed acceptance,
  10 of 10 flights pass S1 and S2; the largest position difference to v1 is 0.27 mm
  ([f0_acceptance](f0_acceptance.md), [f0_multiseed](f0_multiseed.md)).
- **The BANC v3 VNC subgraph is a usable motor stage.** It keeps 25,818 of the BANC neurons
  (3,020,333 pairs, 11,585,331 synapses). Persistence is solved: activity stops within 25 ms of the
  cut in 10 / 10 seeds, where the whole graph does not decay ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **Wing power comes from the VNC.** Bilateral DNg02 drive at 100 Hz drives DLM1-4 (53.8 Hz) and DLM5
  (46.6 Hz) ([f3a_banc_vnc](f3a_banc_vnc.md)). DNg02 itself is silent in the FlyWire brain
  (0.0 Hz in every F0 flight window), so in the hybrid it is a `HAND` baseline
  ([f3a2_checks](f3a2_checks.md)).
- **Three DN families steer ipsilaterally in the VNC.** DNa04 steers from 25 Hz, which is inside its FlyWire
  flight range (L / R 26.1 / 16.8 Hz) ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **The FlyWire brain turns grating direction into DN activity.** In F3b-1 the direction-selective families after Holm are DNa04, DNa10; in F3b-3
  25 of 112 ensemble families are, on fresh seeds ([f3b1_optomotor](f3b1_optomotor.md),
  [f3b3_specificity](f3b3_specificity.md)).
- **Direction information passes through the hybrid reliably.** With the 112-family ensemble bridge the
  steering MN asymmetry is direction-selective in b3 (+2.800 Hz), i1 (+2.750 Hz), hg1 (+2.750 Hz) and hg4
  (-11.650 Hz), replicated on fresh brain seeds with the same signs as F3b-2; the cut-bridge control is silent
  ([f3b3_specificity](f3b3_specificity.md) section 4).
- **The gates behave.** In F3b-3 gate 1 gave no spikes, gate 2 gave 0 / 50 ignitions with all 305
  ensemble neurons and DNg02 at 100 Hz, and the network rate stays at or below 2.8256 Hz (gate-2 condition)
  ([f3b3_specificity](f3b3_specificity.md) section 2).

## 3. What does not work

- **The BANC whole-brain graph with the reference parameters does not drive MN9.** MN9 is 0.00 Hz in every
  condition on the Codex export ([f1_banc_smoke](f1_banc_smoke.md)); on the full v888 pair-level graph it is still
  0.0 Hz against 59.7 Hz in FlyWire ([f2_full_graph](f2_full_graph.md)).
- **Uniform w_syn scaling ignites the network before it makes MN9 respond.** Ignitions after the cut: 20 / 20
  (structural scale), 19 / 20 (class scale), 18 / 20 (fitted to MN9), and 0 / 20 in the unscaled
  reference arm, which is silent at MN9 ([f2_wsyn_calibration](f2_wsyn_calibration.md)).
- **The best BANC arm is lateralised and does not qualify.** On the v3 graph (unchanged parameters) MN9 reaches
  6.8 Hz, just above the 5.97 Hz threshold, but through one neuron on one side, the leg path is silent
  and gate 3 fails ([f2_innorm](f2_innorm.md)).
- **Three DN families are too few.** In F3b-1 no arm is direction-selective in the VNC: the highest applied rate of
  DNa04 is 6.9 Hz against its lowest steering rate of 25 Hz ([f3b1_optomotor](f3b1_optomotor.md)).
- **The effect is not specific to the true wiring.** In F3b-3 the true wiring beats 30 / 39 (b3),
  30 / 39 (i1), 22 / 39 (hg1) and 36 / 39 (hg4) of the random wirings, with permutation
  p of 0.250, 0.250, 0.450 and 0.100 against a criterion of 0.05
  ([f3b3_specificity](f3b3_specificity.md) section 4).

## 4. Negative and mixed results (reported with equal weight)

- **F0, ignition and latency** ([f0_ignition_frequency](f0_ignition_frequency.md),
  [f0_latency_hover_yaw](f0_latency_hover_yaw.md)): the new engine ignited the olfactory self-sustaining mode in
  2 / 112 realisations (1.8 %) against 0 / 136 for v1; the registered 100-seed test found
  0 / 100 ignitions per arm (95 % upper bound 3.6 %), so equality is not demonstrated and not
  excluded. The causal-latency effect on a brain-closed hover flight (about 12 deg of heading, up to
  15-28 deg) is not separable from seed-to-seed variability.
- **F2, BANC reference arm silent at MN9** ([f1_banc_smoke](f1_banc_smoke.md), [f2_full_graph](f2_full_graph.md),
  [f2_innorm](f2_innorm.md)): a data property (sparser graph, known proofreading asymmetry), not something tuned away.
- **F3a, expectations (b) and (c) missed** ([f3a_banc_vnc](f3a_banc_vnc.md)): steering MNs responded on one side only
  on the whole graph, and unilateral DNa02 did not steer. On the subgraph the one-sided MN responses are gone
  ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **F3b-1, the three-family bridge** ([f3b1_optomotor](f3b1_optomotor.md)): not direction-selective in any arm.
- **F3b-2, outcome B** ([f3b2_dn_ensemble](f3b2_dn_ensemble.md)): `ensemble` is direction-selective, but
  so is the shuffled control, with b3 and hg1 of opposite signs (+3.00 and -3.50 Hz for b3;
  +4.40 and -8.70 Hz for hg1). That led to the F3b-3 hypothesis that the wiring sets the sign.
- **F3b-3, the sign hypothesis is not supported** ([f3b3_specificity](f3b3_specificity.md)): the fraction of random
  wirings with the same sign as `ensemble` is 0.590 (b3), 0.462 (i1), 0.538 (hg1) and 0.615
  (hg4); the true wiring does not set the sign.
- **F3b-2 and F3b-3 used different brain seeds.** The F3b-2 ensemble had 19 direction-selective
  families and F3b-3 has 25 (different seeds); the lists overlap but are not equal. Nothing downstream
  depends on the exact list.

## 5. Open items

1. **Which DN families carry steering in flight.** The bridge uses the shared, gate-passing families only; the
   real fly uses more DN types (BANC-only DNs are not bridged).
2. **The sign of the steering MN response.** It is set by the VNC and the ensemble as a whole, not by the specific
   DN wiring. Its relation to the literature sign is partial (below).
3. **The kinematic effect of each steering muscle.** Table 1 of Cheong et al. lists activity on the inner and outer side
   of a turn, not the effect on stroke amplitude; only two muscles have a direct statement in the sources read
   (`docs/08_CLOSED_LOOP_OPTOMOTOR.md` section 4.1).
4. **Neck and haltere MN responses** are lost on the subgraph (they depended on removed BANC brain / SEZ parts) and
   need DNs outside the current set.
5. **No body, no feedback so far.** Everything up to F3b-3 is open loop: no visual feedback, no muscle model, no
   VNC -> brain feedback. T2 (`docs/08_CLOSED_LOOP_OPTOMOTOR.md`) is a design, not a result.
6. **Neuron-model realism** (no adaptation or background; the LIF sits far from threshold) was listed in docs/05 as
   explanation 3 for the weak drive and has not been tested.

## 6. What we can and cannot claim

**We can claim**
- The FlyWire v783 brain model turns the direction of a rotating grating into direction-selective DN activity, and a
  DN ensemble bridge carries that into steering-MN asymmetry of the BANC v3 VNC subgraph, reproducibly across fresh
  seeds, and only when the bridge is on.
- The BANC VNC subgraph is a stable, non-self-sustaining motor stage for wing power and steering MN asymmetry under
  tonic wing-power drive.
- The BANC whole-brain graph with the unchanged reference parameters is not a usable brain replacement.

**We cannot claim**
- That the specific DN wiring steers: random derangements of the same rate streams give MN asymmetries of the same size
  and either sign.
- That the hybrid turns a fly with or against the drum: nothing has been closed in a loop, and the mapping from MN rate
  to wing kinematics has no direct literature support for most types.
- That the match to the literature sign is evidence of function: 4 of 6 comparable types agree and
  3 of the 4 significant ones agree (hg4 is reversed), and this comparison was exploratory.
- Anything about BANC brain circuits, olfaction, landing, take-off or learning: none of that was run in the hybrid.
