# 10. F5 training: design

Status: **design only**, written 2026-10-15 before any F5 code or run. The binding criteria are in
`prereg/14_f5_training.md`; where the two differ, the pre-registration wins. Decision context: the project owner's
record after PR #19 (`docs/05_ARCHITECTURE_PLAN.md`, 2026-10-15): the side audit found no mislabelling, the
anti-directional S is a property of the untrained hybrid, and the project moves to real training at the scale of
flyhard and flybody. Run time does not matter; correctness, controls and resumability do. The sign diagnostic is
postponed.

## 1. Question

RQ4 (docs/05 section 1): does a connectome with fixed topology and trained dynamic parameters learn a flight task, and
does the real wiring do better than control wirings?

F5 asks this for one task: **optomotor yaw (T2, docs/08)**. A drum rotates around the hovering fly; the fly should turn
with it (syndirectional). Until now every result used the untrained spiking hybrid. Its direction signal passes
reliably but with the wrong net sign and no wiring specificity (F3b-3, mv2, side audit). F5 trains the network instead
of reading it out by hand.

The task ladder of docs/05 (hover -> optomotor -> obstacle -> landing) starts here at optomotor. Hover is held by
the plant in docs/08, and only yaw is free.

What F5 can and cannot show:
- It can show that the connectome **learns** the task. It can also show whether the true wiring learns it **better**
  than a degree-matched shuffle, a family derangement, an MLP of similar size and a linear map. The comparison uses
  the same data, budget and test.
- It does **not** show that the untrained fly does this. Trained parameters are labelled `TRAINED` and never mixed with
  `BRAIN` / `VNC` results.
- The spiking reference arm R stays unchanged (rule 8). The rate model is a separate engine, not a variant of R.

## 2. The model: a sparse rate network (flyhard recipe, docs/04 section 1.1)

### 2.1 Dynamics

Each neuron i has a signed scalar state s_i in (-1, 1). Once per update:

```
s  <-  (1 - lam) * s  +  lam * tanh( W_eff s  +  u )
W_eff[i, j] = g_ij * sign_j * n_ij / N_i          (fixed sparsity: only pairs with n_ij > 0)
g_ij  = G_MAX * sigmoid(theta_ij)                  (bounded edge gain, init 1)
lam_i = LAM_MIN + (LAM_MAX - LAM_MIN) * sigmoid(phi_i)   (bounded leak, one per neuron)
```

- **Synapse counts.** n_ij is the synapse count from j to i.
- **Normalisation.** N_i = sum_j n_ij is the total input synapse count of i. Each neuron's summed absolute input
  weight is therefore 1, as in flyhard and flydrive (docs/04 section 4, item 2).
- **Sign rule.** sign_j is the sign of presynaptic neuron j under the reference sign rule of each dataset. For BANC
  that is prereg/01 (Shiu rule: GABA, GLUT -> -1, else +1). For FlyWire it is the rule of the F0 brain.
  - The sign is fixed (Dale's law); a trained gain cannot flip it.
  - flyhard used unsigned weights. Using the signs is our choice, recorded as a difference from flyhard.
- **Trainable parameters.** One gain per edge (theta, init 0, so g = 1) and one leak per neuron (phi). Nothing else in
  the network is trained.
  - The topology is fixed: no edge is added or removed.
  - The gain is bounded in (0, G_MAX) with G_MAX = 2, so training can at most double or nearly silence an edge.
  - The leak is bounded in (LAM_MIN, LAM_MAX) = (0.05, 0.95), as in flyhard. Its initial value lam_0 = 0.25 is the
    5 ms update over the Shiu membrane time constant of 20 ms. phi_0 = logit((lam_0 - 0.05) / 0.9).
- **Time step.** One update per 5 ms coupling interval (`UPDATES_PER_INTERVAL = 1`), the clock of every F3 experiment.
  A k-hop path therefore takes k intervals. This is an explicit latency and is recorded.
- **State is carried.** It is never reset within a trial or flight (fly-self-driving, docs/04 section 1.2). Each
  trial or flight starts from s = 0.
- **Label.** Every motor term produced by a trained arm is labelled `TRAINED` and recorded separately (rule 1).

### 2.2 Graphs

| part | neurons | edges (pairs) | trainable parameters | source |
|---|---|---|---|---|
| BANC v3 VNC subgraph (prereg/08) | 25,818 | 3,020,333 | about 3.05 M | `docs/results/f3a3_vnc_subgraph.md` |
| FlyWire v783 brain | 138,639 | 15,091,983 | about 15.2 M | `docs/results/connectome_stats.json` |
| both (stage B, C) | 164,457 | 18,112,316 | about 18.3 M | sum |

The trainable counts are the edges plus one leak per non-input neuron. They are recounted from the graphs by the
implementation and recorded.

### 2.3 Inputs, bridge and readout

- **Input nodes are clamped.** Their state is set from outside each interval and they have no leak.
  - In the VNC, these are the BANC DNs.
  - In the brain (stage B, C), the visual boundary neurons `vbnd_L` / `vbnd_R` (`sense/vision_boundary.py`) receive
    their drive as u = rate / R_IN through tanh, like any other neuron.
- **Rate scale.** R_IN = 100 Hz per unit of state or drive. It is the scale on which the F3a-3 steering
  dose-responses were measured (10-100 Hz); origin `E`.
- **DN bridge.** The F3b-2 / F3b-3 ensemble: 112 families, 305 BANC neurons, 224 family-side units. It keeps the
  same matching, the same window (5 intervals) and the same latency (1 interval).
  - **Stage A.** The unit rate comes from the recorded FlyWire spike counts. BANC DN state = unit rate / R_IN.
  - **Stage B, C.** The unit value is the window mean of relu(s) over the FlyWire DNs of the unit.
  - **Difference from F3b.** There is no Poisson re-synthesis: the rate model takes the rate itself. This is
    recorded as a named difference.
- **HAND baseline.** The DNg02 baseline is clamped at 100 Hz / R_IN = 1 on both sides, labelled `HAND` (docs/05,
  decision after F3a-2). Every other BANC DN is clamped to 0.
- **Readout (fixed, never trained).** The 16 steering MNs of the Lindsay `lateral` set: b2, b3, i1, i2, hg1, hg2, hg3
  and hg4, one neuron per side (side audit: every type consistent).

  `dA = A_L - A_R = G1 * R_OUT * sum_m sigma_m (s_{m,L} - s_{m,R})`

  - sigma_m are the Lindsay signs (b2 +; the others -).
  - G1 = 0.395 deg/Hz, the v1-fixed value of docs/08 section 10.4.
  - R_OUT = R_IN = 100 Hz per unit of state.
  - The MN state is read as a signed deviation from rest. Only the left-right difference enters, so a common offset
    cancels.
- **Symmetry gate (rule 5).** The 112 families pass the gate in both datasets, and the 8 readout types pass it on the
  subgraph (docs/08). The implementation recomputes both gates and records the results in every stage report.

### 2.4 Causal timing (rule 4)

```
interval k:  brain state k   <- brain update(state k-1, visual drive of interval k-1)
             BANC DN clamp k <- bridge(FlyWire DN values of intervals k-5 .. k-1)        latency 1 interval
             VNC state k     <- VNC update(state k-1, DN clamp k)
             dA_k            <- readout(VNC state k)
             body at k+1     <- dA_k  (command queue, one interval late)
```

Nothing is driven by values produced in the same interval. Each latency is a configuration value written into every
record. A unit test registered in prereg/14 checks this: a perturbation at interval k changes dA no earlier than the
registered total latency.

## 3. Curriculum

Every stage uses the same arms (section 5) and the same budget per arm. Each stage starts only after its gates pass
(section 9). A stage that fails a gate is reported as such and not run; the next stage still runs if its own gates
pass.

### Stage A: VNC subgraph alone, supervised, open loop

- **Input.** Recorded FlyWire DN rates from the spiking reference brain R (section 6), through the bridge.
- **Network.** The rate VNC subgraph. **Output:** dA through the fixed readout.
- **Target.** The syndirectional dA of section 4, per interval.
- **Question.** Can the trained VNC turn the brain's real direction signal into syndirectional steering, and does the
  BANC wiring help?
- **Cost.** This is the cheapest stage. It also exercises the whole training, checkpoint and evaluation machinery.

### Stage B: end to end, open loop

- **Chain.** FlyVis (frozen, the pinned FlyVis 1.2.0 model) -> boundary layer (unchanged) -> rate FlyWire brain ->
  bridge -> rate BANC VNC -> 16 steering MNs -> dA (Lindsay `lateral`, G1).
- **What is trained.** Brain and VNC together; FlyVis is not trained.
- **Same task and targets as A.** The visual stimuli are those of the stage A brain runs.
- **Warm start.** The VNC parameters start from the stage A result of the same arm and training seed. The brain starts
  at gain 1 and lam_0.
- **Precomputation.** FlyVis is frozen and the stimuli are open loop, so the boundary rate sequences are computed once
  per (stimulus, speed, phase). Per-trial noise is then added (section 6.3).

### Stage C: closed loop, behaviour cloning + DAgger

- **Task and plant.** The docs/08 task: 0.5 s grey, then a 2 s drum. The plant is the docs/08 one-degree-of-freedom
  yaw plant (stroke-averaged yaw term, the `REFLEX` haltere yaw damping kept and labelled). The yaw trim is the
  static-grating trim of docs/08 section 10.7. FlyVis runs online, because the image depends on the fly's own yaw.
- **Teacher.** The ideal syndirectional optomotor controller of section 4.
  - It sees only the **retinal slip**, i.e. the drum's angular velocity relative to the eye, which is derivable from
    vision (rule 9).
  - It acts through the same one-interval command queue as the learner.
  - Its commands are labelled `HAND` (teacher) and recorded separately.
- **Behaviour cloning (BC).** Teacher rollouts on the training data seeds. They do not depend on the arm, so they are
  generated once and shared by every arm and training seed. The policy (the stage B network of the same arm and
  training seed) is trained to predict the teacher's dA from the recorded boundary input sequence, using BPTT over the
  whole flight.
- **DAgger.** In each round, the learner flies (beta = 0: the learner's own commands drive the plant). The teacher
  labels every visited interval. The new flights are added to the data set, and training continues from the previous
  round's parameters with the round's step budget. There are 5 rounds; fly-self-driving needed 3 (docs/04 section
  1.2).
- **Evaluation.** Test flights with the learner in control, no teacher.

### Stage D: optional reinforcement fine-tuning

D is not designed here. If it is pursued, it gets its own pre-registration (a new file, committed before any D run).
- Any D result is labelled separately and does not enter the F5 criterion.
- Constraints fixed now:
  - the reward may use only quantities derivable from the senses (retinal slip, the fly's own commands);
  - D starts from the stage C parameters;
  - it uses the same arms, splits and test.

## 4. Targets and teacher

The v1 hand-made turn range (docs/08 section 4.2):
- A_L - A_R spans 25 deg, so A_MAX = 12.5 deg per direction;
- 1 deg of A_L - A_R is worth K_YAW = 28.6 deg/s of steady yaw rate (`yaw_rate_per_turn / (2 turn_dphi_deg)`).

**Teacher law, the same in every stage.** It is a proportional controller on the retinal slip speed v_slip (deg/s,
positive when the pattern moves to the fly's right):

`dA*(t) = clip( K_P * v_slip(t), -A_MAX, +A_MAX )`,   with   `K_P = A_MAX / V_MAX`,   V_MAX = 120 deg/s (fastest drum).

- **Size rule.** The fastest drum just reaches the v1 range limit and slower drums scale linearly inside it, so the
  target size comes from the v1 range and is not chosen by looking at any response. K_P = 12.5 / 120 = 0.104 deg per
  deg/s.
- **Sign.** rot_right (the pattern moves to the fly's right) gives dA* > 0. That is a larger left amplitude, which
  turns the fly to the right, i.e. syndirectional (`body/quasi_steady.turn_to_amplitudes`; docs/08 section 6: S > 0
  is syndirectional).
- **Open loop (stages A, B).** The fly does not turn, so v_slip is the drum speed: dA* = +-3.125, +-6.25 and +-12.5 deg
  at 30, 60 and 120 deg/s. dA* = 0 under static and grey. During the grey lead-in the target is 0, and at drum onset
  it steps to its value.
- **Closed loop (stage C).** v_slip = drum speed - yaw rate. With the steady plant, the teacher's closed-loop
  optomotor gain is c / (1 + c), where c = K_YAW * K_P = 2.98, so g_teacher = 0.75. The implementation recomputes
  this from the configuration, and the teacher's realised gain (with latency and damping) is reported as the
  reference.
- **Why a proportional slip controller.**
  - It is the simplest controller that uses only what the eye sees.
  - It is the classical description of the optomotor response.
  - The same law gives the open-loop targets, so stages A-C share one target definition.
  - An integral term would need the absolute pattern position. It is not used.

## 5. Arms and controls (rule 7)

The same arms run in every stage, with the same data, splits, optimiser, step budget and selection rule.

| arm | network | role |
|---|---|---|
| `true` | the connectome graphs of section 2.2 | the hypothesis |
| `shuffled` | each graph rewired by degree-preserving directed edge swaps (Maslov-Sneppen): two edges a->b, c->d become a->d, c->b. A swap is rejected if it would create a self-loop or a duplicate pair. 10 x E successful swaps. Each edge keeps its synapse count, and the sign stays with the presynaptic neuron. In-degree, out-degree, per-neuron sign (Dale) and the per-neuron normalised input magnitude are therefore identical to `true` | **random graph with the same degrees**; the primary comparison |
| `derange` | the `true` graphs, with the bridge's family -> unit mapping replaced by a random family derangement (the prereg/11 generator; training seed t uses `random_(t+1)`) | **activity-matched shuffle** (the project's sense, prereg/11 section 11): the same DN rate streams on the wrong families |
| `mlp` | a feed-forward MLP from the arm input to the 16 readout MN states, then the same fixed readout | non-connectome control with a similar parameter count |
| `linear` | one linear map from the same input features to the 16 MN states, then the fixed readout | linear control |
| `bridge_off` | `true`, with every bridged DN clamped to 0 (DNg02 baseline kept) | **ablation**; must stay at chance and also detects leakage |
| `untrained` | `true` at initialisation, evaluated without training | what training adds; also the rate model's equivalence check against the spiking hybrid (docs/05 section 3.1) |

- **The shuffle is redrawn per training seed.** In stages B and C both the brain and the VNC graph are shuffled; the
  bridge is unchanged.
- **Input features of `mlp` and `linear`.** These networks have no recurrence. To give them a temporal context
  comparable to the rate network, each input channel is passed through four fixed exponential filters (time constants
  10, 25, 50 and 100 ms).
  - Stage A input: the 224 bridge units.
  - Stage B, C input: the boundary rates.
- **MLP size.** Two hidden layers of equal width h (ReLU). h is the smallest width whose parameter count is at least
  the `true` arm's trainable count, and it is recorded.
- **What the comparison measures.** In `mlp` and `linear` the connectome is replaced. So `true` vs `mlp` asks whether
  any structure beats the connectome's, while `true` vs `shuffled` asks whether the specific wiring matters.
- **Training seeds.** 10 per trained arm (t = 0-9; at least 5 is the rule-7 minimum), and 5 for `bridge_off`.
  - A training seed sets the data order, the shuffle, the derangement, the MLP initialisation, the stage B input noise
    and the stage C plant noise.
  - The `true` initialisation itself is deterministic (all gains 1, all leaks lam_0). Its seed-to-seed variance comes
    only from the data order and the noise. This is stated with every result.
- **Same budget.** The arms get the same optimiser (Adam) and the same step count, batch size and validation
  schedule. The learning-rate grid has the same size for every arm.
  - The learning rate is chosen per (stage, arm) on **validation loss** with training seed 0 and then fixed for seeds
    1-9.
  - It is an optimiser setting, not a model constant. It is chosen on imitation loss, never on closed-loop
    behaviour (rule 3).

## 6. Data

### 6.1 Split by seed (fixed now)

Data seeds index independent noise realisations: brain noise in A, input noise in B, plant noise in C. In new runs
they also set the grating phase.

| split | data seeds | use |
|---|---|---|
| train | 0-159 | gradient steps |
| validation | 160-199 | learning-rate choice, checkpoint choice, DAgger stopping |
| test | 200-239 | read once, at the end of each stage, from the selected checkpoint |

- **Sealed test.** The test seeds are listed in a split file with a SHA-256 that is recorded in every training run.
  Test records may be generated early (A), but no training or selection code path reads them. This is checked by a
  test that fails if a test seed reaches a loader in training mode.
- **Held-out drum speed.** 30 deg/s (temporal frequency 1 Hz) is never used for training or validation in any stage.
  The training speeds are 60 and 120 deg/s (2 and 4 Hz). The wavelength is always 30 deg (`tasks/t0_stimuli.py`).
  - So the generalisation test is an extrapolation one octave below the trained range.
  - It is also the hardest target size (3.125 deg), and the target law predicts it exactly.
- **Grating phase.** It is drawn per data seed, uniform over one wavelength. Without it, test trials would differ from
  training trials only in noise.
  - Seeds 0-39 keep phase 0, so that their existing records stay valid (they are all training seeds).
  - Seeds 40-239 draw the phase from `default_rng(seed)`.

### 6.2 Data for stage A: existing records

All the existing records come from the unmodified spiking FlyWire brain R with the same model and weights:
- `model_sha256` fc45837d...;
- `weights_sha256` d5547fb8...;
- the same 4 stimuli at 60 deg/s;
- 100 grey + 200 stimulus intervals of 5 ms;
- per-interval spike counts of all 1,299 FlyWire DNs.

| records | seeds | stimuli | count | used for A |
|---|---|---|---|---|
| `runs/f3b1/brain/` | 0-9 | rot_left, rot_right, static, grey | 40 | yes (train) |
| `runs/f3b3/brain/` | 10-29 | same | 80 | yes (train) |
| `runs/mv2/t3_R/brain/` | 30-39 | same | 40 | yes (train) |
| `runs/mv2/t3_ACG/brain/` | 30-39 | same | 40 | **no**: a different model variant (ACG), not the reference brain |
| F0 flight, F2, ignition records | - | other protocols | - | no |

In total, 160 existing records (40 seeds x 4 stimuli) are reused. The implementation checks their SHA-256 and
model and weight hashes before use, and fails if they do not match.

### 6.3 Data for stage A: new brain runs

The new runs use the same brain, render chain, input check and record format (the F3b-3 brain stage), with two
additions behind explicit options: drum temporal frequency, and grating phase. The extended script must reproduce an
existing record bit for bit at 60 deg/s and phase 0: `rot_left` seed 10 must equal `runs/f3b3/brain/rot_left_s10.npz`.

| new runs | seeds | conditions | count |
|---|---|---|---|
| new seeds, all conditions | 40-239 (200) | rot_left / rot_right at 30, 60 and 120 deg/s; static; grey | 200 x 8 = 1,600 |
| existing seeds, the two new speeds | 0-39 (40) | rot_left / rot_right at 30 and 120 deg/s | 40 x 4 = 160 |
| **total** | | | **1,760** |

- **Cost.** In F3b-3, 80 runs took 827 s including the engine build (`runs/f3b3/brain/DONE`), so 1,760 runs take
  about 5 h. One record is about 20 KB, so the whole set is about 40 MB.
- **After the runs.** Every data seed 0-239 has 8 records. That makes 1,920 records (about 0.58 M labelled intervals
  at 300 intervals each). The training records are 960: data seeds 0-159 x the 6 training conditions, since 30 deg/s
  is held out. That is the scale of flyhard's 1,200 demonstrations.

### 6.4 Data for stages B and C

- **Stage B.** FlyVis boundary rate sequences for each (stimulus, speed, phase) of section 6.1, computed once.
  - Per data seed, independent Gaussian noise with the Poisson variance of a 5 ms count is added to every boundary
    rate (s.d. = sqrt(rate / 5 ms), rates clipped at 0). This replaces the spiking model's Poisson input.
  - No brain records are needed.
- **Stage C.** Online closed-loop flights.
  - **Plant noise.** Each flight's seed drives an Ornstein-Uhlenbeck yaw-rate disturbance (`body/noise.py`). Its s.d.
    is 6 deg/s (10 % of the 60 deg/s drum) and tau 50 ms (the existing `gust_tau`); origin `E`.
  - **BC data.** Teacher flights for training data seeds 0-159 x the 6 training conditions (no 30 deg/s) = 960
    flights.
  - **DAgger round.** 160 learner flights (one per training data seed; the condition comes from a fixed seeded
    schedule balanced over the 6 training conditions). Plus 40 validation flights (seeds 160-199) for the stopping rule.
  - **Test.** Seeds 200-239 x 8 conditions = 320 flights per arm and training seed, run once.

## 7. Metrics (registered in prereg/14)

The analysis window is the stimulus intervals after the first 100 ms. In A and B that is intervals 120-299 of each
trial; in C it is the last 1.9 s of the drum.

| metric | definition | stages |
|---|---|---|
| test turn-sign accuracy | fraction of test rotation trials at training speeds where the window mean of dA has the sign of dA* | A, B |
| test R2 of dA | 1 - SSE / SST of the window-mean dA against dA* over all test trials at training speeds (rotations, static, grey); **primary** for A and B | A, B |
| samples to set accuracy | training trials presented until validation turn-sign accuracy first reaches 0.90 (evaluated every 50 optimiser steps); censored at the budget | A, B, C (validation flights) |
| generalisation | test turn-sign accuracy and R2 on the held-out speed (30 deg/s) | A, B, C |
| optomotor gain g | `g = (psi_dot(rot_right) - psi_dot(rot_left)) / (2 v_drum)` per test data seed at each training speed, averaged; **primary** for C | C |
| stability | fraction of test flights that are stable: dA at the +-A_MAX limit in < 10 % of window intervals, **and** s.d. of the yaw rate over the last 1 s < 0.5 x drum speed (static, grey: < 30 deg/s) | C |
| drift | mean yaw rate under static and grey | C |
| imitation R2 | interval-level R2 of the learner's dA against the teacher label on test flights | C |

Every learning curve is recorded: validation loss, sign accuracy and R2 every 50 optimiser steps; in C also per DAgger
round.

## 8. Criterion: "the connectome helps"

- **Per stage.** The primary metric M (A, B: test R2; C: test g) is computed per training seed.
  - **Test:** a one-sided paired sign-flip permutation test on d_t = M(`true`, t) - M(`shuffled`, t), t = 0-9.
    It is exact over all 2^10 sign patterns, H1: mean d > 0.
  - **Multiple stages:** alpha = 0.05, Holm over the three stages {A, B, C}. A stage that is not run counts as p = 1.
- **Verdict per stage:**
  - **"The connectome helps" (stage X):** Holm-adjusted p < 0.05.
  - **"No advantage":** otherwise.
  - **"Shuffled better" (descriptive):** a two-sided test at 0.05 in the other direction.
  - **Stability guard (C only).** If either arm has stability < 0.8, the C verdict is "not interpretable
    (instability)", whatever p is.
  - **Ceiling guard.** If both arms have mean M >= 0.98 (A, B) or both reach >= 0.98 g_teacher (C), the verdict also
    states "ceiling: the task does not separate the wirings".
- **Secondary (same test, reported, no Holm, not part of the verdict):**
  - `true` against `derange`, `mlp` and `linear`;
  - samples to set accuracy and generalisation between `true` and `shuffled`.
- **Learning claim.** "Stage X is learned" is reported per arm: test sign accuracy >= 0.9 and R2 > 0 (A, B), or
  g > 0 with stability >= 0.8 (C), in at least 8 of 10 training seeds. It is descriptive and separate from the
  connectome criterion.
- **Negative results count.** "No advantage" is reported with the same weight (rule 10).

## 9. Gates and correctness checks (before any training run of a stage)

1. **Information alignment (rule 9).** Before stage A, two brainless bypass arms. Each needs a cross-validated R2 > 0
   (5-fold by data seed, training seeds only) for the trial-mean dA*:
   - ridge regression on a fixed random projection (2,048 dimensions) of the FlyVis output;
   - ridge regression on the 224 filtered bridge units.
   In C, the same bypass runs on the BC teacher flights, with the interval-level teacher label as the target.
2. **Symmetry gate (rule 5).** Inputs and readouts, recomputed and reported (section 2.3).
3. **Gradient check.** The custom sparse backward (chunked per-edge derivative and stored transpose, as in flyhard)
   is compared with dense autograd in float64 on a 2,000-neuron subgraph: maximum relative error < 1e-6.
4. **Forward reference.** The PyTorch forward equals a plain NumPy implementation of section 2.1 on a 2,000-neuron
   subgraph over 300 intervals: maximum absolute difference < 1e-5 in float32.
5. **Shuffle check.** In- and out-degree vectors are identical, the per-presynaptic sign is identical, there are no
   self-loops or duplicates, and the fraction of edges left in place is reported.
6. **Causality check (rule 4).** An input perturbation at interval k changes dA no earlier than the registered total
   latency.
7. **Record checks.** The SHA-256 and model hashes of the existing records match (section 6.2), and the new-run
   script reproduces `rot_left_s10` bit for bit (section 6.3).
8. **Leakage check (after training).** `bridge_off` must have test sign accuracy within the binomial 95 % interval of
   0.5. If it does not, the stage is marked "leak" and its verdict is withheld.
9. **Resume check.** A run interrupted at step k and resumed gives bit-identical parameters at step 2k to an
   uninterrupted run (CPU, fixed thread count, deterministic algorithms on).

## 10. Resumability and records

- **One directory per run.** `runs/f5/<stage>/<arm>/t<seed>/`; each training seed of each arm is a separate,
  independently resumable run.
  - A completed run (`DONE` with the final test record) is never overwritten (rule 13); a re-run needs
    `--overwrite`.
  - A partial run resumes from its latest checkpoint.
- **Checkpoints.** Every 25 optimiser steps and at every evaluation. A checkpoint holds:
  - parameters and optimiser state;
  - the torch, NumPy and Python RNG states;
  - the data-order position and the DAgger round and data-set index;
  - the step counter;
  - the hashes of the configuration, graph, split and records.

  Each checkpoint is written atomically (temporary file + rename). The last 3 are kept, plus the best by validation
  loss.
- **Run log.** JSON lines per evaluation: step, trials seen, validation loss, sign accuracy, R2 and wall time. Every
  record carries the stage, arm, training seed, source labels, latencies and origin classes.
- **Test evaluation.** One script, run only on the selected checkpoint. It writes `test.json` and refuses to run
  twice.
- **Reports.** `docs/results/f5_<stage>.{json,md}`, generated from the run records; every number is checked by
  `scripts/verify_report.py` (rule 11).
- **Licence.** Trained FlyWire parameters are derived from FlyWire (CC BY-NC 4.0), and trained BANC parameters from
  BANC (CC BY 4.0) (rule 12). Both stay under `runs/` and are not committed.

## 11. Compute plan (estimates, not benchmarks)

There is no code yet, so these are operation counts, not measurements. A wall-time-only benchmark on a non-registered
seed runs in the implementation PR, as before prereg/12. The machine is an 8-core Apple M1 Pro (aarch64 Linux) with
15 GB RAM and no CUDA GPU. Training runs on the CPU in float32.

| stage | per optimiser step | budget per run | runs (arms x seeds) | rough serial wall time |
|---|---|---|---|---|
| A | about 3.0 M edges x batch 16 x 300 intervals, forward + backward: tens of seconds | 2,000 steps | 3 graph arms x 10 + `bridge_off` 5 (+ cheap `mlp`, `linear`) | weeks |
| B | about 18.1 M edges, batch 8: minutes | 1,000 steps | as A | months |
| C | as B per step, plus online FlyVis rollouts (about 23 ms per interval) | BC 1,000 + 5 x 500 steps | as A | months |

- **Memory.** Stage B/C BPTT over 300-500 intervals for 164k neurons at batch 8 needs a few GB of stored states.
  Gradient checkpointing per 50 intervals keeps it inside 15 GB.
- **The owner said run time does not matter.** The design therefore does not shrink the problem. If the projection
  is too long, the pre-registered fallback (prereg/14 section 10) cuts seeds, never steps or controls, in a fixed
  order.
- **Optional GPU.** A rented GPU (docs/05 section 7) would cut this by an order of magnitude. It is the owner's
  decision. Determinism and the resume check must hold on that device as well.

## 12. New constants (rule 2; registered in `configs/assumptions.json` under `f5.*` in the implementation PR)

| constant | value | origin |
|---|---|---|
| G_MAX (edge gain bound), gain init | 2, 1 | E |
| LAM_MIN, LAM_MAX | 0.05, 0.95 | E (flyhard) |
| lam_0 | 0.25 = 5 ms / 20 ms | P (Shiu membrane time constant) |
| updates per 5 ms interval | 1 | E |
| R_IN = R_OUT | 100 Hz | E |
| A_MAX | 12.5 deg | E (v1 range, docs/08) |
| V_MAX, K_P | 120 deg/s, A_MAX / V_MAX | E |
| drum speeds (train / held out) | 60, 120 / 30 deg/s | E |
| grating phase rule | 0 for seeds 0-39, else uniform from default_rng(seed) | E |
| boundary input noise | Poisson s.d. of a 5 ms count | E |
| stage C plant noise | OU yaw rate, s.d. 6 deg/s, tau 50 ms | E |
| MLP / linear input filters | 10, 25, 50, 100 ms | E |
| bypass projection size, ridge alpha grid | 2,048; 10^-3 ... 10^3 (7 values, inner CV) | E |
| optimiser, batch, steps, learning-rate grid, gradient clip | prereg/14 section 6 | E |
| stability thresholds | 10 %, 0.5 x drum speed, 30 deg/s | E |
| G1, Lindsay signs, bridge window and latency, yaw trim, plant | unchanged (docs/08) | as registered |

## 13. Risks and known limits

1. **Compute.** Stages B and C are long on this machine (section 11). Mitigation: resumable runs and a seed-only
   fallback.
2. **Ceiling.** Optomotor yaw may be easy enough that every arm learns it. F3b-3 already showed that the untrained
   VNC passes direction information largely independent of wiring. The ceiling guard reports this honestly. A
   harder task (obstacle, landing) is the next F5 step, not part of this registration.
3. **The rate model is not the spiking model.** Results are about the rate abstraction on the connectome's topology
   and signs. The `untrained` arm reports how far its open-loop responses are from the spiking hybrid.
4. **Inputs.** Stage A trains on the brain's real (noisy, partly lateralised) DN signal. Its quality bounds A,
   whatever the wiring.
5. **The readout is fixed and literature-based.** Training cannot change which MNs steer or with what sign. A wrong
   Lindsay sign would have to be compensated inside the network, and that is reported.
6. **One task, one body model.** The yaw plant is docs/08's one-degree-of-freedom model, not flybody. "flybody
   scale" refers to the size of the training (parameters, demonstrations), not to the body.
7. **Seeds.** The deterministic `true` initialisation means its seed variance is smaller than a shuffle's. The paired
   test uses the same data order for both arms of a seed.

## 14. Implementation order (each step its own PR, after this one is merged)

1. Rate engine (`engine/rate_torch.py`) with the sparse custom backward, plus checks 3, 4, 6 and 9 of section 9, plus
   the wall-time benchmark.
2. Graph builders and shuffles (`train/graphs.py`), plus check 5.
3. New brain runs for A (extended F3b-3 brain script), plus check 7, then the 1,760 runs.
4. Stage A: data loader with the sealed split, teacher, trainer, evaluator, report and `verify_report` extension.
   Bypass gate first.
5. Stage B, then stage C (closed-loop runner on the docs/08 loop, BC, DAgger), each with its report.
