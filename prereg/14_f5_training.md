# Pre-registration 14 — F5 training: optomotor yaw, curriculum A-C (D optional), connectome against controls

Status: committed **before any F5 code exists and before any F5 run** (CLAUDE.md rule 6). It is not edited afterwards;
additions go in a new file and are labelled post-hoc. Decision context: the project owner's record after PR #19
(2026-10-15, `docs/05_ARCHITECTURE_PLAN.md`). The design and its reasons are in `docs/10_TRAINING.md`; this file fixes
what is binding. Earlier pre-registrations are not changed.

## 0. What was looked at before this file

Only metadata of existing records:
- file lists, sizes and array shapes;
- `model_sha256` and `weights_sha256` of `runs/f3b1`, `runs/f3b3` and `runs/mv2/t3_R` (identical: fc45837d...,
  d5547fb8...);
- the wall times in their `DONE` files;
- the per-type neuron counts of `docs/results/side_audit.json`.

No rate model was built, no training ran, and no spike count was analysed for this file. All earlier results (F3b-1,
F3b-3, mv2, docs/08) are known and are the reason for this study; none of them enters a criterion here.

## 1. Hypotheses

- **H1 (primary, per stage A, B, C).** Trained with the same data, budget and test, the true connectome wiring reaches
  a higher primary metric than a degree-preserving shuffle of the same graphs.
- **H2 (descriptive).** Each arm learns the task (section 8.4).
- **Secondary comparisons (descriptive, section 8.3).**
  - `true` against `derange`, `mlp` and `linear`;
  - samples to set accuracy;
  - generalisation to the held-out drum speed.

## 2. Model (fixed; docs/10 section 2)

- **Dynamics.** `s <- (1 - lam) s + lam tanh(W_eff s + u)`, with `W_eff[i, j] = g_ij sign_j n_ij / N_i` (N_i = total
  input synapse count of i). There is one update per 5 ms interval, the state is carried within a trial or flight,
  and s = 0 at the start of each.
- **Trainable.** `g_ij = 2 sigmoid(theta_ij)` with theta init 0 (g = 1), and `lam_i = 0.05 + 0.90 sigmoid(phi_i)` with
  lam init 0.25. Nothing else is trained, and the topology is fixed.
- **Graphs and signs.**
  - VNC: the BANC v3 VNC subgraph of prereg/08 with the prereg/01 reference sign rule.
  - Brain (stages B, C): FlyWire v783 with the F0 reference sign rule.
- **Inputs.**
  - BANC DNs are clamped input nodes.
  - DN bridge: the prereg/11 `ensemble` (112 families, 224 units, 305 neurons), window 5 intervals, latency 1 interval,
    no Poisson re-synthesis.
    - Stage A: BANC DN state = recorded unit rate / 100 Hz.
    - Stages B, C: the window mean of relu(s) of the unit's FlyWire DNs.
  - DNg02 is clamped at 1 (`HAND`, 100 Hz); every other BANC DN is clamped at 0.
  - The visual boundary neurons get u = rate / 100 Hz.
- **Readout (fixed).** `dA = 0.395 deg/Hz x 100 Hz x sum_m sigma_m (s_{m,L} - s_{m,R})` over b2(+), b3, i1, i2, hg1,
  hg2, hg3, hg4 (-), one neuron per type and side.
- **Causal timing (rule 4).** The body receives dA_k at interval k + 1. Every latency is written into the
  configuration and each record.
- **Labels.** Trained arms' dA `TRAINED`; DNg02 baseline `HAND`; teacher commands `HAND` (teacher); visual input
  `FLYVIS`; haltere yaw damping `REFLEX`. Each is recorded separately.

## 3. Stages and order

| stage | trained part | input | task |
|---|---|---|---|
| A | rate BANC VNC subgraph | recorded FlyWire DN spike counts (spiking reference brain R) through the bridge | open-loop dA |
| B | rate FlyWire brain + rate VNC (VNC warm-started from A, same arm and training seed) | frozen FlyVis 1.2.0 -> boundary layer | open-loop dA |
| C | as B, warm-started from B | online frozen FlyVis in the docs/08 closed loop | behaviour cloning, then 5 DAgger rounds (beta = 0) |
| D | optional | - | needs its own pre-registration before any D run; never part of the section 8 verdict |

Each stage runs after its gates (section 9) pass. A stage whose gate fails is reported and not run. Later stages are
still run when their own gates pass, warm-starting from the last completed stage, or from initialisation if none.

## 4. Target and teacher (fixed)

`dA*(t) = clip(K_P v_slip(t), -12.5 deg, +12.5 deg)`, with `K_P = 12.5 / 120 deg per deg/s`, where v_slip is the
retinal slip speed (positive when the pattern moves to the fly's right).
- **Open loop (A, B).** v_slip = the drum speed during the stimulus (rot_right +, rot_left -). dA* = 0 under grey,
  under static, and in the grey lead-in.
- **Closed loop (C).** v_slip = drum speed - yaw rate. The teacher acts through the same one-interval command queue.
- **Plant and trim (C).** Plant, yaw damping (`REFLEX`) and static-grating yaw trim as in docs/08 sections 3 and
  10.7. Flight: 100 grey intervals + 400 drum intervals.

## 5. Arms (rule 7)

The same arms run in every stage. Training seeds are t = 0-9 for the trained arms and t = 0-4 for `bridge_off`.

| arm | definition |
|---|---|
| `true` | the graphs of section 2 |
| `shuffled` | each graph rewired by degree-preserving directed double-edge swaps (a->b, c->d => a->d, c->b; no self-loops, no duplicate pairs), 10 x E successful swaps. Synapse count travels with the edge, sign with the presynaptic neuron, and the swap RNG is `default_rng(20261501 + t)`. In B and C, brain and VNC are both shuffled. **Primary comparison arm.** |
| `derange` | `true`, with the bridge family mapping replaced by the prereg/11 derangement `random_(t+1)` (same generator and seed list; permutation SHA-256 as prereg/11) |
| `mlp` | two ReLU hidden layers of equal width h -> the 16 readout MN states -> fixed readout. h is the smallest width with a parameter count >= that of `true` in the same stage. The input is the arm input passed through exponential filters of 10, 25, 50 and 100 ms. |
| `linear` | one linear map from the same filtered input to the 16 MN states -> fixed readout |
| `bridge_off` | `true` with every bridged DN clamped to 0 (DNg02 kept); the ablation and leakage check |
| `untrained` | `true` at initialisation, evaluated on the test set without training (one evaluation; deterministic) |

## 6. Optimiser and budget (identical for every arm in a stage)

- **Optimiser.** Adam (beta 0.9 / 0.999, eps 1e-8), no weight decay, gradient clipping at global norm 1.0.
- **Loss.** The mean over every interval of a trial or flight of ((dA - dA*) / 12.5 deg)^2. In C, dA* is the
  teacher's label of the visited state. BPTT runs over the whole trial (300 intervals) or flight (500 intervals).
- **Budget per run:**

| stage | batch (trials or flights) | optimiser steps |
|---|---|---|
| A | 16 | 2,000 |
| B | 8 | 1,000 |
| C | 8 | BC 1,000, then 500 per DAgger round x 5 rounds |

- **Learning rate.** The grid is {1e-3, 3e-3, 1e-2}. Per (stage, arm), the rate with the lowest final validation loss
  at training seed 0 is used for seeds 1-9. The seed-0 run at that rate is the seed-0 replicate.
- **Data order.** Epoch-wise shuffle of the training data with `default_rng(1000 x stage_code + t)`, where
  stage_code is A 1, B 2, C 3.
- **Evaluation and checkpoints.** Every 50 optimiser steps on the full validation set.
  - A, B: the checkpoint with the lowest validation loss is selected.
  - C: after BC and after each DAgger round, the learner flies the 40 validation flights. The selected checkpoint has
    the lowest validation imitation loss over these 6 evaluations.
  - The selection never uses g, stability or any test record.

## 7. Data and splits (fixed)

- **Data seeds.**
  - Train: 0-159.
  - Validation: 160-199.
  - Test: 200-239. The test seeds are listed in a split file whose SHA-256 is recorded in every run; no training or
    selection code reads them.
- **Drum speeds.** Training and validation use 60 and 120 deg/s. **30 deg/s is held out** (never trained or used for
  selection). The wavelength is 30 deg and the temporal frequency is speed / 30 deg.
- **Grating phase.** 0 for data seeds 0-39; uniform over one wavelength from `default_rng(seed)` for seeds 40-239.
- **Conditions per data seed.** rot_left and rot_right at 30, 60 and 120 deg/s, static, grey (8).
- **Stage A records.**
  - **Reused:** `runs/f3b1/brain` (seeds 0-9), `runs/f3b3/brain` (10-29) and `runs/mv2/t3_R/brain` (30-39); 160
    records at 60 deg/s.
  - **New:** 1,760 runs of the unmodified spiking brain R with the F3b-3 protocol plus a speed and a phase option:
    seeds 40-239 x 8 conditions, and seeds 0-39 x rot_left / rot_right at 30 and 120 deg/s.
  - **Not used:** `runs/mv2/t3_ACG`.
- **Stage B.** FlyVis boundary sequences per (condition, phase), computed once. Per data seed, Gaussian noise with
  s.d. sqrt(rate / 5 ms) is added to every boundary rate, clipped at 0.
- **Stage C.**
  - Plant noise per flight: OU yaw-rate disturbance with s.d. 6 deg/s and tau 50 ms, seeded by (data seed, training
    seed).
  - BC: teacher flights on training data seeds 0-159 x the 6 training conditions (rot_left / rot_right at 60 and
    120 deg/s, static, grey): 960 flights, shared by every arm and seed.
  - Each DAgger round: 160 learner flights, one per training data seed, over the 6 training conditions in a balanced
    schedule from `default_rng(3000 + round)`.
  - Validation: 40 flights per evaluation (seeds 160-199, the same balanced rule).
  - Test: seeds 200-239 x 8 conditions.
- **Analysis window.** The stimulus intervals after the first 100 ms (A, B: intervals 120-299; C: flight intervals
  120-499, i.e. the drum after its first 100 ms).

## 8. Metrics and decision rules

### 8.1 Metrics (per arm and training seed, from the test set at the selected checkpoint)

| metric | definition | stage |
|---|---|---|
| **R2 (primary A, B)** | 1 - SSE / SST of the window-mean dA against dA* over all test trials at the training speeds (rot_left / rot_right at 60 and 120, static, grey) | A, B |
| turn-sign accuracy | fraction of test rotation trials at the training speeds whose window-mean dA has the sign of dA* | A, B |
| **g (primary C)** | per test data seed and training speed `(psi_dot(rot_right) - psi_dot(rot_left)) / (2 v_drum)` over the window, averaged over seeds and both speeds | C |
| stability | fraction of test flights with dA at +-12.5 deg in < 10 % of window intervals **and** yaw-rate s.d. over the last 200 intervals < 0.5 x drum speed (static, grey: < 30 deg/s) | C |
| samples to set accuracy | training trials (A, B) or flights (C) presented until the validation turn-sign accuracy first reaches 0.90; censored at the budget | A, B, C |
| generalisation | sign accuracy and R2 (A, B), or g (C), on the 30 deg/s test conditions | A, B, C |
| drift, imitation R2 | mean yaw rate under static and grey; interval-level R2 against the teacher label | C |

### 8.2 Primary test (H1)

- **Test.** Per stage X in {A, B, C}, d_t = M(`true`, t) - M(`shuffled`, t) for t = 0-9. One-sided exact paired
  sign-flip permutation test (all 2^n sign patterns), H1: mean d > 0. alpha = 0.05.
- **Multiplicity.** Holm over the three stages. A stage not run has p = 1.
- **Verdicts:**
  1. **"The connectome helps in stage X"** if its Holm-adjusted p < 0.05.
  2. **"No advantage in stage X"** otherwise. If the two-sided test in the opposite direction has p < 0.05, also
     "shuffled better" (descriptive).
  3. **Guards (they qualify the verdict; they do not change p):**
     - C only: if either arm has mean stability < 0.8, the verdict is "not interpretable (instability)".
     - Ceiling: if both arms have mean R2 >= 0.98 (A, B), or mean g >= 0.98 x the teacher's realised test g (C), the
       verdict adds "ceiling: the task does not separate the wirings".
     - Leakage: if `bridge_off` has a mean test sign accuracy outside the binomial 95 % interval around 0.5 for its
       number of test rotation trials, the stage is marked "leak" and its verdict is withheld.
- **Effect size.** Mean d with a 95 % bootstrap interval over training seeds (10,000 resamples, `default_rng(14)`).

### 8.3 Secondary (reported, same test, no Holm, never a verdict)

- `true` against `derange`, `mlp` and `linear` on M.
- `true` against `shuffled` on samples to set accuracy (censored values ranked as the largest) and on generalisation.

### 8.4 Learning (H2, descriptive)

- **A, B.** "Arm learned stage X" if, in at least 8 of 10 training seeds, test sign accuracy >= 0.9 and R2 > 0.
- **C.** "Arm learned stage C" if, in at least 8 of 10 training seeds, g > 0 and stability >= 0.8.
- **Reference.** The `untrained` arm, and in C the teacher, are reported as reference points.

## 9. Gates (before the first training run of each stage)

1. **Information alignment (rule 9).** Cross-validated R2 > 0 (5-fold by training data seed) of the trial-mean dA*.
   - Ridge (alpha chosen from 10^-3 ... 10^3, 7 values, by inner CV) on a fixed random Gaussian projection to 2,048
     dimensions (`default_rng(9)`) of the window-mean FlyVis output. Before A; it serves B too.
   - Ridge on the window-mean 224 bridge units. Before A.
   - Before C: the same FlyVis bypass on the BC teacher flights, with the interval-level teacher label.
   A failed gate stops that stage.
2. **Symmetry gate (rule 5).** |log2(R/L)| < 0.5 on input synapses for the 112 bridge families (both datasets) and the
   8 readout types (subgraph). Recomputed and reported. A failing readout type stops the stage.
3. **Engine checks.**
   - Sparse custom backward against dense float64 autograd on a 2,000-neuron subgraph: maximum relative error < 1e-6.
   - Forward against a NumPy reference over 300 intervals: maximum absolute difference < 1e-5 (float32).
   - Causality: an input perturbation at interval k changes dA no earlier than the registered total latency.
   - Resume: interrupt at step k, resume, and the parameters at step 2k are bit-identical to an uninterrupted run.
4. **Shuffle check.** Identical in- and out-degree vectors and per-presynaptic signs; no self-loops or duplicates.
5. **Records.**
   - The reused records match their recorded hashes.
   - The extended brain script reproduces `runs/f3b3/brain/rot_left_s10.npz` bit for bit at 60 deg/s and phase 0.
   - Every new record passes the prereg/09 input check.

A failure of gate 3, 4 or 5 is a bug. It is fixed and the check re-run before any training; fixing it changes nothing
in this file.

## 10. Compute fallback (fixed now, applied only before a stage's first training run)

- **Benchmark.** The implementation PR runs a wall-time-only benchmark (data seed 9999, not registered; no metric
  read). It projects the serial wall time of each stage.
- **Fallback.** If a stage's projection exceeds 60 days, its training seeds are reduced in this order, never its steps,
  arms, data or test:
  1. `derange`, `mlp` and `linear` to 7 seeds;
  2. then `true` and `shuffled` to 7 seeds (t = 0-6).
- **Why 7.** With 7 seeds the smallest exact p is 1/128, below the first Holm threshold 0.05 / 3. With 5 seeds
  (1/32) H1 could not be accepted, so 5 is not used for the primary pair.
- **Recording.** Whether the fallback was applied is recorded in the stage report, with the projection that
  triggered it. A move to other hardware is allowed if the section 9 checks pass on it.

## 11. Reported for every stage (whatever the outcome)

- All metrics of section 8.1 for every arm and seed.
- Learning curves (validation loss, sign accuracy, R2 every 50 steps; in C per round).
- The gate results; the `untrained` arm.
- The selected learning rates; the MLP widths and parameter counts.
- Wall time, peak RAM, and whether the fallback was applied.
- In C: the teacher's realised g and stability, and the per-round DAgger data size.
- Every number comes from the run records and is checked by `scripts/verify_report.py` (rule 11). Negative results
  are reported with equal weight (rule 10).

## 12. Origin classes (rule 2; registered under `f5.*` in `configs/assumptions.json` in the implementation PR)

- **P:** lam init 0.25 (5 ms / Shiu tau 20 ms).
- **E:**
  - gain bound 2 and init 1; leak bounds 0.05 / 0.95 (flyhard); 1 update per interval; R_IN = R_OUT = 100 Hz;
  - A_MAX 12.5 deg and V_MAX 120 deg/s (v1 range, docs/08); the drum speeds and the held-out speed; the phase rule;
  - boundary noise; plant noise (6 deg/s, 50 ms);
  - filter time constants; projection size and ridge grid;
  - optimiser, batch sizes, steps, learning-rate grid, clip; evaluation period; DAgger rounds and flights;
  - stability thresholds; the seed lists and RNG seeds of this file; the fallback rule.
- **Unchanged and already registered:** G1, the Lindsay signs, the bridge, the yaw plant and trim (docs/08), the sign
  rules (prereg/01), the subgraph (prereg/08), FlyVis and the boundary layer.
