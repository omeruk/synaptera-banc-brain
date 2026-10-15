# Pre-registration 16 — F5 step 2a-2: can the bounded rate VNC reach the target turn amplitude?

Status: committed **before any reachability run** (CLAUDE.md rule 6). It is not edited afterwards; additions go in a new
file and are labelled post-hoc. Decision context: the project owner's instruction after PR #21 (2026-10-15).
- PR #21 (`docs/results/f5_rate_lif.md`) showed that the untrained rate VNC steers with d of about 0.01 state.
- The fixed prereg/14 readout `dA = G1 x R_OUT x sum_m sigma_m (s_{m,L} - s_{m,R})` (G1 = 0.395 deg/Hz,
  R_OUT = 100 Hz) needs `sum_m sigma_m (s_{m,L} - s_{m,R}) = 12.5 / 39.5 = 0.316` for A_MAX = 12.5 deg.
- The edge gains are bounded at G_MAX = 2.

This file registers a measurement of whether 12.5 deg is reachable at all within the prereg/14 bounds. It is a
capacity check, not training. **prereg/14 is not changed by it**; whatever follows is the owner's decision.

## 0. What was seen before this file was written

- The PR #21 results (above).
- A wall-time-only test on the VNC graph, with random clamp values and no objective read: one Adam step at batch 8
  took about 24 s at 200 intervals and 36 s at 300 intervals (8 threads).
- The shuffle of section 3 was run once on the VNC graph structure for timing and checks only: 17 s, 30,203,330
  accepted swaps, all checks passed. No state was computed on it.
- No reachability number was computed before this file.

## 1. Quantity

- Readout as prereg/14: the 16 Lindsay `lateral` MNs, one neuron per type and side; sigma = +1 for b2 and -1 for b3, i1,
  i2, hg1, hg2, hg3, hg4; dA = 39.5 deg x sum_m sigma_m (s_{m,L} - s_{m,R}).
- **Directed amplitude of a trial.** D = mean over the trial's window of c x dA, where c is the intended turn
  direction of the trial (+1 right, -1 left; section 2).
- **Reach of a parameter set on an input set** = the mean of D over the evaluation trials of that set.
- **Reachable** if any procedure of section 4 gives a reach >= 12.5 deg on that input set. Also reported:
  - the smallest D over the trials;
  - the fraction of trials with D >= 12.5 deg;
  - the largest per-interval |dA|.
- **The search gives lower bounds.** It can show that 12.5 deg is reached; if it is not reached, that is "not reached
  by the registered search", not a proof that it is unreachable. The trivial bound |s| < 1, i.e. |dA| < 632 deg, is
  stated with the result.

## 2. Inputs (training-seed data only)

- **Common clamp.** DNg02 (both sides) clamped at 1 (`HAND`, 100 Hz) in every trial. Every DN not named below is
  clamped at 0. Each trial starts from s = 0, with one update per 5 ms interval.
- **I1, F3a-3 drives.** DNa04_L, DNa04_R, DNa15_L and DNa15_R (the `scripts/f3a2_vnc_run.py:groups_of` groups), each
  at 25 Hz and at 100 Hz, clamped to rate / 100 Hz for 200 intervals.
  - 8 trials. c = +1 for a right drive and -1 for a left drive (the ipsilateral direction steered by the LIF, F3a-3).
  - Window: intervals 100-199.
- **I2, recorded FlyWire DN rates at 60 deg/s.** The 80 records `rot_left` and `rot_right` of data seeds 0-39:
  `runs/f3b1/brain` (0-9), `runs/f3b3/brain` (10-29) and `runs/mv2/t3_R/brain` (30-39). All are training seeds of
  prereg/14.
  - Input path: the prereg/11 `ensemble` bridge (112 families, window 5 intervals, latency 1 interval), BANC DN
    state = unit rate / 100 Hz, as prereg/14 stage A.
  - 300 intervals. c = +1 for rot_right and -1 for rot_left (syndirectional).
  - Window: intervals 120-299.
  - These records are at 60 deg/s, where the prereg/14 target is 6.25 deg. The 12.5 deg target belongs to 120 deg/s,
    which has no record yet. I2 therefore asks whether the network can produce 12.5 deg from the brain signal that
    exists, and the 6.25 deg level is reported next to it.

## 3. Graphs

- **`true`.** The prereg/08 VNC subgraph with the prereg/01 signs. Every DN is a clamped input.
- **`shuffled`.** One degree-preserving shuffle of `true`, made by `train/graphs.py:degree_preserving_shuffle` with
  `default_rng(20261501)` (prereg/14, training seed t = 0) and 10 x E accepted swaps.
  - The input edge order is the `engine/rate_torch.py:RateGraph` order (sorted by post, then pre).
  - The shuffle check of prereg/14 gate 4 is reported.

## 4. Procedures (per graph and input set)

- **P0, untrained.** Gain 1, leak 0.25.
- **P1, all gains at the bound.** Every gain = G_MAX = 2 (the supremum of `2 sigmoid(theta)`), leak 0.25.
- **P2, gradient ascent within the bounds.**
  - Objective: maximise J = mean over trials and window intervals of c x dA / 12.5 deg, over theta and phi (all
    gains and leaks). The parameterisation keeps every value inside the prereg/14 bounds.
  - Optimiser: Adam (0.9 / 0.999, eps 1e-8), learning rate 0.05, no clipping, 200 steps from initialisation, float32,
    full BPTT. The learning rate is fixed here and is not searched.
  - Batch, I1: all 8 trials every step.
  - Batch, I2: 8 trials per step, drawn without replacement per epoch from the 80 by `default_rng(16)`.
  - Every 50 steps and at the end, the reach is evaluated on all trials of the set (I1: 8; I2: 80). The value reported
    is the last evaluation. Also reported: whether J still rose by more than 1 % over the last 50 steps ("not
    converged").
  - Wall time per gradient step is measured and reported.

## 5. mlp and linear arms

- **By construction.** Their 16 MN outputs are unbounded linear outputs (prereg/14 section 5: no tanh, no gain
  bound).
  - The linear arm's dA is linear in its weights.
  - The mlp's dA scales with its output-layer weights (ReLU hidden layers).
  - Any amplitude is therefore reachable by scaling. This is stated, not searched.
- **Numerical illustration on I2.**
  - Features: the 224 bridge-unit rates / 100 Hz through the four prereg/14 exponential filters (10, 25, 50,
    100 ms; a causal first-order filter per interval), plus a constant.
  - Fit: one least-squares map from the features to the directed target of +-12.5 deg over the window intervals of
    the 80 I2 trials.
  - Reported: its in-sample reach, the same three extra statistics, and the reach at 2x and 10x the fitted weights.

## 6. Output and reading

- Records: `runs/f5/reach/` (not committed). Report: `docs/results/f5_reachability.{json,md}`, checked by
  `scripts/verify_report.py`.
- Each part runs under `nohup` and is resumable: checkpoints every 25 steps, and a finished part is skipped.
- If 12.5 deg is not reached by `true` on I2, the report lists options (for example one trainable readout scale shared
  by all arms, a larger G_MAX, input scaling) with pros and cons. No option is applied here.
