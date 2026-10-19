# Pre-registration 20 — addendum to prereg/14: stage-A limitations, an exploratory follow-up, and hardware

Status: committed **before any stage-A test evaluation** (CLAUDE.md rule 6). It is not edited afterwards; additions go
in a new file and are labelled post-hoc. **prereg/14 to prereg/19 are not changed**: not the primary test, the Holm
family, the verdicts or the guards. Decision context: the project owner's decisions after PR #29 (2026-10-20).

## 0. What was seen before this file was written

- Only the stage-A validation report (PR #29, `docs/results/f5_stageA_validation.{json,md}`): validation evaluations
  written during training (data seeds 160-199, training speeds), and the guard log.
- No test seed (200-239) and no held-out speed (30, 90 deg/s) has been read. `final-eval` has not run; no run directory
  has `test.json` or `test.lock`.
- For section 2, the measured wall times of the stage-A runs (`scripts/f5_prereg20_estimate.py`); no metric.

## 1. Limitations recorded before unsealing

These are recorded now so that they are on file before any test number exists. They qualify how the stage-A results
are read; they change no metric, test, verdict or guard of prereg/14 section 8.

a. **Budget and learning rate.** `true` and `shuffled` both selected the largest rate of the grid, 0.01, and the
   validation loss of `shuffled` was still falling at step 2,000. A `true` minus `shuffled` difference at the
   registered budget may reflect learning speed rather than a difference that would remain with more training or a
   larger rate.
b. **Turn-sign accuracy is at ceiling.** The `untrained` arm already has validation turn-sign accuracy 0.994, and every
   trained arm except `bridge_off` reaches 1.000. Samples to set accuracy (0.90) is mostly reached at the first
   evaluation, so it is limited by the 50-step evaluation interval (800 training trials) and separates the arms
   little.
c. **`linear` is unstable at every rate.** It diverges at 0.01. At the selected 0.001 its validation loss swings
   widely between evaluations, so its final-step values, and the final-loss rate choice, are noisy. Its selected
   checkpoints are not.
d. **The `mlp` rate choice is a near tie.** 0.003 has a lower final validation loss than 0.001 by a small margin, and
   the `mlp` final-step values are noisy.

Every stage-A report states these four limitations next to the section-8 results.

## 2. Exploratory follow-up (never part of any verdict)

Labelled **exploratory** in every file, run record, figure and report. It never enters the primary test, the Holm
family, a verdict, a guard, the learning-rate choice or the checkpoint selection of the registered runs.

- **Arms:** `true` and `shuffled`. **Training seeds:** 0-4.
- **Configurations:**
  - (i) learning rate 0.03, 2,000 steps;
  - (ii) learning rate 0.01, 6,000 steps.
- **Everything else as registered** (prereg/14 sections 2, 4-7 with prereg/17-19): model, bridge, teacher, records,
  splits, batch 16, Adam, clipping, data order `default_rng(1000 + t)` (continued over as many epochs as the steps
  need), validation every 50 steps, checkpoints every 25 steps, and the checkpoint with the lowest validation loss
  selected. No rate is selected: each configuration runs at its stated rate.
- **Hardware.** Not machine 2. It runs on EVREN or on a rented GPU, after the prereg/14 section 9 checks pass on that
  hardware (prereg/14 section 10: "A move to other hardware is allowed if the section 9 checks pass on it."). The
  stage-A records are the machine-2 records of prereg/17, copied with their manifest and checked against it (gate 5);
  no record is generated anew. GPU results follow prereg/19 section 1: agreement to float32 rounding; the test
  evaluation on the CPU from `best.pt`.
- **Test evaluation.** Once per exploratory run, **only after** the primary stage-A `final-eval` of all registered runs
  has finished. Reported separately from the stage-A report, with the same metrics (prereg/14 section 8.1) and the
  same descriptive `true` minus `shuffled` comparison, without a verdict.
- **If the follow-up is never run,** every stage-A report says so.
- **Estimate** (`scripts/f5_prereg20_estimate.py`, `docs/results/f5_prereg20_estimate.json`): from the measured
  stage-A wall times of `true` and `shuffled` at 0.01 on machine 2 (RTX 4070 Laptop), median step 0.880 s, median
  validation evaluation 1.19 s, median data load 5.8 s. In machine-2 GPU hours:
  - (i) 10 runs, 0.50 h each: 5.0 GPU hours;
  - (ii) 10 runs, 1.51 h each: 15.1 GPU hours;
  - total 20.1 GPU hours, without heat pauses, setup, gates and the CPU test evaluation. Other hardware differs by an
    unmeasured factor.

## 3. The stage-A Holm-adjusted p while B and C are not run

prereg/14 section 8.2, quoted unchanged:

> - **Test.** Per stage X in {A, B, C}, d_t = M(`true`, t) - M(`shuffled`, t) for t = 0-9. One-sided exact paired
>   sign-flip permutation test (all 2^n sign patterns), H1: mean d > 0. alpha = 0.05.
> - **Multiplicity.** Holm over the three stages. A stage not run has p = 1.

So, with stages B and C not run, their p values are 1 and stage A has the smallest p of the three. Its Holm-adjusted p
is min(1, 3 p_A), and the verdict "The connectome helps in stage A" needs 3 p_A < 0.05. With 10 training seeds the
smallest exact p_A is 1/1024, so the smallest adjusted p is 3/1024. Nothing in this section changes prereg/14.

## 4. Hardware: machine 2 is no longer used for long GPU training

- Machine 2 (AMD Ryzen 9 8945HS, RTX 4070 Laptop, WSL2; prereg/17) is no longer used for long GPU training. During
  stage A its GPU reached the 87 °C pause threshold of the training guard repeatedly in the first run.
- Stages B and C move to other hardware under the prereg/14 hardware rule (section 10): the section 9 checks pass on
  that hardware before its first training run, and the move is recorded in the stage report.
- This concerns long GPU training only. It does not change where or how the registered stage-A test evaluation runs
  (prereg/19 section 1: on the CPU, from each run's `best.pt`).

## 5. Unchanged

Everything in prereg/14 to prereg/19 is unchanged: hypotheses, model, arms, budgets, learning-rate grid, data, splits,
metrics, primary test, Holm family, verdicts, guards, gates, the sddmm edge gradient, checkpointing and the GPU
reproducibility rule.
