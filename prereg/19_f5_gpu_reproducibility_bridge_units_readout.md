# Pre-registration 19 — addendum to prereg/14: GPU reproducibility, bridge units, mlp / linear output

Status: committed **before any F5 training run and before any F5 gate run** (CLAUDE.md rule 6). It is not edited
afterwards; additions go in a new file and are labelled post-hoc. **prereg/14, prereg/17 and prereg/18 are not
changed**; this file fixes three points that PR #27 found open or inconsistent, and nothing else. Decision context: the
project owner's decisions after PR #27 (2026-10-18).

## 0. What was seen before this file was written

- PR #27 (`docs/results/f5_stageA_runner.md`): the stage-A runner and its unit tests on synthetic data.
- Three smoke runs (true arm, training seed 0, learning rate 1e-3, 20 steps on the GPU, `runs/f5/A_smoke`), run for
  time and memory only. Their evaluations used training trials, not validation trials. The evaluation values were
  written to the smoke logs and the first run printed them; no decision here uses them.
- The diagnostic `scripts/diag/f5_spmm_determinism.py` (synthetic inputs, no metric).
- No F5 training run exists, no gate of prereg/14 section 9 has run, and no validation or test record has been read by
  any F5 training code.

## 1. GPU training, reproducibility, and the test evaluation on the CPU

**Finding (PR #27).** On the full stage-A VNC (25,818 neurons, 3,020,333 edges) the CSR products of torch / cuSPARSE on
the GPU are not bit-reproducible, even with `torch.use_deterministic_algorithms(True)`:
- the forward product W s: largest difference between identical calls about 1e-7;
- the transposed product W^T s (state gradient): about 3e-6;
- the sddmm edge gradient (prereg/18) is bit-reproducible, and every product is bit-reproducible on the CPU.

Two uninterrupted GPU smoke runs differed at step 10 by 7.6e-9 (theta) and 2.4e-7 (phi). A paused and resumed GPU run
differed from an uninterrupted one by the same order (1.0e-8 and 3.6e-7 at step 20). The prereg/18 statement that a GPU
resume with sddmm is bit-identical rests on PR #25 tests with small synthetic graphs, where cuSPARSE happens to be
reproducible; on the full graph it does not hold.

**Decision (option a).**
- **Training runs on the GPU** with the registered products (torch / cuSPARSE CSR products, sddmm edge gradient,
  deterministic algorithms on). Runs, and a resumed run, agree only to float32 rounding. This is accepted and recorded
  in every run record. Resume still restores the parameters, the optimiser state, every RNG state and the data
  position exactly. The validation evaluations during training run on the GPU, so the checkpoint selection (prereg/14
  section 6) uses GPU validation losses.
- **Measurement of the run-to-run difference.** In each stage, the `true` arm at training seed 0 with the selected
  learning rate is trained a second time: the same configuration, in its own run directory, marked `replicate`.
  Reported (descriptive, no criterion):
  - the difference of the validation loss, turn-sign accuracy and R2 at every evaluation, and at each run's selected
    checkpoint;
  - the two selected steps;
  - the largest absolute parameter difference at the end of training.

  The second run is never used for the learning-rate choice, the checkpoint selection, a test evaluation, a metric
  of section 8 or a verdict. The seed-0 replicate of prereg/14 section 6 remains the first run. Its cost is counted in
  each stage's section-10 projection.
- **Final test evaluation on the CPU.** The test evaluation (prereg/14 sections 7-8) of every run loads the selected
  checkpoint (`best.pt`) and runs on the CPU, with deterministic algorithms on and a fixed thread count that is
  recorded. The CPU products are bit-reproducible, so every reported test number is exactly reproducible from the saved
  checkpoint. The test set is still read once per run.
- prereg/14 gate 3 (resume) is checked on the CPU, as docs/10 section 9 item 9 states, where it must be bit-identical.

## 2. Bridge units: 225 (replaces "224 units" in prereg/14)

prereg/14 section 2 describes the DN bridge as "the prereg/11 `ensemble` (112 families, 224 units, 305 neurons)".
prereg/10 (section 4) and prereg/11 (section 3) register the `ensemble` with **225 units** (221 matched by exact type,
4 by family; family DNg18 has an extra left unit matched by type), 305 BANC targets. Each derangement `random_k` has
224 units.
- The bridge is the prereg/11 `ensemble`, unchanged: **225 units**, 305 BANC targets. "224 units" in prereg/14 is
  read as 225 wherever it refers to the `ensemble`:
  - section 2 (DN bridge);
  - section 9, gate 1 ("Ridge on the window-mean 224 bridge units": 225 units).
- The `mlp` and `linear` input (stage A) is the 225 units of the `true` bridge, each through the four exponential
  filters: 900 features. This is also what the PR #25 benchmark used (`mlp` width 1,346, 3,047,360 parameters).
- `derange` keeps the prereg/11 derangement bridges unchanged (224 units each).

## 3. Output of `mlp` and `linear`

prereg/14 section 5 does not state whether `mlp` and `linear` apply a nonlinearity to their 16 outputs.
- The last layer of `mlp` and the single layer of `linear` are linear maps to the 16 readout MN states, with **no
  output nonlinearity**, so `linear` stays linear. The 16 states then go through the fixed readout of prereg/14
  section 2 (dA = G1 x R_OUT x sum_m sigma_m (s_{m,L} - s_{m,R})).
- The hidden layers of `mlp` keep their ReLU (prereg/14 section 5).

## 4. Unchanged

Everything else in prereg/14, prereg/17 and prereg/18 is unchanged: model, arms, budgets, learning-rate grid, data,
splits, metrics, decision rules, gates, the sddmm edge gradient and the checkpointing settings.
