# Pre-registration 18 — addendum to prereg/14: edge gradient and gradient checkpointing of F5 training

Status: committed **before any F5 training run** (CLAUDE.md rule 6). It is not edited afterwards; additions go in a new
file and are labelled post-hoc. **prereg/14 is not changed**; this file fixes two implementation settings that
prereg/14 leaves open, and nothing else. Decision context: the project owner's decision after PR #25 (2026-10-17).

## 0. What was seen before this file was written

- PR #25 (`docs/results/f5_benchmark.md`, `engine/rate_torch.py`, `tests/test_rate_torch*.py`). Wall time and memory
  only: data seed 9999, synthetic inputs, no metric read (prereg/14 section 10).
- No F5 training run exists. No F5 stage-A record exists.

## 1. Decision

- **Edge gradient.** Every F5 training run of every arm, in stages A, B and C, uses `edge_grad="sddmm"` of
  `engine/rate_torch.py`: the gradient of the per-edge weights by `torch.sparse.sampled_addmm`. The earlier default
  `edge_grad="chunked"` is not used in F5 training. Arms without a sparse graph (`mlp`, `linear`) have no edge gradient;
  for them nothing changes.
- **Gradient checkpointing.**
  - Stage A: none (`checkpoint_every=0`).
  - Stages B and C: every 50 intervals (`checkpoint_every=50`), in every run of every arm that trains through the rate
    graphs (`true`, `shuffled`, `derange`, `bridge_off`).
- Every run record carries `edge_grad` and `checkpoint_every` (the engine record already holds `edge_grad`).

## 2. Evidence (PR #25)

- **Equality.** The forward pass is the same code for both edge gradients and is bit-identical. The gradients agree
  within float32 tolerance (maximum relative error < 1e-5 over 300 intervals) on the CPU and on the GPU.
- **Gate-3 checks pass with sddmm.** Against dense float64 autograd the relative error is < 1e-6; `gradcheck` passes.
- **Reproducibility.** A GPU resume with sddmm is bit-identical (deterministic algorithms, CUDA RNG state in the
  checkpoint), and repeated sddmm runs in deterministic mode are bit-identical.
- **Not bit-identical to chunked.** The summation order differs, so sddmm and chunked gradients differ in the last
  float32 digits. This is why the change needs this addendum.
- **Cost** (RTX 4070 Laptop, `docs/results/f5_benchmark.md`):
  - stage A step: chunked 3.589 s, sddmm 0.953 s;
  - stage B step (no checkpointing): chunked 16.41 s, sddmm 4.81 s;
  - stage C step (checkpointing every 50): chunked 29.47 s, sddmm 10.37 s.
- **Memory.** On the 5.60 GiB per-process GPU cap, the stage-B step without checkpointing peaks at 5.52 GiB with sddmm.
  Checkpointing every 50 intervals lowers this to 2.47 GiB. The stage-C step does not fit without it. Checkpointing
  every 50 is therefore fixed for B and C, so that both stages run with one setting and a margin. Stage A peaks at
  1.32 GiB without it, so it needs none.

## 3. Checkpointing does not change results

In deterministic mode (`torch.use_deterministic_algorithms(True)`), gradient checkpointing changes no result.

- **What is compared:** states, losses, gradients and the parameters after optimiser steps, with and without
  checkpointing. They must be bit-identical.
- **Test:** `tests/test_prereg18_checkpointing.py`, committed with this file. It covers:
  - the stage-A engine and the stage-B/C composite;
  - the CPU and the GPU;
  - full and partial 50-interval segments.
- **Earlier tests:** `tests/test_rate_torch.py`, `tests/test_rate_torch_gpu.py` and `tests/test_rate_hybrid.py`.

A failure is a bug (prereg/14 section 9, gate 3). It is fixed and the check re-run before any training.

## 4. Unchanged

- Everything else in prereg/14 is unchanged: model, arms, budgets, learning-rate grid, data, splits, metrics, decision
  rules, gates.
- Gate 3 runs with the settings of section 1.
- **Section 10 fallback.** The projection is recomputed with the settings of section 1 before each stage's first
  training run. The PR #25 projection used the fastest setting per stage, which for stage B was no checkpointing.
