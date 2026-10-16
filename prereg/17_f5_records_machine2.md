# Pre-registration 17 — addendum to prereg/14: F5 stage-A records on machine 2

Status: committed **before any F5 stage-A record run and before the check of section 4** (CLAUDE.md rule 6). It is not
edited afterwards; additions go in a new file and are labelled post-hoc. **prereg/14 is not changed**; this file
replaces the two prereg/14 passages named in sections 2 and 3, and nothing else. Decision context: the project owner's
decision after PR #23 (2026-10-17).

## 0. What was seen before this file was written

- PR #23 (`docs/results/machine_x86_repro.md`, `docs/05_ARCHITECTURE_PLAN.md` section 7.1). Rerun on machine 2,
  `runs/f3b3/brain/rot_left_s10.npz` is not bit-identical to the M1 record:
  - the float32 FlyVis render differs from the first interval;
  - the spike counts differ from interval 180 on.
- No F5 stage-A record exists. `runs/f5` holds only the step 2a and 2a-2 records of prereg/15 and prereg/16.
- No record has been generated twice on machine 2. Whether machine 2 reproduces its own records is unknown.

## 1. Machines

- **M1:** Apple M1 Pro, aarch64, Ubuntu 24.04 (Asahi), no CUDA. It made every record up to PR #22.
- **Machine 2:** AMD Ryzen 9 8945HS, x86_64, NVIDIA RTX 4070 Laptop, WSL2 Ubuntu 24.04.
  - Environment: `env/neurofly_x86_64.yml` + `env/neurofly_x86_64_nodeps.txt`.
  - FlyVis runs on the CPU (`sense.vision_flyvis.import_flyvis`).

## 2. Stage A records (replaces "Stage A records" in prereg/14 section 7)

- **All 2,400 stage-A brain records are generated on machine 2**, with the protocol of prereg/14 section 7: the
  unmodified spiking brain R, the F3b-3 protocol plus the speed and phase options.
  - The 160 records that prereg/14 reused from the M1 are generated anew on machine 2: data seeds 0-39 x rot_left,
    rot_right, static and grey at 60 deg/s, phase 0.
  - The other 2,240 records are unchanged from prereg/14:
    - seeds 40-239 x rot_left / rot_right at 30, 60 and 120 deg/s, static and grey (1,600);
    - seeds 0-39 x rot_left / rot_right at 30 and 120 deg/s (160);
    - seeds 0-239 x rot_left / rot_right at 90 deg/s (480).
  - The seeds, conditions, splits and every other part of prereg/14 are unchanged.
- **The M1 records stay unchanged** as the evidence of the F3b results: `runs/f3b1/brain`, `runs/f3b3/brain` and
  `runs/mv2/t3_R/brain`. They are not used in F5. No F5 training, selection, test or gate reads them.
- Every stage-A run directory records the machine it ran on (platform, CPU, GPU, environment file SHA-256). A record
  from any other machine is not used in F5.

## 3. Gate 5 (replaces the second item of prereg/14 section 9, gate 5)

prereg/14 reads: "The extended brain script reproduces `runs/f3b3/brain/rot_left_s10.npz` bit for bit at 60 deg/s and
phase 0." This becomes:

- **A machine-2 record reproduces bit for bit on machine 2.**
  - The extended brain script runs rot_left, brain seed 10, at 60 deg/s and phase 0 on machine 2.
  - Its record must be bit-identical to the machine-2 record of the same run made by `scripts/f3b3_brain.py`.
  - The gate is passed only if the section 4 check has passed.
- **Bit-identical** (for this file and its gate) means:
  - the two `.npz` files hold the same members;
  - each member's uncompressed `.npy` bytes (header, dtype, shape and data) are identical.

  The `.npz` file hashes are reported but are not the criterion: `np.savez_compressed` writes the write time into each
  zip member header.
- **The first item of gate 5** ("The reused records match their recorded hashes") is unchanged. No M1 record is reused,
  so it applies to no M1 record.
- **The third item of gate 5** (prereg/09 input check) is unchanged.
- A failure is a bug, as in prereg/14: it is fixed and the check is re-run before any training.

## 4. Check of this addendum (before any stage-A run)

- **Runs.** `scripts/f3b3_brain.py`, unchanged, rot_left brain seed 10, run twice on machine 2. Each run goes into its
  own new scratch folder outside `runs/` and is stopped once `rot_left_s10.npz` is written. Each run renders fresh. No
  other brain record is generated.
- **Criterion.** The two `rot_left_s10.npz` files are bit-identical (section 3).
- **Also reported (descriptive, no criterion):**
  - whether the two renders (`stim_rates.npz`) are bit-identical;
  - the comparison of both records with `runs/machine_x86/f3b3_rot_left_s10/brain/rot_left_s10.npz`, made on machine
    2 for PR #23.
- **Decision rule.**
  - If the criterion holds, machine 2 reproduces its own records, and F5 stage-A records may be generated on it under
    section 2.
  - If it does not, no F5 stage-A record is generated. Every differing member and the size of each difference are
    reported, and the next step is the project owner's decision.
