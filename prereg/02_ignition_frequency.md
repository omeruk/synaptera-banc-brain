# Pre-registration 02 — ignition frequency of the self-sustaining olfactory mode, v1 vs new engine

Status: committed before any run of this test (CLAUDE.md rule 6). Not to be edited afterwards; additions go in a
new file and are labelled post-hoc. Context: `docs/results/f0_brain_mismatch.md` (earlier exploratory sweep, seeds
0-35, p = 0.23; those seeds are not reused here).

## Question
Does the new engine (reference arm, review fix M4) ignite the antennal-lobe / mushroom-body self-sustaining mode
after the F0 input cut more often than the v1 engine under identical wiring and drive?

## Design
- **Drive:** the recorded F0 perch + cut drive, `runs/bisect/f0_drive.npz` (sha256
  `ca1c77ec7f05926b6b7a1b52da6e3693e913a9b47082396a9ffe042e59fee8e9`): 10 perch intervals of 25 ms with the
  recorded sugar / vbnd_L / vbnd_R rates, then 8 cut intervals with all inputs 0, then 3 idle intervals (0 input).
- **Ignition (primary endpoint):** mean network firing rate **> 1 Hz in the last cut step** (interval 17, the 8th
  cut interval). Secondary, reported only: mean rate over the 3 idle intervals > 0.5 Hz (the earlier exploratory
  definition).
- **Arms:**
  1. `v1`: Synaptera v1 `FlightBrain` from a read-only scratch copy of `vendor/synaptera_v1/flight/` (brain.py
     sha256 prefix `8ab3ac0ad4c74245`), the n1 input configuration (olfaction off, vision boundary on).
  2. `new`: `engine/lif_brian2.py` reference arm (Poisson input without delay, M4 fixed).
  3. `new_delay`: the same with the named variant `poisson_delay_v1`.
  Same graph (Connectivity_783 sha256 as recorded by the adapter), same input groups.
- **Seeds:** 100-199 per arm (100 per arm, 300 runs). Seeds 0-35 of the exploratory sweep are not reused. The seed
  is passed to Brian2 (`seed(s)` after the store/restore of the initial state); no seed is dropped or replaced
  after the result is seen; failed runs are re-run with the same seed, and counted if they cannot complete.
- **Software:** Brian2 2.9.0, cython target, this machine; the commit of this file fixes the engine code.

## Analysis
- **Primary test:** Fisher exact test, two-sided, alpha = 0.05, `v1` vs `new` on the primary endpoint.
- **Reported per arm:** ignited / 100, the proportion and its 95 % Clopper-Pearson confidence interval.
- **Also reported (descriptive, no claim of significance):** `new_delay` vs `v1`, `new_delay` vs `new`
  (Fisher, two-sided, uncorrected), and the secondary endpoint.
- **Power note (written before the data):** if the true ignition probability is ~6 % in the new arms and 0 % in v1
  (the exploratory estimate), 100 seeds per arm give roughly 80 % power; a difference of a few percentage points
  is not detectable with this design. A non-significant result is therefore *not* evidence of equality.

## Decision rule
- If the primary test is significant (p < 0.05): **report it and stop**. The engine is not changed in this
  branch; the difference is documented and the decision goes to the project owner.
- If it is not significant: report the three proportions with their intervals, state that equality is not
  demonstrated, and keep M4 as is.
- In neither case is any engine parameter, seed or protocol changed to move the outcome.
