# Ignition frequency, v1 vs new engine (prereg/02_ignition_frequency.md)

Protocol exactly as registered (commit `8f9727d`, runner `scripts/ignition_test.py`, analysis
`scripts/ignition_analysis.py`): recorded F0 perch + cut drive, seeds 100-199 per arm, ignition = network rate > 1 Hz
in the last cut step.

| arm | ignited / 100 | proportion | 95 % CI (Clopper-Pearson) | secondary (idle mean > 0.5 Hz) | seeds |
|---|---|---|---|---|---|
| v1 | 0 | 0.00 | 0.000-0.036 | 0 | [] |
| new | 0 | 0.00 | 0.000-0.036 | 1 | [] |
| new_delay | 0 | 0.00 | 0.000-0.036 | 1 | [] |

Fisher exact, two-sided: v1 vs new (primary) p = 1.0000; new_delay vs v1 p = 1.0000; new_delay vs new p = 1.0000.
Primary test at alpha 0.05: not significant.

## Reading
- **Primary result:** 0 / 100 ignitions in every arm; Fisher exact p = 1.0 (v1 vs new). The pre-registered
  decision rule gives: not significant; equality is *not demonstrated* — the 95 % upper bound is 3.6 % for each
  arm (the power note in the registration applies: a few percentage points are not detectable with 100 seeds).
  M4 stays as is; no engine change.
- **Consistency with the exploratory sweep:** seeds 3 and 7 ignited in 2 / 12 realisations of the new engine
  (docs/results/f0_brain_mismatch.md). Over the 100 registered seeds no realisation of either new arm met the
  primary criterion; together with the sweep the estimated ignition probability of the new engine is
  ~2 / 112 (1.8 %), v1 0 / 136.
- **Post-hoc observation (not part of the registered test):** seed 133 ignites in both new arms (3.3 -> 3.5 Hz
  from the first idle step) but its last cut step is 0.77 Hz (new) / 0.58 Hz (new_delay), below the registered
  1 Hz threshold, so the primary endpoint does not count it; the secondary (idle-mean) endpoint does. Whether the
  mode starts earlier or later in a realisation is a timing detail of the same mode. v1 has no such realisation.
  This is reported, not used to change the outcome.
