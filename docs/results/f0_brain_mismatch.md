# F0 brain-level mismatch: bisection (2026-10-12)

Question: the F0 flight (seed 3) differed from v1 at brain level — MN9 18 Hz vs 53 Hz after landing, feeding in 64 %
of landed steps vs 100 %, network 10.8 Hz vs 7.4 Hz, 2.4 Hz after the input cut vs ~0 — while the trajectory matched.
Nothing was tuned; no engine code behaviour was changed (the only additions are the named variant
`poisson_delay_v1` and diagnostics under `scripts/diag/`). Raw results: `runs/bisect/`, `docs/results/f0_seeds.json`.

## Result in one paragraph
**Root cause: a stochastic ignition of the antennal-lobe / mushroom-body self-sustaining mode in the seed-3
realisation of the new engine; not a port bug and not M4.** With seeds that do not ignite (4 and 0) the new stack
reproduces v1 at brain level (MN9 54 / 55 Hz vs 53, feeding 100 % / 99 %, network 7.4 Hz vs 7.4, 0 Hz after the cut)
and, for seed 4, the trajectory is identical to v1 (0.000 mm). The ignition frequency of the new engine is not
statistically distinguishable from v1's, but a small difference cannot be excluded (below).

## a. Open loop, no body (sugar GRNs 100 Hz for 1 s, then cut; 3 seeds)
| engine | MN9 Hz | DN Hz | network Hz | network Hz, last 100 ms of the cut | first cut step Hz |
|---|---|---|---|---|---|
| v1 (read-only scratch copy) | 55.3 | 0.93 | 0.093 | 0.0 | 0.014 |
| new (Poisson without delay, M4) | 58.5 | 0.97 | 0.093 | 0.0 | 0.058 |
| new, `poisson_delay_v1` | 58.3 | 0.97 | 0.093 | 0.0 | 0.064 |

Seed spread of v1 MN9: 50-58 Hz. Open loop matches; M4 has no effect here.

## b. Coupling mode and interval (full brain, 1 s open loop, same seed)
Per-interval `advance` and `run_coupled` at 25 ms, and `run_coupled` at 5 ms and `advance` at 5 ms: **identical
spike trains** (12,567 spikes each; wall 7.5 s / 2.7 s / 2.7 s / 25.5 s).

## c. Closed-loop F0 input stream, perch -> cut -> flight
* Built networks fingerprinted (v1 FlightBrain vs new engine): synapse lists, weights, rfc, v, g, Poisson wiring
  and targets are **hash-identical**; the only difference is the Poisson delay (M4, 0 vs 1.8 ms).
* Inputs logged interval by interval: the applied rates are zero in every cut interval, and the engine's Poisson
  groups hold zero during every cut interval (read back before each set: the value left by the previous
  boundary); no input is applied twice, vision does not continue into the cut, cut starts at 250 ms (interval 10) as in v1.
* The vision drive equals v1's exactly (`t45_rate_L/R` identical in all 300 steps; loom differs 0.8 % by the
  intended sub-frame-mean fix L1).
* Same recorded drive arrays (perch + cut) fed to **both** engines, 12 seeds: perch network rates of the delay
  variant equal v1's within 0.7 % at every interval (v1 / delay: 6.17/6.15, 10.97/11.05, 11.90/11.90 ... Hz); the
  M4 engine is 7 % / 4 % higher in the first two perch intervals (input arrives immediately instead of 1.8 ms
  later), and its first cut interval is lower (0.37 vs 0.87 Hz) because no delayed input is still in flight.
* Who fires in the ignited state (seed 3, after the cut): KCg-m, KCab, lLN1_bc, lLN2P, DA1 lPN, a few ORNs — the
  olfactory self-sustaining mode that v1 documented (criterion (a), flights with olfaction on show 1.5-3.4 Hz).

## Ignition frequency (open-loop identical drive, "ignited" = mean idle rate > 0.5 Hz)
| engine | seeds | ignited |
|---|---|---|
| v1 | 0-35 | 0 / 36 |
| new (M4) | 0-11 | 2 / 12 (seeds 3, 7) |
| new + `poisson_delay_v1` | 0-35 (seed 0-2 not run) | 2 / 33 (seeds 3, 7) |

v1's recorded flights without olfaction (seeds 3, 6-8, 10-14, 37 flights) all end the cut at 0.0 Hz. Fisher exact,
one-sided, delay variant vs v1 (36): p = 0.23; vs all 73 v1 realisations: p = 0.10. The same seeds ignite with and
without the delay (same noise stream); v1 does not ignite for those seed numbers, i.e. the engines do not share
noise realisations even with identical wiring and delay. **Not shown to be a real difference; not excluded.**

## Checked and excluded
input applied twice; vision input continuing after the cut; cut timing; store/restore of the initial state (v1
with store/restore unchanged); rfc of Poisson targets; synapse order; the 5 ms vs 25 ms interval; run() per
interval vs run_coupled.

## Consequence for F0
The F0 acceptance flights should use a seed that does not ignite or report the ignition frequency alongside; the
seed-3 record (`runs/f0/f0_lat0_data.h5`) is kept as the ignited example. Whether the AL/KC mode ignites more often
in the new engine than in v1 needs ~100 more v1 realisations to resolve (each ~45 s); not run.
