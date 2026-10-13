# Latency test on a flight where the brain closes the loop: v1 `flight_v30` and control `flight_v31`

A **v1 reproduction only, not a new result.** Hover at (440, -170, 160) mm, yaw perturbation +30 deg at t = 0.5 s,
brain-only path (yaw command = hand-made bridge `turn_dn`, no HAND odour / FlyVis terms), sB vision, no olfaction,
seed 6, 60 steps of 25 ms. The v1 drive is DNp15, which fails the symmetry gate (R/L 1.66x): run with
`--waive DNp15` as a labelled exception (recorded in the run meta). The DNa02 version is a new experiment for F2
and was not run. Since 2026-10-13 `--waive` defaults to none (CLAUDE.md rule 5): the v1 reproduction
commands in this report need `--waive DNp15` explicitly; the records here were run with it. Code `tasks/hover_yaw.py`; records `runs/hover_yaw/`; numbers `docs/results/f0_latency_hover_yaw.json`.
Latency 0 = v1 same-interval coupling (labelled ACAUSAL), latency 1 = causal.

## v30 (steering on)
| | total heading change | peak turn_dn | heading diff to v1 (max / rms) |
|---|---|---|---|
| v1 | -54.9 deg | 0.958 | - |
| new, latency 0, seed 6 | -59.7 deg | 0.957 | 24.7 / 13.9 deg |
| new, latency 1, seed 6 | -30.3 deg | 0.845 | 31.3 / 13.3 deg |

Latency 0 vs 1 (seed 6): heading differs by up to 27.9 deg (rms 9.8, final 27.9); the causal flight turns about half
as far as the same-interval flight.

**Context (not requested; run to read the numbers above against noise): seeds 0 and 1 at both latencies.**
| seed | latency 0 | latency 1 | lat 0 vs 1 heading diff (max / rms) | lat 0 vs v1 (max / rms) | lat 1 vs v1 (max / rms) |
|---|---|---|---|---|---|
| 0 | -50.5 deg | -42.7 deg | 16.2 / 7.8 | 21.4 / 8.1 | 23.7 / 10.7 |
| 1 | -30.0 deg | -32.3 deg | 15.5 / 7.4 | 38.8 / 19.8 | 31.5 / 16.6 |
| 6 | -59.7 deg | -30.3 deg | 27.9 / 9.8 | 24.7 / 13.9 | 31.3 / 13.3 |

Seed-to-seed difference at the same latency is as large as the latency effect: latency 0, seeds 0 vs 1: up to
31.4 deg (rms 15.8); latency 1: up to 18.3 deg (rms 8.3). Mean total heading change over the three seeds: latency 0
-46.7 deg, latency 1 -35.1 deg, v1 -54.9 deg (one realisation).

## v31 (steering readout ablated: replaced by its perch baseline)
| | total heading change | turn_dn (constant) |
|---|---|---|
| v1 | +5.6 deg | 0.058 |
| new, latency 0 | +123.7 deg | 0.222 |
| new, latency 1 | +122.8 deg | 0.222 |

Latency 0 vs 1: 1.6 deg (the ablated readout is constant, so there is nothing for latency to delay).
The control does not reproduce v1 because its constant is the perch baseline, which is quantised: DNp15 is one neuron
per side and the baseline is 10 intervals, so one spike is 4 Hz. Baseline DNp15 L/R: v1 8 / 52 Hz (s_hat -0.058),
new 12 / 44 Hz (s_hat +0.222). A difference of one to two spikes between realisations moves the "brainless" constant
by 0.28 and changes the open-loop turn by ~120 deg in 1.5 s. v1 reported the same property (final_v2a: the
ablation constant was +0.244, "not brainless"); it is a property of this control design, not of the port.

## Reading
- The brain does close the loop in this flight (peak turn_dn ~1, a 30-55 deg turn), so unlike n1 this is a place
  where latency can matter; with three seeds the latency effect (mean ~12 deg of total heading change, 15-28 deg
  max difference) is **not separable from seed-to-seed variability**. No causal-latency effect is demonstrated, and
  none is excluded.
- Against v1 the new stack gives the same peak command and a turn of the same size at latency 0 (-59.7 vs -54.9 deg
  for seed 6), but trajectories are realisation-dependent (diff to v1 up to 39 deg), as expected for a flight whose
  command is a few DNp15 spikes per interval.
- v31 should not be used as a reproduction target without fixing the baseline realisation; a better control for
  F2 is a constant zero command (a pre-registration decision for F2, not made here).

## Finding about v1: its steering-ablated control is fragile (decision of the project owner, 2026-10-12)
The v1 control "ablate the steer readout" replaces the readout by its perch baseline. With one DNp15 neuron per side
and a 10-interval baseline (one spike = 4 Hz) the constant it produces depends on the noise realisation of the
perch: DNp15 L/R baseline 12/44 Hz (this port, seed 6) vs 8/52 Hz (v1, seed 6) shifts the constant by 0.28 and
changes the heading by ~120 deg in 1.5 s (+123.7 deg vs +5.6 deg). A control whose effect size equals the effect
under test is not a control. For all new experiments the control arm is a constant zero command, named
**`zero_turn`** (yaw command exactly 0, readouts still recorded). The perch-baseline ablation is kept only to
reproduce v1.

## Decisions on further runs
No further v30 seeds: the latency effect is measured in F2 with DNa02 as the drive and >= 10 seeds per latency,
pre-registered (planned item in docs/05, F2).
