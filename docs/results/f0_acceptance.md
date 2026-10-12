# F0 acceptance flight (docs/06 section 3) — result

Run: `scripts/run_f0.py` (tasks/f0_flight.py), seed 3, 25 ms intervals, full brain (138,639 neurons), v1 n1
configuration. Numbers: `docs/results/f0_acceptance.json` (copied from runs/f0/*_report.json).

**Labels.** The replayed terms are labelled REPLAY in the record (`behavior/<field>.attrs["label"]`, meta
`replayed_terms`): turn_hand, turn_flyvis, thrust_hand, pitch_hand, roll_hand. They make this a regression test of
the port only, never a result.

**What was tested.** The v1 HAND `turn_hand` and FlyVis `turn_flyvis` terms are NeuroFly adaptations and are not
carried as code; their recorded per-step values (and the recorded collective / tilt targets) are replayed open-loop
from the v1 HDF5. Everything else is the new code. In n1 the brain does not enter the command (turn_brain = 0), so
the trajectory tests body / scaffold / vision / coupling; the brain contributes the feeding decision (S2).

| | latency 0 (v1 same-interval, labelled ACAUSAL) | latency 1 interval (causal) |
|---|---|---|
| S1 (no tower contact before touchdown) | pass | pass |
| S2 (feeding after touchdown) | pass | pass |
| max position difference to v1 | 0.27 mm | 0.27 mm |
| phases / wings_on identical to v1 | yes / yes | yes / yes |
| first feeding step (v1: 120) | 122 | 123 |

Latency 0 vs 1: position differs by at most 0.23 mm (only through the feeding-pose timing), feeding starts one
interval later. The trajectory is insensitive to latency here because the brain steering term is 0 in n1.

**Brain-level differences from v1 (not reproduced numerically; open).** Mean MN9 rate after landing 18 Hz (v1 53 Hz),
feeding in 64 % of landed steps (v1 100 %), mean network rate 10.8 Hz (v1 7.4 Hz), and 2.4 Hz network rate in the
last input-cut step (v1 ~0). The engine differs from v1 by review fix M4 (Poisson input without the 1.8 ms delay);
whether M4 alone explains these is not established.

## Update after the bisection (docs/results/f0_brain_mismatch.md)
The brain-level differences above belong to the seed-3 realisation, in which the olfactory self-sustaining mode
ignited after the input cut. With seeds 4 and 0 (latency 0): S1 pass, S2 pass; MN9 after landing 53.9 / 54.6 Hz (v1
53.1), feeding in 100 % / 99 % of landed steps (v1 100 %), network rate 7.41 / 7.43 Hz (v1 7.42), 0 Hz after the
cut, position difference to v1 0.000 mm (seed 4) and 0.037 mm (seed 0). Numbers: `docs/results/f0_seeds.json`.

## What the latency comparison does not test
n1 has `turn_brain = 0` (no brain steering), so latency 0 vs 1 changes the trajectory only through the feeding
pose; it does **not** test the brain's causal effect on flight.

## Latency test to run once the mismatch is resolved (not run)
v1 records where the brain enters the command (`turn_brain != 0`), from `~/Documents/NeuroFly/simulations/`:
* hover + yaw perturbation, brain-only path (no HAND / FLYVIS turn terms; `turn_brain` up to 0.9-1.1):
  `flight_v30` / `v32` (seed 6, yaw +30 / -30 deg), `v34` / `v36` (seed 7), `v38` / `v40` (seed 8); the matching
  steering-ablated controls are v31, v33, v35, v37, v39, v41. These drive the yaw with DNp15, which fails the
  symmetry gate (R/L 1.66x): reproducing them needs `--waive DNp15` as a labelled exception, or DNa02 as the drive
  (a new, pre-registered experiment).
* hybrid with brain steering: `flight_v11` (final_a, `turn_brain` up to 1.44; includes REPLAY HAND/FLYVIS terms).

## F0 result: multi-seed acceptance (seeds 0-4, latency 0 and 1; 10 flights)
This table is the F0 result (decision of the project owner: the seed-3 result is kept, labelled "ignited"; no seed
was chosen after seeing outcomes). `scripts/f0_table.py` -> `docs/results/f0_multiseed.{json,md}`. Latency 0 =
v1 same-interval coupling (labelled ACAUSAL), latency 1 = causal; 25 ms intervals; REPLAY terms as above (the seed-3
records predate the relabel, same code and inputs). Ignited = network rate > 1 Hz in the last cut step.

| seed | latency | S1 | S2 | MN9 after landing (Hz) | feeding fraction | network (Hz) | last cut step (Hz) | ignited | max pos diff to v1 (mm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | pass | pass | 53.9 | 0.99 | 7.43 | 0.000 | no | 0.037 |
| 0 | 1 | pass | pass | 55.3 | 0.99 | 7.43 | 0.000 | no | 0.076 |
| 1 | 0 | pass | pass | 54.5 | 1.00 | 7.43 | 0.000 | no | 0.000 |
| 1 | 1 | pass | pass | 50.8 | 0.99 | 7.43 | 0.000 | no | 0.037 |
| 2 | 0 | pass | pass | 52.2 | 0.99 | 7.42 | 0.000 | no | 0.004 |
| 2 | 1 | pass | pass | 50.8 | 0.99 | 7.42 | 0.000 | no | 0.037 |
| 3 | 0 | pass | pass | 18.5 | 0.64 | 10.83 | 2.420 | yes | 0.272 |
| 3 | 1 | pass | pass | 19.5 | 0.67 | 10.83 | 2.420 | yes | 0.270 |
| 4 | 0 | pass | pass | 53.1 | 1.00 | 7.41 | 0.000 | no | 0.000 |
| 4 | 1 | pass | pass | 50.0 | 0.99 | 7.42 | 0.000 | no | 0.037 |

- S1 and S2 pass in 10 / 10 flights; the trajectory is within 0.28 mm of v1 in all of them (0.000-0.076 mm in the
  four non-igniting seeds that reproduce v1's feeding pose exactly or nearly so).
- 4 of 5 seeds do not ignite and reproduce v1 at brain level (MN9 50-55 Hz vs 53, feeding 99-100 %, network
  7.4 Hz, 0 Hz after the cut); seed 3 ignites at both latencies (MN9 18-20 Hz, feeding 64-67 %, network 10.8 Hz).
- Latency 0 vs 1 changes MN9 by a few Hz and nothing in S1 / S2; as noted, n1 has `turn_brain = 0`, so this does
  not test the brain's causal effect (see `docs/results/f0_latency_hover_yaw.md`).
- Ignition frequency in the registered test (prereg/02): 0 / 100 in v1, new and new + poisson_delay_v1; see
  `docs/results/f0_ignition_frequency.md`.

## The self-sustaining olfactory mode was predicted structurally
`docs/02_CONNECTOME_MODEL_PHYSICS_TESTS.md` section 1.6 derives the persistent antennal-lobe state from the sign
rule: 77 % of the internal LN + PN synapses are excitatory (650,947 / 192,129), 54 aminergic LNs are counted as
excitatory, and the leading eigenvalue (~2,184 synapses) exceeds the ~2.3 Hz self-sustaining threshold. The ignited
population in seed 3 (KCg-m, KCab, lLN1_bc, lLN2P, DA1 lPN, ORNs) is that mode. It is a property of the Shiu
sign rule, not of the port; its occurrence in a given realisation is stochastic.
