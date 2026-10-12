# F0 acceptance flight (docs/06 section 3) — result

Run: `scripts/run_f0.py` (tasks/f0_flight.py), seed 3, 25 ms intervals, full brain (138,639 neurons), v1 n1
configuration. Numbers: `docs/results/f0_acceptance.json` (copied from runs/f0/*_report.json).

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
