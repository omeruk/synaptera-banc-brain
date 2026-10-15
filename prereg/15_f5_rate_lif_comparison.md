# Pre-registration 15 — F5 step 2a: the untrained rate VNC against the LIF VNC on the F3a-3 protocol

Status: committed **before the rate model is run on any connectome condition** (CLAUDE.md rule 6). It is not edited
afterwards; additions go in a new file and are labelled post-hoc. Decision context: the project owner's instruction for
F5 step 2a (2026-10-15): implement the sparse PyTorch rate model of prereg/14 section 2 and validate it qualitatively
against the LIF VNC on the F3a-3 protocol (DNg02 recruits the power MNs; DNa04 / DNa10 / DNa15 steer ipsilaterally).
prereg/14 is not changed by this file.

## 0. What was seen before this file was written

- The committed F3a-3 results (`docs/results/f3a3_vnc_subgraph.{md,json}`): every LIF number used below.
- The VNC subgraph was loaded once to check the edge signs: 1,753,807 excitatory and 1,266,526 inhibitory edges, no
  silenced edge, and every presynaptic neuron has one sign. No rate-model state was computed on it.
- A wall-time test of three sparse products (random edge values, random states) to choose the implementation. No
  result of it enters this file.

## 1. What is compared

- **Rate model.** `engine/rate_torch.py`, the prereg/14 section 2 dynamics, at initialisation (the `untrained` arm:
  every gain 1, every leak 0.25), float32, on the prereg/08 VNC subgraph with the prereg/01 sign rule. It is
  deterministic: there is no noise, so one run per condition replaces the LIF's 10 seeds.
- **LIF reference.** The F3a-3 record of the unmodified reference arm R (`docs/results/f3a3_vnc_subgraph.json`), seed
  means.
- **Nothing is fitted.** No rate-model constant is chosen, changed or tuned by this comparison (rule 3). A mismatch is
  reported; what follows from it is the owner's decision.

## 2. Protocol (F3a-3 transposed to the rate model)

- One update per 5 ms interval. Drive: 200 intervals (1 s, as the F3a-3 drive). Cut: 40 intervals (200 ms, as the F3a-3
  R1 / R3 cut). s = 0 at the start.
- A DN group driven at r Hz has every one of its neurons clamped to r / R_IN (R_IN = 100 Hz). Every other DN is clamped
  to 0. The groups are the F3a-3 groups (`scripts/f3a2_vnc_run.py:groups_of`).
- Conditions: the F3a-3 R1 drives (G20, G50, G100, GL100, GR100) and the 25 R3 conditions (baseline `B` = DNg02 both
  sides at 100 Hz, `HAND`; plus DNa04 / DNa10 / DNa15 on one side at 10, 25, 50, 100 Hz).
- Response of an MN type and side: the mean state over the 200 drive intervals and over the neurons of that type and
  side. It is also reported as R_OUT x state (Hz-equivalent, R_OUT = 100 Hz). The state is signed; negative means
  inhibited below rest.
- Steering effect, as in F3a-3: d = (s_R - s_L) under the steering condition minus (s_R - s_L) under `B`.
  Ipsilateral steering means d < 0 for a left drive and d > 0 for a right drive. A |d| below 1e-6 (state units) is
  counted as no response.

## 3. Registered criteria

- **C1, power recruitment.** Under G100, the mean state of DLM1-4, DVM1a-c, tp2 and ps1 is > 0 on both sides (8
  type-sides). In the LIF all 8 are above 20 Hz. C1 agrees if all 8 are > 0.
- **C2, steering sign.** For every F3a-3 R3 test that steers in the LIF (paired t-test, Holm p < 0.05, any rate; the
  27 tests; the LIF mean d of every one is ipsilateral), the rate model's d has the LIF sign and |d| >= 1e-6. C2
  agrees if every such test agrees. The count of agreeing tests is reported per family.

The comparison as a whole is reported as "qualitative agreement" if C1 and C2 both agree; otherwise each failing item
is listed as a mismatch.

## 4. Descriptive (reported, no criterion)

- Dose order of the C1 types under G20, G50, G100.
- G100 recruitment of every other readout type, and the unilateral GL100 / GR100 drives.
- The Spearman correlation across type-sides between the rate model's G100 state and the LIF G100 rate.
- Steering: the rate-model d for every family, rate, side and gate-passing MN type, the agreement of its sign with the
  LIF mean d where the LIF mean d is non-zero, the lowest rate at which each family gives an ipsilateral d on a type
  that steers in the LIF, and the off-target effects (types that do not steer in the LIF).
- Decay after the cut (the F3a-3 R2 analogue): mean |s| over the network in the last 100 ms of the cut against the
  last 100 ms of the drive under `B`.
- The prereg/14 engine checks on a 2,000-neuron subgraph of the real VNC (forward against NumPy, sparse backward
  against dense float64), next to the same checks on synthetic graphs in the unit tests.
