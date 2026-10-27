# Plan 21: post-hoc analyses of stage A (speed generalisation and structure)

Status: **post-hoc, exploratory; written after the stage-A test, before these analyses** (CLAUDE.md rule 6). This file is
not a pre-registration in the sense of prereg/14: it fixes the analyses before they are computed, but every choice in it
was made after the stage-A test report had been read. Nothing here enters or changes a stage-A verdict, the Holm family,
a guard, a learning-rate choice or a checkpoint selection. prereg/14 to prereg/20 are not changed. The file is not
edited after the analyses start; additions go in a new file.

Every output (JSON, Markdown, figure, run record) carries the label "post-hoc, exploratory".

## 0. What was seen before this file was written

- The stage-A test report (`docs/results/f5_stageA_report.{json,md}`, PRs #32 and #33), including the per-seed test
  metrics of every registered run at the training speeds and at 30 and 90 deg/s. On the test split, `true` had higher
  R2 than `shuffled` at 30 deg/s and lower R2 at 90 deg/s, with turn-sign accuracy at ceiling for both.
- The stage-A validation report (`docs/results/f5_stageA_validation.{json,md}`) and the validation evaluations logged
  during training (data seeds 160-199, training speeds 60 and 120 deg/s only).
- The visualisation work on branch `omeruk/viz-positions`, which read validation seeds 160-199 at 60 and 120 deg/s
  only.
- **Never read:** the validation seeds 160-199 at 30 and 90 deg/s. No record at a held-out speed has been opened for
  any data seed other than the test seeds read once by `final-eval`.

## 1. Rules for these analyses

- No training. No test seed (200-239) is read again; `final-eval` is not run; no `test.lock` or `test.json` is created,
  changed or deleted. The test numbers used in section 4.4 are read from the committed report JSON only.
- CPU only, one process at a time, deterministic algorithms on, `A.TEST_THREADS` threads (as the final evaluation).
- The reader of the held-out-speed validation records is new code in `scripts/f5_stageA_posthoc.py`; it does not call
  `final_evaluation()` and refuses to run inside it. It accepts only data seeds 160-199, the stimuli `rot_left` and
  `rot_right` and the speeds 30 and 90 deg/s; any other seed, stimulus or speed is refused before a file is opened, a
  test seed with `SealedSplitError`. Each record is checked against `runs/f5/stageA/manifest.sha256`, whose own SHA-256
  is pinned to `33ddb2423ba40f9ed3a189f5afb979981ad0606a8cae06935c9386bc78c26476` (the manifest when this file was
  written).
- The 60 and 120 deg/s validation trials are read through the existing `train.f5_stageA.load_trials("val", ...)`.
- No existing file is changed, except that one check is appended to `scripts/verify_report.py` (part 3).
- Outputs: `docs/results/f5_stageA_posthoc.{json,md}` (committed); predictions and figure under
  `runs/f5/A_posthoc/` (never committed), the figure copied to the Windows preview folder for viewing. An existing
  output is never overwritten without `--overwrite` (rule 13).

## 2. Runs

Every registered stage-A run at its selected checkpoint (`best.pt`) and selected learning rate
(`runs/f5/A/<arm>/lr_selected.json`):

| arm | seeds | used in |
|---|---|---|
| `true` | 0-9 | A (replication, tuning), B (deficit) |
| `shuffled` | 0-9 | A (replication, tuning), B (deficit) |
| `derange` | 0-9 | A (replication, tuning) |
| `mlp` | 0-9 | A (replication, tuning) |
| `linear` | 0-9 | A (replication, tuning) |
| `untrained` | 0 | A (tuning only) |

`bridge_off` and the prereg/19 replicate run are left out. Each model is built as in `scripts/f5_train.py final-eval`
(`A.make_bridge`, `A.make_model` with the cached shuffled graphs of `runs/f5/A/graphs/`) and loaded from `best.pt` on
the CPU.

**Consistency check, before any held-out-speed record is read.** For every run except `untrained`, the validation
metrics at the training speeds (240 trials, `A.TRAIN_CONDITIONS`, `A.metrics`) are recomputed on the CPU and compared
with the validation evaluation logged at the selected step (`log.jsonl`). They must agree to |delta R2| <= 1e-4 and
|delta loss| <= 1e-4, with equal turn-sign accuracy (the logged values come from GPU training, so they agree only to
float32 rounding, prereg/19). If any run fails, the analysis stops and the failure is reported; no held-out-speed
record is read.

## 3. Part A: speed generalisation on the validation seeds

### 3.1 Data

Validation data seeds 160-199. Held-out speeds: `rot_left` and `rot_right` at 30 and at 90 deg/s (80 trials per speed,
all rotation trials), read with the new reader. Training speeds: the 60 and 120 deg/s validation trials through
`load_trials` (the rotation trials of the 240 trials of the consistency check).

### 3.2 Replication of the held-out-speed metrics

Per run, `A.metrics` on the groups of `A.TEST_GROUPS["held_out_primary_90"]` and
`A.TEST_GROUPS["held_out_secondary_30"]` (the same conditions as `final-eval`, here at seeds 160-199): R2 and turn-sign
accuracy of the window mean (`A.WINDOW`).

**The one primary post-hoc test.** d_t = R2_30(`true`, t) - R2_30(`shuffled`, t) on the validation seeds, t = 0-9.
H1: mean d > 0 (the direction of the test result). One-sided exact sign-flip test over all 1,024 sign patterns
(`sign_flip` of `scripts/f5_stageA_report.py`, imported unchanged); effect size: mean d with a 95 % percentile
bootstrap interval over the 10 seeds (10,000 resamples, `default_rng(21)`). The result is labelled "replicated on the
validation seeds (post-hoc)" if mean d > 0 and one-sided p < 0.05, otherwise "not replicated on the validation seeds
(post-hoc)". The smallest attainable p is 1/1024. This label never changes the stage-A verdict.

Everything else in part A is descriptive: per-run R2 and sign accuracy at 30 and 90 deg/s for every arm; paired
differences `true` minus each other arm with 95 % bootstrap intervals (10,000 resamples, a fresh `default_rng(21)` per
comparison); no p-value.

### 3.3 Speed tuning

Per run (including `untrained`), on the 320 validation rotation trials (8 conditions x 40 seeds; 30, 60, 90, 120 deg/s,
both directions):

- **Response per condition:** the mean over the 40 seeds of the window-mean dA, per speed and direction.
- **Gain and offset:** ordinary least squares of the window-mean dA on the teacher dA* (`A.target_value`) over the 320
  trials: dA = g dA* + b. Reported: g, b and the R2 of the fit.
- **Gain ratio per speed:** G(v) = mean over the 80 trials at speed v of the directed dA, sign(dA*) x dA, divided by
  |dA*(v)| = K_P v. G = 1 is the teacher.

Summaries per arm: mean and SD over the 10 seeds of g, b and G(v); per-seed values in the JSON.

## 4. Part B: structure (no training)

### 4.1 Graphs

The `true` VNC graph of `runs/f5/A/setup.npz` and the 10 cached `shuffled` graphs (`runs/f5/A/graphs/shuffled_t<t>.npz`,
t = 0-9, read through `A.shuffled_post`, which checks their seed and the `true` graph SHA-256). Each edge keeps its
synapse count and its presynaptic sign (prereg/01) in the shuffled graphs; only the postsynaptic end moves. `derange`
is left out (its VNC graph is the `true` graph).

Sources: the bridge-target BANC DNs of the `true` bridge (`bridge.target_index`), with their side from the BANC DN
table. Targets: the 16 steering MNs of the readout (`setup.readout`; 8 types x left / right, `setup.meta["readout_names"]`).
DNs are graph inputs with no incoming edges, so no path passes through a second DN.

### 4.2 Definitions

For a source DN i and a target MN j, with n(a, b) the synapse count of edge a -> b and s(a) the sign of presynaptic
neuron a:

- 1-hop weight W1(i, j) = n(i, j); 2-hop weight W2(i, j) = sum over x of n(i, x) n(x, j) (product of synapse counts).
- Path weight P = W1 + W2. Signed path weight Q = s(i) n(i, j) + sum over x of s(i) s(x) n(i, x) n(x, j) (the sign of a
  path is the product of the signs of its presynaptic neurons).
- Path counts: the number of 1-, 2- and 3-hop walks from the sources to the targets (powers of the binary adjacency
  matrix). With no self-loops and no edges into DNs, 1- and 2-hop walks are paths; a 3-hop walk can revisit a target
  MN, and is counted as a walk.
- Shortest-path length: unweighted hop count per (DN, MN) pair (breadth-first search on the reversed graph from each
  MN), over all source x target pairs; unreachable pairs counted separately.

### 4.3 Measures

Computed for the `true` graph and each of the 10 shuffled graphs:

| id | measure |
|---|---|
| S1 | mean shortest-path length over reachable pairs (with the full length distribution: 1, 2, 3, 4, >= 5, unreachable) |
| S2 | fraction of pairs with shortest path <= 2 hops |
| S3, S4, S5 | numbers of 1-, 2- and 3-hop walks |
| S6 | summed 1-hop weight, sum of W1 |
| S7 | summed 2-hop weight, sum of W2 |
| S8 | ipsilateral fraction: sum of P over pairs with DN side = MN side, divided by the sum of P over all lateralised pairs |
| S9 | excitatory fraction: sum of the positive part of the per-path signed weight, divided by the sum of P |
| S10-S17 | signed lateral influence index per MN type m (b2, b3, i1, i2, hg1-hg4): L_m = [Q(S_L -> m_L) - Q(S_R -> m_L)] - [Q(S_L -> m_R) - Q(S_R -> m_R)], with S_L, S_R the left- and right-side targets |
| S18 | readout-aligned lateral index: sum over m of r_m L_m, with r_m the readout sign of type m (b2 +1; b3, i1, i2, hg1-hg4 -1) |

For each measure: the `true` value, the 10 shuffled values (mean, SD, range), the rank of `true` among the 11 graphs
(rank 1 = largest; ties get the average rank), and whether `true` lies outside the shuffled range. No test.

### 4.4 Structure against the per-seed deficit

The deficit vectors, per training seed t = 0-9, d_t = R2(`true`, t) - R2(`shuffled`, t):

- D1: test split, training speeds (the primary d of `docs/results/f5_stageA_report.json`);
- D2: test split, 30 deg/s (`g30_r2` of the same report JSON);
- D3: validation split, 30 deg/s (section 3.2).

D1 and D2 are read from the committed report JSON; no test record is opened. For each measure, the shuffled-graph value
x_t of the graph of training seed t is correlated with each deficit vector: Spearman rho (average ranks), n = 10, no
correction, descriptive. A measure that is constant over the shuffles gets no correlation. With 18 x 3 = 54
correlations and no correction, about 2.7 are expected at |rho| >= 0.648 (the two-sided 5 % critical value at n = 10)
when nothing is there; the md says so next to the table.

## 5. Figure

One Nature-style figure, four panels:

- **a** Speed tuning: gain ratio G(v) against drum speed per arm (seed means with per-seed points; `untrained` as one
  run), the teacher at G = 1, training and held-out speeds marked.
- **b** 30 deg/s replication per seed: validation R2 of `true` and `shuffled` per training seed, paired, with the
  primary post-hoc test result in the panel.
- **c** Structure, `true` against the shuffles: per measure, the `true` value against the 10 shuffled values on a common
  standardised axis ((value - shuffled mean) / shuffled SD).
- **d** The strongest structure-deficit correlation: the (measure, deficit vector) pair with the largest |rho| in
  section 4.4 (ties: the first in the order S1-S18, then D1-D3), as a scatter of x_t against d_t with the 10 seeds
  labelled. The caption says that it was selected as the largest of 54 uncorrected correlations.

## 6. Outputs and checks

- `docs/results/f5_stageA_posthoc.json`: every number above, per run and per graph, with the SHA-256 of every input
  (records, `best.pt`, setup, shuffled graphs, split file, report JSON) and of this file.
- `docs/results/f5_stageA_posthoc.md`: generated from the JSON; every number is printed by the script (rule 11).
- `scripts/verify_report.py`: one appended check, which recomputes the part-B measures from the graphs, the
  primary post-hoc test and the speed-tuning summaries from the saved predictions, and checks the md text against the
  JSON.

## 7. Limitations, stated now

- The direction of the primary post-hoc test and the choice of 30 deg/s were made after seeing the test result.
- The validation seeds are not independent of model selection: the learning rate and the checkpoint were chosen on the
  validation loss of the same data seeds at the training speeds. The held-out-speed records of these seeds were never
  used.
- The 30 deg/s teacher is small (K_P x 30 deg/s), so the R2 at 30 deg/s, computed over the 80 rotation trials of that
  speed only, is sensitive to small offsets.
- Part B has one `true` graph: its rank among 11 graphs is descriptive, and the structure-deficit correlations have
  n = 10 with no correction.
