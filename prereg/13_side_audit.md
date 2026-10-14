# Pre-registration 13 — side-convention audit (data and literature only)

Status: committed **before the audit script (`scripts/side_audit.py`) is written or run** (CLAUDE.md rule 6). It is not
edited afterwards; additions go in a new file and are labelled post-hoc. Decision context: the project owner's
instruction after PR #18 (2026-10-14, recorded in `docs/05_ARCHITECTURE_PLAN.md`). Earlier pre-registrations are not
changed.

No simulation of any neural network or flight is part of this audit. The only computations are:
- table look-ups and counts on the annotation files;
- a MuJoCo pose query of the FlyGym body (`mj_forward`, no time step), used for the body-frame check of link L1;
- re-analysis of existing run records.

## 0. What was seen before this file was written (scoping, disclosed)
To scope the audit, the following were inspected interactively. They are re-derived by the script; nothing here is
changed because of them.
- **BANC MN table.** For the wing, haltere and neck MNs: Codex `Soma side` against the meta `nerve` and `side`.
- **F3b-3 input check.** `runs/f3b3/input_check.json`, and the T4a/T4b/T5a/T5b window means per eye from
  `runs/f3b3/stim_rates.npz`.
- **DN side match.** BANC DN `side` against the FlyWire `side` of the `fafb_match` neuron, for all DNs (not per
  family).
- **FlyWire side columns.** `flywire_annotations.tsv` `side` against the Codex `classification.csv.gz` `side`.
- **Literature:**
  - the Codex FAQ text on the FAFB left/right inversion;
  - Azevedo et al. 2024 Supplementary Table 2 (wing MN nerves);
  - the Lindsay et al. 2017 text on the imaged side;
  - the FlyBase / Virtual Fly Brain entry for DLM motor neuron MN5.

## 1. The side rule (owner's rule)
- **Rule.** A motor neuron's side is the side of its muscle, i.e. the side of the peripheral nerve its axon leaves
  through.
- **Source of the nerve.**
  1. The BANC v888 meta `nerve` annotation. Its side is the prefix `left_` / `right_`.
  2. If the nerve has no side (e.g. `abdominal_nerve_trunk`) or is missing, cited published anatomy.
  3. If neither exists, the side is **undetermined**. The soma side is kept, and the neuron is listed.
- **Scope.** Every MN of the BANC meta table (`super_class` motor), not only the readout types.
  - Detailed per-neuron table: every wing, haltere and neck MN type, plus TTMn.
  - Summary per cell class for all other MN classes.
- **Per-neuron verdict.** `match` (nerve side = Codex soma side, the side used by the model), `relabelled` (they
  differ), or `undetermined`.
- **Per-type verdict.**
  - `consistent`: all neurons match.
  - `swapped`: all neurons relabelled, so the type's left and right exchange.
  - `mixed`: some neurons relabelled.
  - `undetermined`.
- **Literature cross-check, per type, where a source exists.** It is reported next to the data and does not override
  them.
  - The nerve of each wing MN type (Azevedo et al. 2024, Nature, Supplementary Table 2: ADMN, mesoAN, PDMN).
  - Soma-muscle side relations of *Drosophila* flight MNs: DLM MN1-4 ipsilateral, MN5 contralateral (Ikeda & Koenig
    1988, as summarised by FlyBase FBbt_00004079).
  - Where no source was read, the report says so.

## 2. The left/right chain: links and verdicts
Each link gets **OK**, **SWAPPED**, or **UNRESOLVED** (neither criterion met; reported with the numbers).

| link | from -> to | OK if | SWAPPED if |
|---|---|---|---|
| L1 | `rot_right` -> eye images | a constant-phase point of condition 1 (`rot_right`) moves to negative azimuth (fly's right; azimuth + = left), and the FlyGym bodies named left (`LEye_cam` and the six `L*` coxae) have body-frame y > 0 and the right ones y < 0 | both reversed |
| L2 | eye images -> FlyVis eye | `camera_rays` eye 0 is `LEye_cam` and its centre ray has azimuth > 0; eye 0 is FlyVis batch row 0 and the `L` group (code) | eye 0 is the right eye |
| L3 | FlyVis eye -> boundary side | every neuron of the `L` boundary group has column-table hemisphere `left` (and `R`: `right`); **and** in the F3b-3 renders, (T4a + T5a) - (T4b + T5b) per eye has the sign expected for front-to-back motion (T4a/T5a prefer front-to-back: Maisak et al. 2013): `rot_right` right eye > 0, left eye < 0; `rot_left` the reverse | all four signs reversed |
| L4 | boundary side -> FlyWire side | column-table hemisphere = FlyWire annotation `side` for >= 99 % of the boundary neurons, **and** annotation `side` = Codex `classification` `side` for >= 99 % of shared neurons (Codex FAQ: Codex sides are the fly's own, after the FAFB inversion was corrected) | agreement <= 1 % |
| L5 | FlyWire side -> bridge | `coupling/dn_bridge.py`: a synthetic two-side table driven on one side only produces rates only on the same side of the BANC target | rates only on the other side |
| L6 | bridge -> BANC DN side | BANC DN Codex `Soma side` = FlyWire `side` of the `fafb_match` neuron for >= 95 % of the matched DNs, both over all DNs and over the 112 ensemble families | <= 5 % |
| L7 | BANC DN side -> BANC MN side | same annotation column within one dataset (no conversion), and no MN of the v2 `lateral` mapping types (b2, b3, i1, i2, hg1-hg4) is relabelled by section 1 | every neuron of those types relabelled |
| L8 | BANC MN side -> mapping side | the mapping applies s_m to the rate of the MN on the side of the wing whose amplitude it changes. This must agree with Lindsay et al. 2017 (imaged side and "ipsilateral" convention, quoted) and with the model's turn convention (larger left amplitude -> right turn: `body/quasi_steady.turn_to_amplitudes`, `tests/test_arena_wing.py::test_turn_sign_in_wrench`) | the convention is reversed |

Other rules:
- L1-L3 together are checked end to end by the T4/T5 sign test. A failure there is reported as SWAPPED for the first
  link whose code check fails, else UNRESOLVED.
- The absolute chirality of each EM volume (whether "left" is the fly's left) cannot be measured from connectivity: a
  mirror image has identical connectivity. It is taken from the dataset documentation (Codex FAQ for FlyWire). The
  same question for BANC is reported as documentation-only.

## 3. Post-audit recomputation (only if any neuron is relabelled)
If section 1 relabels any MN of a readout type, new files labelled **post-audit** are written:
`docs/results/side_audit_post_audit.{json,md}`. Nothing existing is overwritten.
- **Records (existing, not rerun).**
  - F3b-3: `runs/f3b3/f3b3_v3.json` (seeds 10-29, 39 random wirings).
  - mv2 R: `runs/mv2/t3_R.json` (seeds 30-39, 19 random wirings).
- **Relabelling of the records.**
  - Readout rates are stored per type and soma side. A `swapped` type has its two keys exchanged.
  - A `mixed` type cannot be relabelled from aggregate rates, because per-neuron rates are not recorded. It is listed
    as "not recomputable" and left out of any recomputed set that contains it.
- **Prediction.** docs/08 section 10 v2 `lateral` mapping (b2+, b3-, i1-, i2-, hg1-, hg2-, hg3-, hg4-), gain G1 =
  0.3949 deg/Hz.
- **Quantities.**
  - S per seed and its mean.
  - One- and two-sided t-test p.
  - n_pos / n_neg over seeds.
  - The permutation count |S(random_k)| >= |S(ensemble)|, with p = (1 + count) / (n_random + 1).
  - Per-type T = mean D_m and the contribution -s_m T / 2.
  - Turn in degrees at G1.
- **Pre-audit vs post-audit.** Both are given side by side. If no mapping type is relabelled, the script states and
  checks that every post-audit number equals the pre-audit one.

## 4. Earlier results a swap would change (listed, not recomputed)
- For every relabelled type, the script lists the result files under `docs/results/` whose records contain that type
  with a side key or a left/right comparison.
- Examples: F3a-3 steering doses per MN side, the symmetry-gate rows, the mv2 MN rates.
- For a `swapped` type, the symmetry gate |log2(R/L)| is unchanged (only its sign flips). A `mixed` type changes its
  per-side counts, so its gate row would change.
- Nothing is recomputed in this audit.

## 5. Outputs and checks
- **Outputs.** `scripts/side_audit.py` writes `docs/results/side_audit.{json,md}` (and section 3 if triggered). It
  refuses to overwrite without `--overwrite` (rule 13).
- **Verification.** `scripts/verify_report.py` rebuilds the JSON from the data and compares it, re-renders the markdown,
  and recounts the per-type verdicts and S with plain loops.
- **Tests.** `tests/test_side_audit.py`: the nerve-side parser, the per-type verdict rule, the key swap, and the
  link-L5 bridge check.
