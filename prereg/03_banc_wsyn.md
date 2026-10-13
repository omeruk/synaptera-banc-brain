# Pre-registration 03 — BANC `w_syn` variants: structural prior vs Shiu's own fit

Status: committed **before any run of the variants** (CLAUDE.md rule 6). Not to be edited afterwards; additions go in a
new file and are labelled post-hoc. Context: `docs/results/f1_banc_smoke.md` (the reference arm leaves MN9 silent on
BANC), `docs/results/f1_banc_qa.md` section 3, `docs/05_ARCHITECTURE_PLAN.md` risk register. This file registers the
calibration and the sanity gates only. The F2 behavioural / open-loop tests are **not** run under this registration.

## 1. Arms and graphs
- **Reference arm `R`:** the Shiu parameters unchanged (`w_syn` = 0.275 mV, hash-pinned model), reference sign rule
  (`prereg/01_banc_sign_rule.md`), the BANC Codex export as it is (pairs with >= 3 synapses, 3,037,361 pairs,
  23,556,214 synapses). *Graph threshold:* the Shiu FlyWire graph is **not** thresholded (minimum 1 synapse per pair;
  82 % of pairs have < 5, `f1_banc_qa.md` section 3), so no threshold is imposed on BANC: the graph the engine runs is
  the export, and k below is computed on exactly these two graphs. A >= 5 view is a sensitivity table only (`f1_banc_smoke.md`).
  `R` is reported as the reference-arm result: MN9 is silent (0.00 Hz, all conditions, `f1_banc_smoke.md`).
- **Variants** are named model variants switched on by a flag; nothing else differs from `R`. In all variants the
  scale multiplies the *recurrent synaptic weights* (`Brian2LIFEngine.set_weight_scale` /
  `set_weight_scale_per_post`); the Poisson input weight (`w_syn * f_poi`) is not scaled, so a driven GRN fires as in `R`.

## 2. Structural k (data only, no simulation)
`k` = FlyWire / BANC ratio of the mean number of input synapses per postsynaptic neuron, computed by
`scripts/f2_wsyn_k.py` on the two graphs as the engine runs them, on a **matched population** (neurons whose cell type
name exists in both datasets; super class mapped optic_lobe_intrinsic -> optic, central_brain_intrinsic -> central).
`docs/results/f2_wsyn_k.json`; all numbers below are written by `scripts/f2_wsyn_constants.py`.

|---|---|---|---|---|---|---|
| all | 131,788 / 90,558 | 2.71 | 2.72 | 2.64 | 2.06 | 2.19 |
| sensory | 15,933 / 8,102 | 1.89 | 1.90 | 1.39 | 1.25 | 1.43 |
| central | 29,710 / 28,411 | 2.80 | 2.80 | 2.91 | 2.24 | 2.40 |
| optic | 76,535 / 45,930 | 3.60 | 3.60 | 4.12 | 2.64 | 2.88 |
| visual_projection | 7,987 / 6,734 | 4.57 | 4.57 | 4.60 | 3.22 | 3.74 |
| descending | 961 / 814 | 1.75 | 1.84 | 1.96 | 1.56 | 1.63 |
| ascending | 54 / 25 | 0.13 | 1.68 | 0.23 | 0.08 | 0.07 |
| motor | 32 / 31 | 3.50 | 3.50 | 1.30 | 3.25 | 3.45 |

Reading: k is 1.9 (sensory) to 4.6 (visual projection); the literal as-run ratio of **ascending** neurons (0.13) is a
compartment artefact: BANC ascending neurons carry their dendrites in the VNC, which FlyWire (brain only) does not
contain. Counting BANC input in brain and SEZ neuropils only removes it (1.68) and leaves the other classes
unchanged (the brain-neuropil column). The numbers of matched neurons of ascending and motor classes are small (54 /
25 and 32 / 31). The "k both >= 3 / >= 5" columns apply the same threshold to both graphs; they are sensitivity
columns and do not define a variant. (Decision made after seeing these data-only numbers, before any simulation.)

## 3. Variants
| name (flag) | definition | origin |
|---|---|---|
| **A `wsyn_struct`** (primary) | every recurrent weight x k_global = ratio "all", "k as run, matched" | P (structural prior, `configs/banc_wsyn.yaml`) |
| **A2 `wsyn_struct_class`** | recurrent weight onto a neuron x k of the neuron's super class ("brain neuropils" column); neurons of other / unassigned classes use k_global | P |
| **B `wsyn_fit_mn9`** (secondary) | recurrent weights x s_B, where s_B is Shiu's own criterion applied to BANC: see section 4 | F, `fitted_to` = that criterion |

Under B the sugar -> MN9 open-loop test is circular (the scale is chosen from MN9 under sugar drive). That F2 test is
**excluded from the F2 evaluation of B and marked as such**; B is judged only on the other tests and the gates below.

## 4. Variant B: Shiu's criterion and its operational definition
Shiu et al. (Nature 2024, Methods, "Computational model"): *"We chose W_syn such that activation of sugar GRNs at
100 Hz resulted in roughly 80% of maximal MN9 firing"*. The paper does not define "maximal" (checked in the paper
text; the repository only labels `w_syn` "Free parameter"). Operational definition fixed here:
- drive: the labellar sugar GRN set at constant Poisson rates for 1 s (25 ms steps, as `f1_banc_smoke.py`); MN9 rate =
  mean over MN9 neurons and seeds of spikes / 1 s;
- r(f; s) = MN9 rate at GRN rate f for recurrent scale s; **maximal** = max over f in {20, 40, ..., 200} Hz (a subset of
  Shiu's Fig. 1d grid that contains 100 Hz and 200 Hz) at the same s;
- criterion: **c(s) = r(100; s) / r_max(s) = 0.80**; s_B is the smallest s in [1, 12] with c(s) >= 0.80, located by
  bisection on log s to a relative tolerance of 2 %; where r_max = 0 (MN9 silent) c is undefined and counts as < 0.80;
- calibration seeds 1000 and 1001 (the evaluation seeds 0-2 and the gate seeds are different); s_B is then fixed.
- **Validation of the definition on FlyWire first** (same procedure, FlyWire v783 graph, sugar GRN set
  `data/sugar_grn_783.csv`): it must return s within [0.9, 1.1] (w_syn within 0.2475-0.3025 mV of 0.275 mV).
  Otherwise the operational definition is declared not validated, B is reported with that label and **not used**.
- B's fitted s_B and the FlyWire check go to `configs/banc_wsyn.yaml` after the calibration run, in a commit that
  is separate from this file.

## 5. Sanity gates (identical for all variants; fixed now)
1. **No input:** all inputs 0 for 1 s, seeds 0-4: network rate = 0 Hz (0 spikes in every seed).
2. **After the cut:** protocol of `prereg/02`: 10 intervals of 25 ms with labellar + leg sugar GRNs at 100 Hz, 8 intervals
   all inputs 0, 3 idle intervals; **ignition = mean network rate > 1 Hz in the 8th cut interval**; 20 seeds
   (200-219). The gate passes at 0 / 20 ignitions (Clopper-Pearson interval reported).
3. **No runaway:** under 100 Hz sugar drive (1 s, 3 seeds, seeds 0-2), the network rate must lie within 0.2-5x the
   FlyWire rate under the homologous drive (labellar: `data/sugar_grn_783.csv`; leg: the `selected` rows of
   `data/leg_sugar_grn_783.csv`; same protocol, run by the same script on the FlyWire v783 graph). Because the number of
   driven GRNs differs strongly (labellar 90 vs 36; leg 298 vs 12), the **gate quantity is the mean rate of the
   non-driven neurons per driven GRN**; the literal all-neuron network rate ratio is reported next to it.
   (Decision made here, before any variant run; the literal ratio would be set by the GRN count, not by runaway.)
A variant that fails a gate is reported and **not used**. No gate, window, seed or protocol is changed afterwards.

## 6. Run list (calibration and gates only)
- `R`, A, A2, B: gates 1-3; for each: MN9 rate per neuron (left / right) and network rate for labellar-only, leg-only,
  labellar-left-only and labellar-right-only sugar drive, 100 Hz, 1 s, seeds 0-2. Peak RAM and wall time.
- FlyWire reference runs for gate 3 and the B validation; BANC calibration of s_B.
- Nothing else. The F2 tests (sugar -> MN9, tarsal GRN -> MN9, visual, steering, take-off) are not run.

## 7. Decision rules
- Variant A is the primary; A2 and B are reported in full. The variant used downstream is not chosen by its MN9 rate:
  A is used if it passes the gates; if A fails a gate, this is reported and the next step is a new pre-registration.
- No variant is added, and no k is recomputed, after these runs. A variant that is added later is labelled post-hoc.
- Negative results (a variant that fails a gate, MN9 still silent under A) are reported with equal weight.

## 8. Origin classes (CLAUDE.md rule 2)
k_global and the class k: **P**, registered in `configs/banc_wsyn.yaml` with their source. s_B: **F**, `fitted_to`: the
criterion of section 4 (MN9 rate under labellar sugar drive; open-loop physiology, no behaviour; CLAUDE.md rule 3).
