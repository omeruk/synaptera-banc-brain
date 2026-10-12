# Pre-registration 01 — transmitter-to-sign rule for BANC

Status: committed before any BANC network run (CLAUDE.md rule 6). Date: 2026-10-12. Decision owner: omeruk.
Not to be edited afterwards; additions go in a new file and are labelled post-hoc.

## Decision
- **Reference arm = the Shiu rule exactly as in Synaptera v1:** GABA and glutamate (`GLUT`) -> -1; every other
  predicted transmitter (ACh, DA, SER, OCT, HIST, TYR) -> +1; a missing prediction -> +1.
  Sign is a property of the presynaptic neuron (neuron-level `Predicted NT type` of the Codex BANC export).
- **Named, flag-only model variants** (never the reference arm, rule 8):
  - `hist_inhibitory`: HIST -> -1 (histamine acts through the HisCl chloride channel, e.g. photoreceptors).
  - `nt_impute`: a neuron without a prediction takes the synapse-weighted majority sign of the predicted
    neurons of its cell type (v1 variant N1; votes use `BANC_PRED_SIGN`, i.e. the Shiu mapping with HIST/TYR = +1).
  - The two may be combined; the combination is recorded as both names.
- **QA report:** every BANC run records the rule name (`sign_rule`), the variants applied (`sign_variants`),
  and the neuron / outgoing-synapse counts per transmitter label (`nt_counts`, label `none` = missing) and the
  per-rule effect (`sign_rules`).

## Implementation
`connectome/sign_rules.py`: `assign_signs(cx, variants)`, `shiu_neuron_sign`, `nt_counts`, `BANC_PRED_SIGN`.
Constants are registered in `configs/assumptions.json` (`connectome.inhibitory_nt`, `connectome.banc_pred_sign`).
