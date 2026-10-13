# Pre-registration 04 — reference arm on the full BANC v888 graph (`banc888_full`)

Status: committed **before any simulation on this graph** (CLAUDE.md rule 6). Not to be edited afterwards; additions go
in a new file and are labelled post-hoc. Context: `prereg/03_banc_wsyn.md` and `docs/results/f2_wsyn_calibration.md`
(uniform `w_syn` scaling cannot make BANC MN9 respond without self-sustained activity). Hypothesis tested here:
part of BANC's apparent sparseness comes from the Codex export filter (pairs with >= 3 synapses), not from the brain.
This file registers one reference-arm run and the sanity gates. No scaling variant, no F2 test.

## 1. Data (fixed)
- **Source `banc888_full`:** Harvard Dataverse doi:10.7910/DVN/7WTH1N (BANC deposit, version 3.0, CC BY 4.0), files
  `compiled_data/banc_888_edgelist_simple_v2.feather` and `compiled_data/banc_888_meta.feather`; names, SHA-256,
  materialization and licence in `configs/data_sources.yaml`. Pair-level edge list at materialization v888, **no
  synapse-count threshold**, autapses excluded at build time.
- **Which edge list (v2 vs v3):** v2 is used because it is the table of the BANC paper (all figures) and is the same
  synapse model family as the Codex export (checked data-only: the v2 pairs with >= 3 synapses among the Codex neurons
  are 3,146,614 against 3,037,361 in the Codex export, `docs/results/f2_full_graph_qa.md`). The v3 edge list
  (`size >= 10`, 13.5 M pairs) is not run in this step.
- **Neuron sets (fixed before any run):** *primary* = the 158,262 neurons of the Codex neuron table (the same neurons
  and annotations as every earlier BANC result; edges among them; engine index order of the Codex table); *sensitivity*
  = all 188,508 rows of `banc_888_meta.feather` (annotations of the extra rows from the meta table, NT names mapped).
  The Codex Root IDs are the v888 ids (all in the meta table). Decisions are taken on the primary set only.
- Sign rule: the reference rule of `prereg/01_banc_sign_rule.md`, unchanged. Neurons driven: the sugar GRN sets of
  `prereg/03` section 5 (labellar `body_part == labellum`, leg `front/middle/hind_leg`, function contains "sugar",
  sensory and sensory_ascending), by name in the Codex table, unchanged.

## 2. Arm
Reference arm `R`: Shiu parameters unchanged (`w_syn` = 0.275 mV, hash-pinned model, weight scale 1.0, Poisson input
weight unscaled). Nothing else differs from `R` of `prereg/03`; only the graph differs.

## 3. Sanity gates (identical to prereg/03 section 5; same code, seeds and FlyWire reference)
1. No input: all inputs 0 for 1 s, seeds 0-4, 0 spikes.
2. After the cut (prereg/02 protocol: 10 intervals drive, 8 cut, 3 idle; ignition = network rate > 1 Hz in the 8th cut
   interval), 20 seeds (200-219); passes at 0 / 20.
3. No runaway: non-driven rate per driven GRN within 0.2-5x of FlyWire (labellar and leg, 100 Hz, 1 s, seeds 0-2;
   FlyWire reference from `runs/f2/flywire_v2.json`, same protocol); the literal all-neuron network-rate ratio is
   reported next to it.

## 4. Readouts (reported, not tuned)
- MN9 firing rate per neuron (left / right) and mean, network rate and non-driven rate, for labellar-only, leg-only,
  labellar-left-only and labellar-right-only sugar drive, 100 Hz, 1 s, **3 seeds** (0-2).
- **MN9 responds** (decision quantity, fixed now): mean MN9 rate over the two MN9 neurons and 3 seeds under labellar
  drive >= 10 % of the FlyWire reference under the same protocol (`runs/f2/flywire_v2.json`; ~6 Hz if the FlyWire
  reference is ~60 Hz). The per-side rates are reported with the symmetry gate result for MN9
  (|log2(R/L)| < 0.5 of input synapses on this graph; CLAUDE.md rule 5). A lateralised response is reported as such.
- Graph size: pairs, synapses, FlyWire / BANC in-synapse ratio per super class for both sets
  (`scripts/f2_full_graph_qa.py`, data only, already produced), peak RAM and wall time.

## 5. Decision rules
- If gates 1-3 pass **and** MN9 responds under labellar drive: the full graph is the candidate BANC reference graph;
  the next step (F2 tests on it) needs its own pre-registration.
- If a gate fails, or MN9 does not respond: stop and report; the next route (per-neuron input normalisation or an
  E/I-preserving scale) needs its own pre-registration and a decision by the project owner. No scale is tried here.
- No threshold, window, seed or protocol is changed afterwards. Negative results are reported with equal weight.

## 6. Origin classes (CLAUDE.md rule 2)
No new constant: `w_syn` stays 0.275 mV (M, Shiu et al.); the 10 % response fraction is a reporting threshold of this
document (E), not a model parameter.
