# Pre-registration 05 — v3 edge list (arm R3) and per-neuron input normalisation (arm N)

Status: committed **before any simulation of these arms** (CLAUDE.md rule 6). Not to be edited afterwards; additions go
in a new file and are labelled post-hoc. Context: `prereg/04_banc_full_graph.md` and `docs/results/f2_full_graph.md`
(the reference arm on the full v2 v888 graph leaves MN9 silent; FlyWire / BANC in-synapse ratio 1.76). The project
owner's decision of 2026-10-13 fixes the two steps below. No F2 test is run under this registration.

## 1. Data (fixed)
- **Source `banc888_full_v3`:** Harvard Dataverse doi:10.7910/DVN/7WTH1N (BANC deposit, version 3.0, CC BY 4.0), file
  `compiled_data/banc_888_edgelist_simple_v3.feather` (datafile id 13918810; v3 synapse set rolled up at
  `size >= 10`; pair level, no count threshold, autapses excluded at build time; 13,620,865 rows, 42,309,621
  synapses in the downloaded file) with the unchanged `banc_888_meta.feather`. SHA-256 and MD5 in
  `configs/data_sources.yaml` (`banc888_edgelist_v3`).
- **Neuron set:** the 158,262 neurons of the Codex neuron table, edges among them (the *primary* set of prereg/04); the
  engine index order of the Codex table. The "all rows" set is not used here.
- Sign rule, driven GRN sets, FlyWire reference runs (`runs/f2/flywire_v2.json`): exactly as in prereg/04.
- Data-only QA first (`scripts/f2_v3_qa.py`, `docs/results/f2_v3_qa.md`): graph size and the FlyWire / BANC in-synapse
  ratio per super class (matched types and all neurons), next to the Codex export and the v2 full graph.

## 2. Arms
| arm | graph | recurrent weights | origin |
|---|---|---|---|
| **R3** | `banc888_full_v3`, primary set | Shiu unchanged: `w_syn` = 0.275 mV, scale 1.0 | M (Shiu et al.) |
| **N** `innorm_homolog`, primary | `banc888_full` (v2, prereg/04 graph), primary set | weight of every synapse x `s[post]` | P |
| **N** `innorm_homolog`, secondary | `banc888_full_v3`, primary set | weight of every synapse x `s[post]` | P |

The Poisson input weight (`w_syn * f_poi`) is **not** scaled in any arm, so a driven GRN fires as in the reference arm.
The reference arm of the FlyWire graph is unchanged.

### 2.1 Definition of `s` (arm N; fixed now, computed from data only)
- **Homolog (crosswalk):** a BANC neuron has a FlyWire homolog type when its primary `cell_type` equals a primary
  `cell_type` of the FlyWire v783 neuron table (evidence source `name`, the matching of `scripts/f2_wsyn_k.py`),
  plus the two named critical-type entries of `connectome/crosswalk.py`: BANC `MN9` -> FlyWire `CB0701`, BANC `LB3a..d`
  (regex `^LB3[a-z]?$`) -> FlyWire `LB3`. A type is the pooled set of all neurons (both sides, all subtypes of an entry).
- **Per type:** `s_type = mean(FlyWire input synapses per neuron of the type) / mean(BANC input synapses per neuron of
  the type)`, where the input synapse count of a neuron is the sum of `n_syn` over all its incoming pairs **on the graph
  the engine runs** (FlyWire: the Shiu graph, every pair; BANC: the graph of the arm, all neuropils including VNC
  compartments). A type with a zero mean on either side has no `s_type` and is treated as having no homolog.
- **Per neuron:** `s = s_type` of its type. A neuron without homolog gets the **median `s` over the homolog neurons
  of its super class** (BANC super classes mapped optic_lobe_intrinsic -> optic, central_brain_intrinsic -> central;
  a class with no homolog neuron uses the global median over homolog neurons).
- **Clip:** `s` is clipped to [0.25, 4] (E: engineering guard). Reported: number and fraction of neurons clipped (low /
  high), and the distribution of `s` (quantiles 5 / 25 / 50 / 75 / 95 %, mean) per super class and for homolog vs
  fallback neurons. Expectation (not a criterion): about 1 for the antennal lobe and sensory neurons, above 1 for
  central and visual neurons.
- **E/I:** all incoming weights of a neuron (excitatory and inhibitory) get the same `s`, so its E/I balance is kept.
- Known artefact, **not corrected**: BANC ascending neurons receive their inputs largely in the VNC, which the FlyWire
  brain graph does not contain; their `s` is expected to be small (clipped at 0.25). The registered rule is applied as
  written and the effect is reported.
- Implemented by `connectome/innorm.py`; engine call `Brian2LIFEngine.set_weight_scale_per_post(s)`; the SHA-256 of `s`
  is recorded with the run.

## 3. Sanity gates (identical to prereg/04, more seeds)
1. **No input:** all inputs 0 for 1 s, seeds 0-4; 0 spikes in every seed.
2. **After the cut:** prereg/02 protocol (10 intervals drive, 8 cut, 3 idle; ignition = network rate > 1 Hz in the 8th
   cut interval), **50 seeds (200-249)**; passes at 0 / 50. The rate and its 95 % Clopper-Pearson interval are reported.
3. **No runaway:** non-driven rate per driven GRN within 0.2-5x of FlyWire under the same drive (labellar and leg,
   100 Hz, 1 s), now over **10 seeds (0-9)** on the BANC side; FlyWire reference = `runs/f2/flywire_v2.json`
   (seeds 0-2, unchanged). The literal network-rate ratio is reported next to it.

## 4. Readouts (reported, not tuned)
- MN9 rate per neuron (left / right), mean, network rate and non-driven rate for labellar, leg, labellar-left and
  labellar-right drive (100 Hz, 1 s), **10 seeds (0-9)**, with the mean and the standard error over seeds.
- **MN9 responds** (decision quantity): mean MN9 rate over the two MN9 neurons and the 10 seeds under labellar drive
  >= 10 % of the FlyWire labellar reference (`flywire_v2.json`, 59.7 Hz; threshold 5.97 Hz).
- MN9 left / right symmetry gate (|log2(R/L)| < 0.5 of input synapses on the graph of the arm, with the weights as the
  engine uses them); a lateralised response is reported as such.
- **Antennal-lobe leading eigenvalue** of the ALLN + ALPN subnetwork (`connectome/qa_report.al_structure`, signed and
  excitatory-only, weights x `s[post]` for arm N), for R3, N (v2) and N (v3), next to the FlyWire value of the same
  function (unscaled).
- Peak RAM and wall time per arm.

## 5. Decision rules (fixed now)
- A graph or arm is a **candidate BANC reference** only if gates 1-3 pass **and** MN9 responds (labellar drive).
- If several arms qualify, preference order: **R3** (no parameter change) > **N on v2** (primary) > **N on v3**.
- If none qualifies: stop and report. No further scale, route or F2 test is started without a new pre-registration and
  the project owner's decision.
- Any arm that qualifies is **not** declared the reference by this document: it is reported as the candidate; F2 tests
  on it need their own pre-registration.
- No threshold, window, seed or protocol is changed afterwards. Negative results are reported with equal weight.

## 6. Origin classes (CLAUDE.md rule 2)
`s` per neuron: **P** (structural, computed from data only; not fitted to any behaviour or MN9 rate). Clip bounds
[0.25, 4]: **E**. `w_syn` = 0.275 mV: **M**. The 10 % response fraction is a reporting threshold (E).
