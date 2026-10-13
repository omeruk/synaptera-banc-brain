# F2 prep: full BANC v888 graph, data-only QA (`scripts/f2_full_graph_qa.py`)

Source: `banc_888_edgelist_simple_v2.feather` (Harvard Dataverse doi:10.7910/DVN/7WTH1N, synapse set v2 at `size >= 5`, pair level, no count threshold, autapses excluded) and `banc_888_meta.feather`. Codex Root IDs are the v888 ids (all 158,262 are in the meta table). Numbers are written by the script.

## 1. Graph size

| graph | neurons | pairs | synapses | median syn/pair | pairs >= 3 | pairs >= 5 | mean in-edges | mean in-syn |
|---|---|---|---|---|---|---|---|---|
| FlyWire v783 (Shiu) | 138,639 | 15,091,983 | 54,492,922 | 2 | 4,916,231 | 2,700,513 | 108.9 | 393.1 |
| codex_export_ge3 | 158,262 | 3,037,361 | 23,556,214 | 5 | 3,037,361 | 1,528,585 | 19.2 | 148.8 |
| banc888_full_codex | 158,262 | 11,582,757 | 35,431,027 | 1 | 3,146,614 | 1,606,327 | 73.2 | 223.9 |
| banc888_full_all | 188,508 | 11,752,828 | 35,733,096 | 1 | 3,170,319 | 1,614,479 | 62.3 | 189.6 |

## 2a. FlyWire / BANC ratio of mean input synapses per postsynaptic neuron, matched cell types

| class | codex_export_ge3 | banc888_full_codex | banc888_full_all |
|---|---|---|---|
| all | 2.71 (90,558) | 1.76 (90,558) | 1.77 (90,956) |
| sensory | 1.89 (8,102) | 0.94 (8,102) | 0.94 (8,108) |
| central | 2.80 (28,411) | 1.92 (28,411) | 1.92 (28,418) |
| optic | 3.60 (45,930) | 2.20 (45,930) | 2.21 (46,310) |
| visual_projection | 4.57 (6,734) | 2.44 (6,734) | 2.43 (6,737) |
| descending | 1.75 (814) | 1.39 (814) | 1.39 (814) |
| ascending | 0.13 (25) | 0.09 (25) | 0.09 (25) |
| motor | 3.50 (31) | 2.86 (31) | 2.86 (31) |

## 2b. FlyWire / BANC ratio of mean input synapses per postsynaptic neuron, all neurons of the class

| class | codex_export_ge3 | banc888_full_codex | banc888_full_all |
|---|---|---|---|
| all | 2.64 (158,262) | 1.76 (158,262) | 2.07 (188,508) |
| sensory | 1.39 (16,557) | 0.74 (16,557) | 0.74 (16,564) |
| central | 2.91 (31,782) | 1.99 (31,782) | 1.99 (31,887) |
| optic | 4.12 (72,582) | 2.41 (72,582) | 2.40 (73,039) |
| visual_projection | 4.60 (7,321) | 2.48 (7,321) | 2.47 (7,325) |
| descending | 1.96 (1,316) | 1.53 (1,316) | 1.53 (1,316) |
| ascending | 0.23 (1,849) | 0.16 (1,849) | 0.16 (1,849) |
| motor | 1.30 (805) | 1.17 (805) | 1.17 (805) |

Values are FlyWire / BANC (number of BANC neurons in the class). The BANC super classes are mapped as optic_lobe_intrinsic -> optic, central_brain_intrinsic -> central; ascending and motor include VNC compartments that FlyWire (brain only) lacks, so their ratios are read with the brain-neuropil caveat of `f2_wsyn_k.json`.
