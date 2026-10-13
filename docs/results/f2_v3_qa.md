# F2 prep: v3 full graph and input normalisation, data-only QA (`scripts/f2_v3_qa.py`)

Primary neuron set: the 158,262 Codex neurons, edges among them, no threshold. `banc888_full_v3` = `banc_888_edgelist_simple_v3.feather` (v3 synapse set, `size >= 10`; Harvard Dataverse doi:10.7910/DVN/7WTH1N). Numbers are written by the script; criteria are those of `prereg/05_banc_v3_and_innorm.md`.

## 1. Graph size

| graph | neurons | pairs | synapses | median syn/pair | pairs >= 3 | pairs >= 5 | mean in-edges | mean in-syn |
|---|---|---|---|---|---|---|---|---|
| FlyWire v783 (Shiu) | 138,639 | 15,091,983 | 54,492,922 | 2 | 4,916,231 | 2,700,513 | 108.9 | 393.1 |
| codex_export_ge3 | 158,262 | 3,037,361 | 23,556,214 | 5 | 3,037,361 | 1,528,585 | 19.2 | 148.8 |
| banc888_full_v2 | 158,262 | 11,582,757 | 35,431,027 | 1 | 3,146,614 | 1,606,327 | 73.2 | 223.9 |
| banc888_full_v3 | 158,262 | 13,414,125 | 41,938,506 | 1 | 3,738,568 | 1,915,917 | 84.8 | 265.0 |

## 2a. FlyWire / BANC ratio of mean input synapses per postsynaptic neuron, matched cell types

| class | codex_export_ge3 | banc888_full_v2 | banc888_full_v3 | v2 after N | v3 after N |
|---|---|---|---|---|---|
| all | 2.71 (90,558) | 1.76 (90,558) | 1.47 (90,558) | 0.85 | 0.84 |
| sensory | 1.89 (8,102) | 0.94 (8,102) | 0.64 (8,102) | 0.50 | 0.49 |
| central | 2.80 (28,411) | 1.92 (28,411) | 1.65 (28,411) | 0.96 | 0.95 |
| optic | 3.60 (45,930) | 2.20 (45,930) | 1.74 (45,930) | 1.00 | 0.99 |
| visual_projection | 4.57 (6,734) | 2.44 (6,734) | 2.02 (6,734) | 0.98 | 0.97 |
| descending | 1.75 (814) | 1.39 (814) | 1.19 (814) | 0.92 | 0.92 |
| ascending | 0.13 (25) | 0.09 (25) | 0.08 (25) | 0.33 | 0.31 |
| motor | 3.50 (31) | 2.86 (31) | 2.34 (31) | 0.96 | 0.95 |

## 2b. FlyWire / BANC ratio of mean input synapses per postsynaptic neuron, all neurons of the class

| class | codex_export_ge3 | banc888_full_v2 | banc888_full_v3 | v2 after N | v3 after N |
|---|---|---|---|---|---|
| all | 2.64 (158,262) | 1.76 (158,262) | 1.48 (158,262) | 0.86 | 0.89 |
| sensory | 1.39 (16,557) | 0.74 (16,557) | 0.53 (16,557) | 0.40 | 0.46 |
| central | 2.91 (31,782) | 1.99 (31,782) | 1.70 (31,782) | 0.99 | 0.98 |
| optic | 4.12 (72,582) | 2.41 (72,582) | 1.92 (72,582) | 1.13 | 1.13 |
| visual_projection | 4.60 (7,321) | 2.48 (7,321) | 2.05 (7,321) | 0.99 | 0.98 |
| descending | 1.96 (1,316) | 1.53 (1,316) | 1.31 (1,316) | 1.03 | 1.04 |
| ascending | 0.23 (1,849) | 0.16 (1,849) | 0.14 (1,849) | 0.65 | 0.58 |
| motor | 1.30 (805) | 1.17 (805) | 1.06 (805) | 0.35 | 0.40 |

Values are FlyWire / BANC (number of BANC neurons). Ascending and motor include VNC compartments that FlyWire (brain only) lacks. The two last columns multiply the BANC input synapses of each neuron by its factor `s` before the ratio (arm N); the clip makes classes with a ratio outside [0.25, 4] stay off 1.

## 3. Arm N factors `s` (`connectome/innorm.py`, computed from data only)

### banc888_full_v2

90,689 of 158,262 neurons have a homolog type (7,351 types); 67,573 use the super-class median. Clipped: 1,866 low (0.25), 6,272 high (4), 5.1 % of neurons. Global median of homolog s: 2.11. s of MN9: 2.49, LB3: 2.17. SHA-256 of s: `c4996ada8168a8fc…`.

| super class | n | mean s | q05 | q25 | median | q75 | q95 | homolog n | fallback n | clipped low / high |
|---|---|---|---|---|---|---|---|---|---|---|
| optic | 72,582 | 2.16 | 1.52 | 1.87 | 1.98 | 2.17 | 3.54 | 45,930 | 26,652 | 0 / 2,983 |
| central | 31,782 | 2.24 | 0.90 | 1.61 | 2.22 | 2.62 | 4.00 | 28,408 | 3,374 | 11 / 2,457 |
| sensory | 16,557 | 1.85 | 1.04 | 1.80 | 1.80 | 1.80 | 2.80 | 8,235 | 8,322 | 0 / 132 |
| ventral_nerve_cord_intrinsic | 12,851 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 0 | 12,851 | 0 / 0 |
| visual_projection | 7,321 | 2.79 | 1.85 | 2.32 | 2.70 | 3.18 | 4.00 | 6,734 | 587 | 0 / 649 |
| ascending | 1,849 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 25 | 1,824 | 1,842 / 0 |
| descending | 1,316 | 1.47 | 0.64 | 1.29 | 1.41 | 1.55 | 2.69 | 814 | 502 | 9 / 10 |
| motor | 805 | 3.34 | 3.35 | 3.35 | 3.35 | 3.35 | 3.35 | 33 | 772 | 0 / 10 |
| sensory_ascending | 516 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 1 | 515 | 0 / 0 |
| visual_centrifugal | 472 | 1.94 | 1.01 | 1.58 | 1.59 | 2.19 | 4.00 | 467 | 5 | 1 / 27 |
| glia | 458 | 2.17 | 2.17 | 2.17 | 2.17 | 2.17 | 2.17 | 1 | 457 | 0 / 0 |
| visceral_circulatory | 221 | 2.85 | 2.84 | 2.84 | 2.84 | 2.84 | 2.84 | 7 | 214 | 0 / 3 |
| sensory_descending | 13 | 0.31 | 0.25 | 0.30 | 0.30 | 0.30 | 0.44 | 10 | 3 | 3 / 0 |
| trachea | 8 | 2.29 | 2.29 | 2.29 | 2.29 | 2.29 | 2.29 | 1 | 7 | 0 / 0 |
| not_a_neuron | 7 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 0 | 7 | 0 / 0 |
| ascending_visceral_circulatory | 5 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 2.11 | 0 | 5 | 0 / 0 |

### banc888_full_v3

90,688 of 158,262 neurons have a homolog type (7,350 types); 67,574 use the super-class median. Clipped: 1,873 low (0.25), 3,021 high (4), 3.1 % of neurons. Global median of homolog s: 1.65. s of MN9: 2.04, LB3: 1.49. SHA-256 of s: `e4adf42e32f9f88c…`.

| super class | n | mean s | q05 | q25 | median | q75 | q95 | homolog n | fallback n | clipped low / high |
|---|---|---|---|---|---|---|---|---|---|---|
| optic | 72,582 | 1.72 | 1.22 | 1.45 | 1.54 | 1.69 | 2.89 | 45,930 | 26,652 | 0 / 953 |
| central | 31,782 | 1.92 | 0.78 | 1.38 | 1.73 | 2.29 | 4.00 | 28,407 | 3,375 | 15 / 1,713 |
| sensory | 16,557 | 1.14 | 0.62 | 1.05 | 1.05 | 1.05 | 2.04 | 8,235 | 8,322 | 2 / 39 |
| ventral_nerve_cord_intrinsic | 12,851 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 0 | 12,851 | 0 / 0 |
| visual_projection | 7,321 | 2.30 | 1.52 | 1.83 | 2.26 | 2.66 | 3.73 | 6,734 | 587 | 0 / 278 |
| ascending | 1,849 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 25 | 1,824 | 1,841 / 0 |
| descending | 1,316 | 1.24 | 0.52 | 1.08 | 1.18 | 1.33 | 2.11 | 814 | 502 | 11 / 6 |
| motor | 805 | 2.64 | 2.62 | 2.62 | 2.62 | 2.62 | 2.62 | 33 | 772 | 0 / 10 |
| sensory_ascending | 516 | 0.28 | 0.28 | 0.28 | 0.28 | 0.28 | 0.28 | 1 | 515 | 0 / 0 |
| visual_centrifugal | 472 | 1.69 | 0.89 | 1.35 | 1.41 | 1.88 | 3.25 | 467 | 5 | 1 / 21 |
| glia | 458 | 1.49 | 1.49 | 1.49 | 1.49 | 1.49 | 1.49 | 1 | 457 | 0 / 0 |
| visceral_circulatory | 221 | 2.31 | 2.29 | 2.29 | 2.29 | 2.29 | 2.29 | 7 | 214 | 0 / 0 |
| sensory_descending | 13 | 0.36 | 0.25 | 0.32 | 0.39 | 0.39 | 0.43 | 10 | 3 | 3 / 0 |
| trachea | 8 | 2.13 | 2.13 | 2.13 | 2.13 | 2.13 | 2.13 | 1 | 7 | 0 / 0 |
| not_a_neuron | 7 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 0 | 7 | 0 / 0 |
| ascending_visceral_circulatory | 5 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 1.65 | 0 | 5 | 0 / 0 |

## 4. Antennal-lobe leading eigenvalue (ALLN + ALPN subnetwork, reference sign rule; synapse units)

| graph | signed | excitatory only | ALLN / ALPN |
|---|---|---|---|
| FlyWire v783 | 2,184 | 2,408 | 429 / 685 |
| codex_export_ge3 | 1,806 | 1,930 | 429 / 699 |
| banc888_full_v2 | 1,912 | 2,037 | 429 / 699 |
| banc888_full_v2 after N | 1,999 | 2,177 | 429 / 699 |
| banc888_full_v3 | 2,248 | 2,409 | 429 / 699 |
| banc888_full_v3 after N | 1,818 | 2,011 | 429 / 699 |
