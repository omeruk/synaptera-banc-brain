# Side-convention audit: post-audit recomputation (prereg/13 section 3)

Label: **post-audit**. Existing records re-analysed with the motor-neuron sides of the owner's rule; nothing was rerun. Mapping: docs/08 section 10 v2 `lateral` (b2+, b3-, hg1-, hg2-, hg3-, hg4-, i1-, i2-), gain G1.

Swapped readout types (keys exchanged): ADNM2, CvN6, CvN9, DLM5, PSI. Not recomputable (mixed; per-neuron rates not recorded): ADNM1, CvN7.

| record | version | S mean (Hz) | sd | p one-sided | p two-sided | seeds +/- | random abs >= | perm p | turn at G1 (deg) | bridge_off S (p) |
|---|---|---|---|---|---|---|---|---|---|---|
| f3b3 | pre-audit | -1.02 | 3.38 | 0.9044 | 0.1911 | 5 / 14 | 32 / 39 | 0.825 | -0.40 | -0.03 (0.3299) |
| f3b3 | post-audit | -1.02 | 3.38 | 0.9044 | 0.1911 | 5 / 14 | 32 / 39 | 0.825 | -0.40 | -0.03 (0.3299) |
| mv2_R | pre-audit | -4.10 | 2.90 | 0.9992 | 0.0016 | 1 / 9 | 7 / 19 | 0.400 | -1.62 | +0.00 (1.0000) |
| mv2_R | post-audit | -4.10 | 2.90 | 0.9992 | 0.0016 | 1 / 9 | 7 / 19 | 0.400 | -1.62 | +0.00 (1.0000) |

- `f3b3` (runs/f3b3/f3b3_v3.json): post-audit identical to pre-audit: yes.
- `mv2_R` (runs/mv2/t3_R.json): post-audit identical to pre-audit: yes.

Per-type T = mean over seeds of D_m (Hz) and contribution to S = -s_m T / 2, post-audit:

| type | f3b3 T | f3b3 contribution | mv2_R T | mv2_R contribution |
|---|---|---|---|---|
| b2 | -1.55 | +0.78 | -3.50 | +1.75 |
| b3 | +2.80 | +1.40 | +1.70 | +0.85 |
| hg1 | +2.75 | +1.38 | -1.60 | -0.80 |
| hg2 | +0.00 | +0.00 | +0.00 | +0.00 |
| hg3 | -0.20 | -0.10 | +0.10 | +0.05 |
| hg4 | -11.65 | -5.83 | -15.60 | -7.80 |
| i1 | +2.75 | +1.38 | +3.40 | +1.70 |
| i2 | -0.05 | -0.03 | +0.30 | +0.15 |
| iii1 | +0.00 | - | +0.00 | - |
| iii4 | -2.25 | - | -5.00 | - |
| ps1 | -0.30 | - | -3.20 | - |
| ps2 | -0.05 | - | +0.00 | - |
| tp1 | +0.00 | - | +0.00 | - |
| tp2 | -0.20 | - | -2.10 | - |
| tpn | +0.00 | - | +0.00 | - |
