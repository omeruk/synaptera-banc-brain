# 09. Neuron model v2 (open loop): adaptation, conductance synapses, electrical synapses

Status: design, written 2026-10-14 before any v2 run. The criteria are in `prereg/12_neuron_model_v2.md`.
Decision context: the project owner's instruction after PR #17 (recorded in `docs/05_ARCHITECTURE_PLAN.md`).

## 1. Why

Every result so far (F0-F3, docs/08) uses the Shiu et al. 2023 LIF network. Its neurons have:
- no adaptation;
- current-based synapses: a synapse adds a fixed voltage, whatever the membrane potential;
- no electrical synapses.

Three consequences follow:
- A neuron driven hard fires at a rate set only by its refractory period.
- Inhibition can push the membrane potential below any physiological reversal potential.
- The fastest pathway in the fly, the giant fiber (GF) system, is mostly electrical but is chemical-only in the model.

v2 adds the three mechanisms as named variants and asks two questions:
- Do the existing open-loop results survive?
- Does the hybrid optomotor signal change (docs/08: weakly anti-directional, not significant)?

Nothing is tuned on behaviour (CLAUDE.md rule 3). Every new constant comes from the literature or from a
single-synapse criterion fixed now. The reference arm R is not modified (rule 8).

## 2. The variants (`engine/model_variants.py`, switched on by `Brian2LIFEngine(..., model_variant=...)`)

The Shiu neuron, in its own units:
- t_mbr dv/dt = v_0 - v + g;
- tau dg/dt = -g, and g += n_syn * w_syn when a presynaptic spike arrives (after t_dly);
- threshold v_th, reset to v_rst, refractory period t_rfc.

All Shiu parameters stay as they are in every arm: v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms,
t_rfc = 2.2 ms, t_dly = 1.8 ms. So do the sign rule and the Poisson input path (v += w_syn * f_poi, no delay, no
refractory period for the targets).

### 2.1 `adapt`: AdEx adaptation
The owner's equation:
- dw/dt = (a (V - V_rest) - w) / tau_w;
- w += b at each spike.

The adaptation current is subtracted from the membrane equation. In the voltage units of the Shiu equation (current /
g_L):
- t_mbr dv/dt = v_0 - v + I_syn - w_ad;
- tau_w dw_ad/dt = a' (v - v_0) - w_ad;
- w_ad += b' at each spike.

The values (origin `P`):

| constant | value | source |
|---|---|---|
| a' = a / g_L | 4 nS / 30 nS = 0.1333 | Brette & Gerstner 2005, J Neurophysiol 94:3637, Table 1 |
| b' = b / g_L | 0.0805 nA / 30 nS = 2.683 mV | same |
| tau_w | 144 ms | same |

- **Why a general value.** No fitted AdEx (or other adaptation) parameter set for *Drosophila* central neurons was found.
  The owner's rule then asks for a cited general value. Brette & Gerstner fitted these values to a regular-spiking
  cortical neuron model.
- **Why divide by g_L.** It keeps the paper's ratios (adaptation conductance relative to leak, spike-triggered jump in
  mV) without importing its capacitance or its membrane time constant (9.4 ms, not Shiu's 20 ms).
- **What is not added.** The exponential spike term of AdEx. The Shiu threshold and reset stay.
- **Order of magnitude.** At a steady rate f, the mean w_ad is about b' tau_w f, i.e. about 0.39 mV per Hz.

### 2.2 `cond`: conductance synapses
- **Equation.** I_syn = g_e (E_exc - v) + g_i (E_inh - v). g_e and g_i are relative to the leak (dimensionless) and
  decay with the Shiu tau (5 ms). A presynaptic spike adds n_syn * dg_exc to g_e (excitatory presynaptic neuron) or
  n_syn * dg_inh to g_i (inhibitory), after t_dly.

| constant | value | origin | source |
|---|---|---|---|
| E_exc | 0 mV | P | owner's value; cation (nicotinic) reversal |
| E_inh | -74 mV | M | chloride reversal of adult *Drosophila* flight motoneuron MN5, -74 +/- 4 mV; Ryglewski et al. 2017, Neuron 93:632, Fig. 4C,E |
| dg_exc | 0.0052912003 | F | single excitatory synapse: peak PSP at rest = Shiu peak PSP (0.04331 mV) |
| dg_inh | 0.0125153142 | F | single inhibitory synapse: peak PSP at rest = Shiu peak PSP |

- **E_inh.** It is used for both GABA (Rdl) and glutamate (GluCl) synapses; both are chloride channels.
  - It was measured in one identified cell type and is applied to every neuron.
  - Ryglewski et al. report the MN5 rest at -63 mV, so E_Cl lies 11 mV below that rest. The model's rest is -52 mV, so
    the inhibitory driving force at rest is 22 mV.
  - An Antennal-lobe whole-cell value exists (E_GABA about -40 mV in local neurons, Wilson & Laurent 2005 supplement).
    It is not used, because the authors attribute it partly to the pipette solution.
- **dg calibration.** `calibrate_dg` integrates the single PSP (RK4, 1 us) and bisects; no network run is involved.
  The values agree with the small-PSP limit w_syn / |E - v_0| within 0.2 %.
- **Effect.** One synapse at rest acts exactly as in R. Many coincident synapses saturate toward E_exc or E_inh, and
  inhibition becomes shunting near E_inh.
- **Integration.** Exponential Euler at the 0.1 ms clock. The v equation is no longer linear, so Brian2's exact
  `linear` method no longer applies.

### 2.3 `gap`: electrical synapses (`configs/gap_junctions.yaml`)
- **Used pairs.** Only pairs with published physiological evidence:
  - GF -> TTMn and GF -> PSI (BANC types DNp01, TTMn, PSI).
  - Both are heterotypic ShakB junctions. They rectify: depolarisation passes from the GF side.
  - The TTMn response follows the GF with < 1 ms latency, and the shakB2 mutant loses this (Phelan et al. 2008, Curr
    Biol 18:1955, Figs. 2-4; Tanouye & Wyman 1980).
- **Pairing rule (`E`, fixed from anatomy now).** A GF is coupled to every TTMn / PSI on which it has >= 1 chemical
  synapse.
  - On BANC this gives left GF -> left TTMn, right GF -> right TTMn, and each GF -> both PSIs.
  - The GF has chemical synapses on both PSIs (3-4 synapses each), and the sources do not say whether the junction is
    ipsi- or bilateral. This is listed for the owner.
- **Junction model.** The point neuron has no spike waveform. A coupling conductance would therefore transmit only
  subthreshold voltage, and the GF, an input node driven by Poisson kicks, sits at rest between spikes. So the junction
  is reduced to its measured effect, 1:1 sub-millisecond transmission:
  - each GF spike raises the postsynaptic v by 68.75 mV (w_syn * f_poi, the model's own suprathreshold kick; origin
    `E`);
  - without delay;
  - in one direction only.

  The chemical GF synapses stay as they are.
- **Not used (listed for the owner).** GF-GCI (no annotation), GF-GFC1..4 (dye coupling only), antennal-lobe PN-eLN
  coupling (class-level evidence), lobula plate tangential cells (blowfly evidence).
- **FlyWire has no registered pair:** its GF has no VNC part. So `gap` changes nothing in the FlyWire brain, and ACG
  equals AC there. ACG is still run on FlyWire, and the identity is checked.

### 2.4 Arms
| arm | switches |
|---|---|
| R | none (Shiu reference, unchanged) |
| A | adapt |
| AC | adapt + cond |
| ACG | adapt + cond + gap |

The same arm runs on the FlyWire v783 brain and on the BANC v3 VNC subgraph of prereg/08. In the hybrid both halves use
the same arm.

## 3. Tests (criteria in prereg/12)
1. **(i) FlyWire does not break.**
   - Drive: the 36 labellar sugar GRNs at 100 Hz for 1 s, then a 200 ms cut; 50 seeds.
   - Pass: MN9 rate >= 10 % of R's, and ignition after the cut no more frequent than in R.
   - A failing arm is reported and dropped from (ii)-(iv).
2. **(ii) BANC VNC**, the prereg/08 protocol (R1 recruitment, R2 persistence, R3 steering dose-response of
   DNa04 / DNa10 / DNa15, gates). The headline is the lowest steering rate per family next to R.
3. **(iii) Hybrid optomotor**, the prereg/11 protocol.
   - Setup: fresh brain seeds 30-39, the 112-family ensemble, `bridge_off`, and 19 random wirings.
   - Primary: S under the v2 `lateral` mapping (Lindsay signs, gain G1). It must be syndirectional and significant
     (one-sided, Holm over arms), and |S(ensemble)| must exceed all 19 random wirings.
4. **(iv) GF -> TTMn.** GF at 100 Hz; ACG vs R. This mainly checks that the `gap` implementation reproduces the
   literature property it was built from. It is not an independent finding.

## 4. Run time
- **Measured before the prereg:** wall time only, on a non-registered condition (seed 9999); prereg/12 section 9.
- **Outcome:** 4 arms; the total is in prereg/12.
- **Owner's rule:** if the projected time exceeds 6 h, only R and ACG run, and the owner is told.

## 5. Limits stated now
- The adaptation parameters are a cortical fit applied to every neuron of both datasets. *Drosophila* neurons differ by
  type: some adapt strongly, some barely.
- One E_inh for every neuron, measured in one motoneuron.
- The gap junction is a kick, not a conductance; it is valid only for junctions known to transmit spikes 1:1.
- Activity-matched shuffled graphs and degree-matched random graphs (rule 7) are not part of this open-loop model
  comparison, as in prereg/06-11. The F3b-3 random wirings and `bridge_off` are the controls of test (iii).
