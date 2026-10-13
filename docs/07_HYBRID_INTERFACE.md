# 07 — Hybrid interface: FlyWire v783 brain -> descending neurons -> BANC v3 VNC (design only)

Status: **design for review; no implementation.** Date 2026-10-13. Decision basis: owner's decision after PR #9 (docs/05) and
the F3a result (`docs/results/f3a_banc_vnc.md`, PR #10). Numbers below are copied from `docs/results/f3_dn_crosswalk.{md,json}`
(data only, `scripts/f3_dn_crosswalk.py`) and `docs/results/f3a_banc_vnc.md`; nothing here is a new measurement. Everything
registered as a *choice* below is a proposal; each needs the owner's decision and, before any run, its own pre-registration.

## 1. What is being connected

```
FlyWire v783 brain (Shiu LIF, validated F0)  --DN spike rates, per type and side, 1 interval late-->
BANC v3 VNC (Shiu LIF reference parameters)  --motor-neuron rates (label VNC)--> wing / haltere / neck / leg MN readout
```
Both datasets are female brains; the VNC half is a different individual from the brain half (wiring variability is a
property of the hybrid, not an error).

## 2. DN type crosswalk (FlyWire v783 <-> BANC v3)

Matching key: `cell_type` exactly; where FlyWire splits a type by a trailing letter (DNg02_a ... DNg02_h) the **family**
(type without a single trailing `_<letter>`) is the unit. BANC has `DNg02` (32 neurons) plus `DNg02_d` and `DNg02_h`.

| quantity | value |
|---|---|
| DN neurons | FlyWire 1,299; BANC 1,316 |
| DN families | FlyWire 454; BANC 537; shared by name 348 (1,064 FlyWire neurons, 927 BANC neurons) |
| shared families passing the symmetry gate in **both** datasets | 112 |
| shared families with identical L/R neuron counts | 187 |
| BANC DN families with direct synapses onto wing / haltere / neck / TTMn MNs | 386 |
| ... of which without a same-name FlyWire family | 132 |
| ... of which shared and gate-passing in both | 85 |

Types that matter for the first hybrid (per side L/R neurons; gate = |log2(R/L)| of input synapses < 0.5, total and
excitatory, on each dataset's own graph):

| family | FlyWire L/R | BANC L/R | gate FlyWire | gate BANC | direct synapses in BANC -> wing |
|---|---|---|---|---|---|
| DNg02 | 13/12 | 17/18 | **FAIL** (+0.58) | pass (-0.25) | 2,223 |
| DNa02 | 1/1 | 1/1 | pass (+0.10) | pass (+0.13) | 82 (neck 136, leg 710) |
| DNp01 (GF) | 1/1 | 1/1 | pass (+0.25) | pass (-0.10) | 13 (TTMn 17) |
| DNp31 | 1/1 | 1/1 | pass | pass | 1,895 |
| DNa04, DNa10, DNa16, DNa15 | 1/1 each | 1/1 each | pass | pass | 558 / 618 / 201 / 314 |

The full table is in `f3_dn_crosswalk.md`. Findings that shape the design:
1. **Name coverage is incomplete in the direction that matters.** 132 of the 386 BANC DN families that synapse directly onto
   wing, haltere, neck or TTMn motor neurons have no same-name FlyWire family (e.g. the DNnt-series, DNxn146/148). They
   cannot receive a brain rate by name. The strongest neck-MN drivers in BANC are partly in this group (DNnt019/020).
2. **DNg02 fails the symmetry gate in FlyWire (R/L +0.58) and passes in BANC.** Any left/right DNg02 readout must come from
   the BANC side only; on the FlyWire side the family is usable only bilaterally averaged.
3. **DNp31 is the largest named single-pair wing driver in BANC** (1,895 direct synapses to wing MNs) and was not driven
   in F3a; it passes the gate in both datasets and is a candidate for the first hybrid type set.

## 3. Rate transfer

Proposal (all `E`, engineering scaffold; none is fitted):
- **Unit and statistic.** At every coupling interval k (5 ms, docs/05 layer table) the FlyWire engine reports spikes per DN
  neuron over a trailing window W. Per matched unit u = (type or family, side) the brain-side rate is the mean over the
  FlyWire neurons of u: rho_u(k) in Hz per neuron. No total-spike conservation: a BANC unit with more neurons than its
  FlyWire counterpart (DNg02: 17/18 against 13/12) gets proportionally more total drive. Alternative (total conservation)
  is a registered sensitivity arm, not the default.
- **Re-synthesis.** Every BANC neuron of u receives an independent Poisson process of rate g_u * rho_u(k-1), g_u = 1.
  Spike-timing correlations among DNs (synchrony) are lost by construction; this is the largest structural simplification
  and is stated in every report.
- **Latency.** The rate measured in interval k is applied in interval k+1 at the earliest (`dn_bridge.latency_intervals = 1`
  in the configuration and in the record). 0 is the labelled ACAUSAL variant used only as a control (docs/05 latency
  test). FlyWire -> BANC axonal conduction is not modelled beyond this.
- **Matching order.** Exact `cell_type` first (so BANC `DNg02_d` takes FlyWire `DNg02_d`), then family, then silent. Side:
  left <-> left, right <-> right; `center` neurons (8 in FlyWire) are matched to same-name centre neurons only.
- **Unmatched BANC DNs** are silent (rate 0). **Unmatched FlyWire DNs** are dropped; each run records the fraction of
  FlyWire DN spikes that found no BANC unit (per type), so the loss is a reported number.
- **Gated readouts.** A left/right readout is used only for types passing the symmetry gate in **both** datasets (112 shared
  families; a candidate set for steering). Other types are driven per side but enter only bilateral readouts.
- **Labels (CLAUDE.md rule 1).** DN rates from the FlyWire brain: `BRAIN`; MN rates from the BANC VNC: `VNC`. The GF escape
  (see risk 6) cannot be a `BRAIN`/`VNC` term and would need a `HAND` or `REFLEX` label.

## 4. Risks (ordered by how likely they stop the project)

1. **Operating range mismatch.** In F3a, DNg02 at 20 Hz per neuron gives 0 Hz in DLM; the wing MNs are recruited between
   50 and 100 Hz (network rate 0.017 -> 0.18 Hz). If the FlyWire brain's DNg02 rates under realistic drive stay below ~50 Hz,
   the hybrid wing MNs stay silent. This is checkable before any implementation from existing F0 records: the per-neuron
   DNg02 rate distribution. Not assumed here.
2. **Lateralised VNC response.** Under symmetric DNg02 drive DVM3a-b, b3, hg1 and hg4 fired on the right only (10/10 seeds
   right, 0-3/10 left); DNa02 left/right drive left the wing steering MNs silent. A steering readout from the BANC VNC may
   have no usable left/right contrast at the single-neuron DNa02 level; DNa02 is currently the planned primary yaw drive
   (docs/05). Possible cause: the extra right-sided DNg02 subtypes and proofreading state; this is a data property.
3. **Activity persists after the cut.** At DNg02 100 Hz the network stays at ~0.18 Hz for the whole 200 ms cut (below the
   registered 1 Hz ignition threshold). With causal coupling this appears as motor output after the brain's DN rates have
   dropped; its duration and carriers are unmeasured. A longer-cut record with MN rates is needed before closed loop.
4. **Name-matching holes** (section 2 finding 1): the descending drive of neck and haltere MNs is partly unreachable.
5. **Shuffle/controls (rule 7).** Controls needed: DN->type map permuted within DN class (activity-matched), degree-matched
   random VNC, and an ablation (DN bridge off). Not yet designed in detail.
6. **GF -> TTMn.** 17 direct synapses, no gap junctions in the model; F3a met expectation (d): TTMn 0 Hz, DLM/DVM 0 Hz under GF
   drive (0 direct GF -> DLM/DVM synapses). Take-off via the giant fibre is outside what the spiking path can produce.
7. **Two individuals.** Brain and VNC come from different animals; counts differ (DNg02 13/12 vs 17/18). Reported, not
   corrected.
8. **Rate coding only.** The interface passes rates; bursts and synchrony in the DNs are not represented.

## 5. Questions for the owner (answers change what is built)

1. Unit of matching: family (proposed) or exact type only (drops DNg02_a..g from the brain side)?
2. Rate statistic: per-neuron mean (proposed) or total conservation as default?
3. First type set: DNg02 + DNa02 + DNp31 only, or all 112 gate-passing shared families?
4. Before implementation, run the two checks of risks 1 and 3 (F0 DNg02 rate distribution; a longer-cut F3a record with MN
   rates), each under its own pre-registration?

## 6. What implementation would add (not started)

Modules `coupling/dn_bridge.py` (mapping table + rate transform, config-recorded) and a VNC engine instance on `banc888_full_v3`
(~2.7 GB RAM measured in F3a; with the FlyWire engine ~3.5 GB, total below the 6 GB guard only on a >= 12 GB machine);
pre-registration F3b before any coupled run: criteria on MN rates and the symmetry-gated readouts of section 3, gates 1-3'
of `prereg/06`, controls of risk 5.
