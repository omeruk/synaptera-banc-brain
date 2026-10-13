# 05 — Architecture plan for the next study

**Working title:** *Synaptera-2 — BANC-based, with a nerve cord (VNC), causal and trainable flight*
Date: 2026-10-12. Basis: `01_CODE_REVIEW.md` (bugs), `02_CONNECTOME_MODEL_PHYSICS_TESTS.md` (structural analysis),
`03_BANC_OPPORTUNITIES.md`, `04_RESEARCH_NOTES.md`.

---

## 1. Research questions and definition of success

| RQ | question | output that counts as a "result" |
|---|---|---|
| RQ1 | Can BANC's VNC paths replace the hand-made DN → wing bridge? | in open loop, DN stimulation → correctly signed responses in the right wing MN groups; in closed loop, hover sustained without the bridge |
| RQ2 | Is the flight attitude reflex (haltere/ocelli/HS → wing steering MNs) stable in closed loop from the connectome? | pre-registered criteria for a ±30° yaw/roll perturbation against the hand-made PD (like Synaptera b1–b3) |
| RQ3 | Can the feeding trigger come by the natural path (tarsal GRN → VNC → AN → SEZ → MN9)? | removal of the "HAND-MADE trigger" label |
| RQ4 | Does a connectome with fixed topology and trained dynamic parameters learn flight tasks, and is the real topology better than control graphs? | task success in the TRAINED arm + a difference against rewired/random/MLP controls in ≥3 seeds |
| RQ5 | Can the take-off decision (looming → DNp01 → TTM) and landing come from the brain? | at least one of the take-off/landing phases moves from HAND to BRAIN/VNC |

Each RQ is pre-registered separately; if one fails the others continue (a negative result is also an output).

## 2. Principles kept and new

**Kept from Synaptera:** one source label per motor/sensory term and separate recording; pre-registration + verification
script; negative results reported with equal weight; data pinned by hash; DEV subnets are never results.

**New:**
1. **Extended label set:** `BRAIN` (brain), `VNC` (nerve-cord connectome), `HAND`, `REFLEX` (hand-made), `TRAINED`,
   `FLYVIS`. In addition a **parameter origin class** (from Fly.exe): `M` measured, `P` population prior, `F` fitted (what it
   was fitted to is written down), `E` engineering scaffold, `I` irrecoverable.
2. **Causal timing:** the body is always one coupling interval behind the neural engine (command queue); the latency is
   explicit in the configuration and in the HDF5.
3. **Quality gates (pre-registered):** L/R symmetry (|log2 R/L| < 0.5), proofreading status, in/out degree, path (hop)
   existence — no readout or path is used in an experiment before it passes the gate.
4. **External pre-registration:** OSF (or equivalent) + the in-repository record.
5. **Meaningful seeds:** physical noise (gusts, sensor noise, start pose) changes with the seed; ≥5 seeds.
6. **Controls in every experiment:** activity-matched shuffled graph, random graph with the same degrees, ablation; in
   training also linear + MLP.

## 3. Layered architecture

```
┌──────────────────────── 6. Experiment / task layer (tasks/, prereg/) ───────────────────────┐
│ task ladder T0..T6, criteria, seeds + physical noise, evaluation, report/verify               │
└───────────────▲───────────────────────────────────────────────────────────┬────────────────┘
                │ measurement/record                                         │ config (YAML + hash)
┌───────────────┴──────┐  state   ┌───────────────────────────┐            │
│ 1. Body / physics    │────────▶ │ 3. Sensory interfaces     │            │
│ MuJoCo: flybody or   │          │ vision: FlyVis + L3 lum.  │            │
│ NeuroMechFly;        │          │ haltere/wing CS, JO,      │            │
│ wing model           │          │ tarsal GRN, (olfaction)   │            │
└──────────▲───────────┘          └─────────────┬─────────────┘            │
           │ force/torque, joints             rate/current (labelled)      │
┌──────────┴───────────┐                        ▼                          │
│ 4. Motor interfaces  │ MN activity   ┌──────────────────────────────────┐ │
│ wing power/steering  │◀───────────── │ 2. Neural engine (engine/)       │◀┘
│ MN → stroke; haltere,│   (queue:     │  graph: BANC (+FlyWire783,       │
│ leg, neck, MN9       │   1 interval  │   MaleCNS adapters)              │
└──────────────────────┘   latency)    │  (a) Shiu LIF spiking —          │
                                        │      "unmodified model" arm      │
                                        │  (b) rate + trainable gain/leak  │
                                        │      (PyTorch, GPU)              │
                                        │  (c) conductance+adaptation+gap  │
                                        └──────────────▲───────────────────┘
                                                       │
                                        ┌──────────────┴───────────────────┐
                                        │ 0. Connectome data layer         │
                                        │ common schema, type crosswalk, QA│
                                        └──────────────────────────────────┘
```

### 3.0 Connectome data layer (`connectome/`)
- **Adapters:** `banc.py` (Codex export or Dataverse 7WTH1N: `banc_888_edgelist_simple_v3.feather`, `banc_888_meta.feather`,
  NT predictions), `flywire783.py` (the existing Shiu files), `malecns.py` (optional, for comparison).
- **Common schema:** `neurons(id, cell_type, cell_class, super_class, side, region∈{brain,SEZ,neck,VNC}, nt, nt_conf, soma_xyz, status)`,
  `edges(pre, post, n_syn, sign)`. The sign rule is in the configuration (Shiu rule by default; variants labelled).
- **Type crosswalk** (`crosswalk.csv`): FlyWire ↔ BANC (DNs, MN9, LB3 sugar GRNs, HS/VS/H2, LC4/LPLC2, T4/T5, haltere
  afferents, wing MNs); a source column (NBLAST, name, connectivity profile).
- **QA report** (automatic for every data version): L/R symmetry table (generalised `dn_lr_asymmetry.py` from this study),
  degree distributions, neurons without input, excitatory/inhibitory ratios, hop distances and signed influence of the
  critical paths (generalised `connectome_structure.py`).

### 3.1 Neural engine (`engine/`)
Common interface (proposal):
```python
class NeuralEngine(Protocol):
    def reset(self, seed: int) -> None: ...
    def set_inputs(self, drive: dict[str, np.ndarray]) -> None:     # group name -> rate (Hz) or current
    def advance(self, dt_ms: float) -> None:                        # one coupling interval
    def rates(self, groups: list[str], window_ms: float) -> dict[str, np.ndarray]
    def state_dict(self) -> dict: ...                               # replay / audit
```
- **(a) Spiking (reference arm):** Shiu parameters unchanged; Brian2 (CPU) or GeNN/Brian2CUDA (GPU). "Unmodified model"
  claims come only from this arm. Poisson input without delay, as Shiu's `PoissonInput`; a current-injection variant for
  graded cells, labelled.
- **(b) Rate + trainable (TRAINED arm):** the flyhard recipe — `s ← (1−λ)s + λ·tanh(Ŵs + u)`, `Ŵ` = signed, input-normalised
  synapse count × (edge gain); learned: edge gain + neuron leak (+ optionally a gain per cell type). **State carried** (no
  reset). Custom sparse backward (chunked per-edge derivative + stored transpose).
- **(c) Biophysical variant:** conductance-based IF + adaptation + gap junctions (the Uploading Lab approach); to test the
  resting-activity problem (76 % of DNs silent).
- **Equivalence test:** open-loop response correlation between (a) and (b) (e.g. sugar → MN9, looming → DNp01) is reported.

### 3.2 Sensory interfaces (`sense/`)
| sense | source | target neurons (BANC) | note |
|---|---|---|---|
| vision | FlyGym eyes → FlyVis | boundary layer (existing rule) + **L3 luminance pathway** | more sub-frames near obstacles (H4) |
| haltere | body angular velocity (Coriolis) | haltere campaniform afferents (VNC) | critical for the reflex arc; ms scale |
| wing mechanosensors | stroke parameters | wing CS afferents | muscle feedback |
| antenna (JO) | air flow, wind | JO-A/B/C/E | speed/wind sense |
| leg taste (tarsal GRN) | tarsus–food contact | tarsal sugar GRNs (VNC) | RQ3; replaces the HAND shortcut |
| leg proprioception | joint angles/contact | leg CS/chordotonal | landing and walking |
| olfaction | 3D odour field | ORNs (antennal lobe) | off until the AL lock-up is solved (see §6) |

Each mapping is a `Transduction` object: input quantity → rate; its parameters carry an origin class (`M/P/F/E/I`).

### 3.3 Motor interfaces (`motor/`)
- **Wing:** power-muscle MNs (DLM/DVM; wingbeat frequency/amplitude) + steering MNs (b1–b3, i1–i2, iii1–iii3, hg, tp;
  amplitude asymmetry, stroke deviation). `MuscleModel`: MN rate → activation (first-order filter) → change of stroke
  parameter; parameters **only from literature physiology** (open loop), never tuned on behaviour; physical limits
  (Φ ≤ ~180°).
- **Haltere MNs, neck MNs** (the head reflex can stop being hand-made), **leg MNs** (TTM take-off, landing extension),
  **MN9** (proboscis; a physical effect if a proboscis joint is added to MuJoCo).
- **Bridge arm:** the existing hand-made DN → stroke bridge stays as a **control arm** (for comparison).

### 3.4 Body/physics (`body/`)
- **Option A (recommended start):** the existing FlyGym/NeuroMechFly + an improved stroke-averaged model (amplitude limit,
  stroke-plane angle, anisotropic damping). Fast, compatible with the existing code.
- **Option B (later):** the flybody (Vaxenburg 2025) wing-jointed fluid model + pattern generator; MN → modulation of the
  pattern generator. More realistic but a small time step and expensive.
- Physics dt 0.05–0.1 ms; **coupling interval 5 ms** (25 ms is too coarse for reflexes); vision 100–200 Hz.
- A (seeded) physical-noise generator: gusts, sensor noise.

### 3.5 Coupling and timing
```
for every 5 ms interval k:
    sensory_k = sense(body_state_k)               # input to the neural engine
    engine.advance(5 ms)                          # neural state k → k+1
    mn_k      = engine.rates(motor_groups)        # MN output of this interval
    body.step(motor(mn_{k-1}), 5 ms)              # body ONE INTERVAL BEHIND (causal)
    record(k)
```
In Brian2 the overhead of a 5 ms `run()` call is high → couple inside one `run()` with a `network_operation`, or use
GeNN/PyTorch.

### 3.6 Recording, replay, verification
- The existing HDF5 schema extended: every term + label + origin class; MN rates; the latency parameter; hash of the QA report.
- **Replay audit** (Synaptera already has it, "replay exact"): with the recorded inputs the neural engine is bit-identical.
- `verify_report.py`: recomputes the report numbers from the records (existing approach).
- Numeric-tolerance comparison (across platforms).

## 4. Task ladder and phases (gated roadmap)

| phase | work | output | gate |
|---|---|---|---|
| **F0** | Fixes of the v1 design built into the new code: H1 (queue), H2 (symmetry gate), M1, M2, H5, M7 | the ported code + an n1 re-run (same result expected) | n1 criteria pass unchanged |
| **F1** | BANC data pipeline + crosswalk + QA report | QA tables; MN9/DN matches | critical types matched (DNp01, DNg02, DNa02, MN9, haltere afferents, wing MNs); symmetry report published |
| **F2** | Open-loop reproductions (spiking arm): sugar→MN9, tarsal GRN→MN9, looming→DNp01→TTM, haltere→steering MN, HS→DN | 5 pre-registered tests | ≥3/5 pass; failures reported as negative findings |
| **F3** | Wing motor interface + muscle model; haltere reflex in closed loop during hover (RQ1, RQ2) | REFLEX → VNC label change (or negative) | ±30° perturbation criteria (like b1–b3), ≥5 seeds |
| **F4** | Take-off (GF/TTM), landing, natural feeding trigger (RQ3, RQ5) | per-phase label table | separate criteria per phase |
| **F5** | TRAINED arm: rate engine, BC + DAgger, tasks: hover → optomotor → obstacle → landing (RQ4) | learning curves, control arms | ≥3 seeds; comparison with controls pre-registered |
| **F6** | Full task: take-off → towers → landing → feeding; video; paper | final report | — |

**Planned F2 items (added 2026-10-12, after the F0 statistics):**
- **Latency effect with DNa02:** hover + yaw perturbation (the v1 `flight_v30` protocol) with DNa02 as the yaw drive
  (it passes the symmetry gate; DNp15 does not), latency 0 (labelled ACAUSAL) vs 1 interval, **>= 10 seeds per
  latency**, criteria written to `prereg/` before any run. Control arm: `zero_turn` (constant zero yaw command,
  readouts recorded), not the perch-baseline ablation (fragile: `docs/results/f0_latency_hover_yaw.md`).
- **Ignition frequency** of the self-sustaining olfactory mode is reported alongside every F2 result that uses a
  perch + cut protocol (`docs/results/f0_ignition_frequency.md`).

**Findings from the F2 prep calibration (2026-10-13; `docs/results/f2_wsyn_calibration.md`, PR #6; numbers from the records by script):**
- **Uniform w_syn scaling cannot make BANC MN9 respond without self-sustained activity after the cut.** Ignited after
  the cut (20 seeds, prereg/02 definition): A `wsyn_struct` 20/20, A2 `wsyn_struct_class` 19/20, B `wsyn_fit_mn9` 18/20;
  the reference arm R is 0/20 but silent at MN9. The scale at which MN9 responds is already in the self-sustaining
  regime of the BANC graph. Reported as a negative result; no variant is used for F2.
- **The BANC sugar -> MN9 path is strongly lateralised.** Under A (labellar drive, 100 Hz, 1 s, 3 seeds) MN9 left 32 Hz
  / right 11 Hz; right-sided labellar drive gives 35 / 10 Hz, left-sided drive 6 / 5 Hz, so right labellar drive
  dominates. Any left/right MN9 readout is subject to the symmetry gate (rule 5).
- Variant B's definition fails its FlyWire validation (it does not reproduce w_syn = 0.275 mV).

**Finding from the full BANC v888 graph (2026-10-13; `prereg/04_banc_full_graph.md`, `docs/results/f2_full_graph.md`,
`docs/results/f2_full_graph_qa.md`):** the Codex export threshold explains part of the sparseness but not the silence.
The pair-level v888 edge list (no threshold) among the 158,262 Codex neurons has 11.6 M pairs and 35.4 M synapses (Codex
export: 3.04 M / 23.6 M; FlyWire 15.1 M / 54.5 M); the FlyWire / BANC in-synapse ratio on matched types falls from 2.71
to 1.76. On this graph the unmodified reference arm still leaves MN9 silent (0.0 Hz, all four drives, 3 seeds; gates 1
and 2 pass, gate 3 fails below 0.2; the all-neuron sensitivity set gives the same). Stopped per the registration; the
next route (per-neuron input normalisation or an E/I-preserving scale) needs its own pre-registration.

**Finding from the v3 graph and per-neuron input normalisation (2026-10-13; `prereg/05_banc_v3_and_innorm.md`,
`docs/results/f2_innorm.md`, `docs/results/f2_v3_qa.md`):** no arm qualifies as a candidate BANC reference. The v3 edge
list (`size >= 10`; 13.4 M pairs, 41.9 M synapses among the Codex neurons) lowers the FlyWire / BANC in-synapse ratio to
1.47; its antennal-lobe leading eigenvalue (2,248) equals FlyWire's (2,184). Gates 1 and 2 pass in all three arms (0 / 50
ignitions). Gate 3 fails on the lower bound for the leg drive in every arm (0.10-0.18) and, for the labellar drive, in
R3 (0.11) but not in N (0.35-0.40). MN9 under labellar drive: R3 6.8 Hz, N on v2 6.8 Hz (just above the 5.97 Hz
threshold, 11 % of FlyWire), N on v3 3.5 Hz (no); in R3 and N on v2 the response is carried by one MN9 neuron (left,
13.6 Hz; right 0.0 Hz) although the MN9 input-synapse symmetry gate passes, and the left neuron responds to the
right-sided labellar drive only. Leg drive: MN9 0.0 Hz in all arms. For arm N the leg results carry the registered
VNC-dendrite caveat (ascending neurons sit at the 0.25 clip); R3, without normalisation, shows the same leg result.
Stopped per the registration; any further route needs a new pre-registration and the project owner's decision.

**Decision after prereg/05 (project owner, 2026-10-13, PR #9 merged):**
1. The BANC whole-brain track stops here as a reported result. R3 (v3 graph, unchanged Shiu parameters) is the first BANC
   graph on which MN9 responds (6.8 Hz), but the response is lateralised (left MN9 only, driven by right labellar GRNs),
   the leg pathway is silent and gate 3 fails. This matches the known proofreading asymmetry (optic lobe R/L 1.56, LB3b
   20 L / 40 R). It is a data property, not something to tune.
2. The main track becomes the **hybrid**: FlyWire v783 brain (validated in F0) -> descending neurons matched by type ->
   BANC v3 VNC (wing, haltere, neck and leg motor neurons). Both datasets are female.
3. `banc888_full_v3` is the BANC graph from now on.
4. Next steps: F3a, open loop, BANC VNC only (`prereg/06_banc_vnc_dn_to_mn.md`, PR "F3a: BANC VNC open loop, DN -> motor
   neurons"); design document `docs/07_HYBRID_INTERFACE.md` (no code) for review before any implementation.

**Finding from F3a, BANC v3 VNC open loop (2026-10-13; `prereg/06_banc_vnc_dn_to_mn.md`,
`docs/results/f3a_banc_vnc.md`):** gates 1, 2 and 3' pass (0 spikes without input; 0 / 50 ignitions, CI 0-0.071; maximum
network rate under drive 0.18 Hz) and expectation (a) is a hit, so by the registered rule the BANC v3 VNC is usable for
the hybrid. Bilateral DNg02 drive at 100 Hz drives DLM1-4 (54 Hz), DLM5 (47 Hz), DVM1a-c (24 Hz), tp2 (55 Hz) and ps1
(114 Hz), with a steep recruitment between 50 and 100 Hz. Expectation (b) is a miss: DVM3a-b, b3, hg1 and hg4 respond on
the right side only (10/10 seeds right, 0-3/10 left). Expectation (c) is a miss: unilateral DNa02 drive (one neuron per
side) leaves the wing steering MNs silent (largest |L - R| 0.2 Hz), while neck MNs respond ipsilaterally (CvN1, CvN11,
DProN3a/b). Expectation (d) met: GF drives neither TTMn nor DLM/DVM (17 direct GF -> TTMn synapses; 0 to DLM/DVM). Post-hoc:
at G100 the network activity (0.18 Hz) does not decay within the 200 ms cut (0.18 Hz), below the 1 Hz ignition threshold;
its carriers and duration were not recorded. Stopped there; the interface design is `docs/07_HYBRID_INTERFACE.md`.

**Finding from F3a-2, pre-checks for the hybrid (2026-10-13; `prereg/07_f3a2_rate_persistence_steering.md`,
`docs/results/f3a2_checks.md`):** gates 1, 2 and 3' pass. C1 (existing F0 and v1 screen records, no simulation): DNg02 is
silent in the FlyWire brain in every F0 window and every v1 visual condition (docs/02 confirmed: 0 spikes in all 25
neurons); the six steering families fire at 3.4-73.2 Hz per neuron in F0 flight, mostly below the 50-100 Hz range in which F3a
recruited wing power MNs. C2: after DNg02 drive the activity does not decay within a 1 s cut (about 0.19 Hz network);
wing power MNs stop within 100-200 ms, neck MNs and right b3/hg4 keep firing; post-hoc, about three quarters of the cut
spikes are in BANC brain-region neurons. C3 (DNg02 100 Hz baseline labelled HAND, unilateral 100 Hz steering drive):
DNa04, DNa10 and DNa15 steer by the registered rule (all ipsilateral); DNa02, DNp31 and DNa16 do not. The steering
effects were tested at 100 Hz only, above the FlyWire rates of DNa04, DNa10 and DNa15. Which families go into F3b is the
project owner's decision.

**Decision after F3a-2 (project owner, 2026-10-13, PR #11 merged; the joint Holm correction over both drive sides, 180
tests, is accepted as the stricter reading):**
1. Wing power in the hybrid comes from a `HAND`-labelled tonic DNg02 baseline. C1 justifies this: DNg02 is silent in the
   FlyWire brain in every F0 window and every visual condition.
2. The brain's steering candidates are DNa04, DNa10 and DNa15 (ipsilateral steering in C3). DNa02 is dropped as the
   planned yaw drive: it does not steer in the BANC VNC and is lateralised in FlyWire (L 45 / R 11 Hz). DNp31 is recorded
   as a non-lateral modulator.
3. The hybrid uses a BANC VNC **subgraph**, not the whole BANC graph, because the FlyWire brain replaces the BANC brain.
   It is defined from data only, before any run: neurons whose region is VNC or neck, plus the motor neurons and the DNs
   (as input nodes); ascending neurons kept as outputs and recorded, with their inputs from BANC brain neurons removed.
4. Next step: F3a-3, open loop on that subgraph (`prereg/08_f3a3_vnc_subgraph.md`): recruitment, persistence (registered
   decay criterion) and steering dose-response at 10-100 Hz.

**Finding from F3a-3, BANC VNC subgraph open loop (2026-10-13; `prereg/08_f3a3_vnc_subgraph.md`,
`docs/results/f3a3_vnc_subgraph.md`):** the subgraph keeps 25,818 of 158,262 neurons (3,020,333 pairs, 11,585,331
synapses); no neuron has the neck as its dominant region. Gates 1, 2 and 3' pass. R1: the DNg02 wing power recruitment
(DLM, DVM1, tp2, ps1) is nearly unchanged against the whole graph; the right-only b3/hg1/hg4 of F3a, the haltere MNs and
the neck MNs at G100 are gone (post-hoc: they depended on the removed BANC brain/SEZ part or on the synapses onto DNs).
R2 passes: activity stops within 25 ms of the cut in 10 / 10 seeds (whole graph: no decay within 1 s). R3 (HAND DNg02
baseline): all three families steer ipsilaterally; lowest steering input rate DNa04 25 Hz, DNa15 50 Hz, DNa10 100 Hz.
Only DNa04 steers at a rate inside its FlyWire F0 flight range (L 26.1 / R 16.8 Hz); DNa10 (23.2 / 3.4 Hz) and DNa15
(7.1 / 3.7 Hz) steer only well above theirs. What follows for F3b is the project owner's decision.

**Decision after F3a-3 (project owner, 2026-10-13, PR #12 merged):**
1. The BANC VNC subgraph of prereg/08 is adopted for the hybrid.
2. Persistence is solved on the subgraph: activity stops within 25 ms of the cut (R2, 10 / 10 seeds).
3. DNa04 is the primary steering family: it steers at 25 Hz, inside its FlyWire flight range. DNa15 and DNa10 steer only
   above their FlyWire ranges.
4. Open item, for later: the neck and haltere MN responses are lost on the subgraph. The FlyWire brain may drive the neck
   MNs (through DNs that are not in the current set).
5. Answers to docs/07 section 5: matching by exact `cell_type` first, then family; per-neuron mean rate; latency one
   coupling interval (5 ms); Poisson re-synthesis; everything recorded. DN set: DNa04, DNa10, DNa15 driven from the
   FlyWire brain; DNp31 recorded as a modulator; DNg02 is the `HAND` baseline at 100 Hz.
6. Next step: F3b-1, the first coupled experiment, open loop and without a body (`prereg/09_f3b1_optomotor.md`): the
   FlyWire brain sees rotating gratings, and its DN rates drive the BANC VNC subgraph. Arms `ref` (gain 1) and
   `rate_gain` (per-family gain from F3a-3 and C1), with the controls `bridge_off` and `shuffled`.

**Finding from F3b-1, hybrid optomotor test (2026-10-13; `prereg/09_f3b1_optomotor.md`,
`docs/results/f3b1_optomotor.md`):**
- **Checks.** The input check and VNC gates 1, 2 and 3' pass.
- **Brain.** The FlyWire brain's DNa04 and DNa10 are direction-selective under the rotating gratings (Holm over 3
  families). The left DNa04 fires 5.9 Hz under `rot_left` and 0.9 Hz under `rot_right`. DNp31 (modulator) is also
  direction-selective (uncorrected).
- **VNC (primary test).** No arm shows direction-selective steering in the BANC VNC: not `ref` (gain 1), not
  `rate_gain`, and no control. Both registered outcomes are "not direction-selective".
- **Post-hoc.** In this open-loop visual protocol, the steering DNs fire below their F0 flight rates. The highest
  applied rate stays below the F3a-3 lowest steering rate for every family, including with `rate_gain` (DNa04 6.9 vs
  25 Hz). The brain's direction signal exists but does not reach the VNC's steering range.
- What follows is the project owner's decision.

**Decision after F3b-1 (project owner, 2026-10-13, PR #13 merged):**
1. The FlyWire brain turns grating direction into DN activity: DNa04 and DNa10 are direction-selective after Holm.
2. The signal reaches the VNC far below its steering range (DNa04 at most 6.9 Hz against 25 Hz). With three bridged
   families, the hybrid therefore shows no direction-selective steering.
3. Three candidate explanations, to be tested in this order:
   1. too few bridged DNs (in flies, steering is distributed over many DN types);
   2. no flight-state gain in the model;
   3. the neuron model (no adaptation or background; the LIF sits far from threshold).
4. Next step: F3b-2 tests explanation 1 (`prereg/10_f3b2_dn_ensemble.md`). The bridge carries all 112 shared DN families
   that pass the symmetry gate in both datasets, at gain 1, with the same bridge rules. It reuses the F3b-1 brain
   records. If the ensemble is not direction-selective either, the work stops there; the next step (neuron model v2)
   is the owner's decision.

**Finding from F3b-2, DN ensemble bridge (2026-10-13; `prereg/10_f3b2_dn_ensemble.md`,
`docs/results/f3b2_dn_ensemble.md`):**
- **Checks.** Gates 1, 2 (all 305 ensemble neurons and DNg02 at 100 Hz) and 3' pass. The F3b-1 brain records were
  reused (SHA-256 checked).
- **Brain.** 19 of the 112 ensemble families are direction-selective after Holm (e.g. DNp20, DNb06, DNg41, DNa11).
- **VNC (primary test, outcome B).** `ensemble` is direction-selective (b3, i1, hg1, hg4 after Holm), but
  `ensemble_shuffled` is also direction-selective (b3, hg1). The result is therefore **not specific**, and under the
  registered rule it does not support explanation 1. `bridge_off` is silent.
- **Post-hoc.** The two arms that are significant on b3 and hg1 have opposite signs. The shuffled control keeps the
  direction-selective rate streams and only moves them to other families. So it tests the family wiring, not whether
  direction information reaches the VNC.
- What follows is the project owner's decision.

**Decision after F3b-2 (project owner, 2026-10-13, PR #14 merged):**
1. **Outcome.** Under the registered rule, F3b-2 has outcome B. Both `ensemble` and `ensemble_shuffled` are
   direction-selective, so the result is not specific.
2. **Post-hoc observation, not a claim.** On b3 and hg1, the two arms have opposite signs. So the family wiring
   appears to set the sign of the steering response. The single-shuffle control tested the family mapping, not whether
   direction information reaches the VNC. This is a hypothesis for a confirmatory test.
3. **Next step: F3b-3** (`prereg/11_f3b3_specificity.md`), a confirmatory specificity test with fresh data.
   - Fresh brain runs with new seeds 10-29.
   - Arms: `ensemble` (the true wiring), `bridge_off`, and 39 random family derangements.
   - A replication test (Holm over b3, i1, i2, hg1, hg4), then a magnitude permutation test of |T_m| against the 39
     random wirings. Sign agreement is reported as a secondary result.

**Finding from F3b-3, confirmatory specificity test (2026-10-13; `prereg/11_f3b3_specificity.md`,
`docs/results/f3b3_specificity.md`):**
- **Checks.** Gates 1, 2 and 3' pass. The fresh brain runs (seeds 10-29) pass the input check, and their render equals
  the F3b-1 render. On the fresh seeds, 25 of the 112 ensemble families are direction-selective in the brain after
  Holm.
- **(a) Replication: yes.** In `ensemble`, b3 (+2.80 Hz), i1 (+2.75), hg1 (+2.75) and hg4 (-11.65) are significant
  after Holm. The signs are the same as in F3b-2. `bridge_off` is not direction-selective.
- **(b) Magnitude specificity: no.** |T_m(`ensemble`)| exceeds the 39 random wirings in 30 (b3), 30 (i1), 22 (hg1) and
  36 (hg4) of 39, so p_perm = 0.250, 0.250, 0.450 and 0.100. The criterion is 38 of 39 (p <= 0.05).
  - **Outcome under the registered rule: replicated, not specific.** The true family wiring gives a reproducible
    direction-selective response, but its size is typical of random wirings of the same rate streams.
- **(c) Sign specificity (secondary).** The fraction of random wirings with the same sign as `ensemble` is between
  0.46 and 0.62 for the four significant types. The F3b-2 hypothesis (the wiring sets the sign) is therefore not
  supported as a property that distinguishes the true wiring.
- **Exploratory (not used for decisions).** Literature-expected signs (Cheong et al., eLife, Table 1): b3, i1 and hg1
  agree, while hg4 has the opposite sign. 4 of the 6 comparable types agree.
- What follows is the project owner's decision.

**Decision after F3b-3 (project owner, 2026-10-14, PR #15 merged):**
1. **Outcome.** F3b-3 is "replicated, not specific" under the registered rule.
2. **Interpretation.** The BANC VNC subgraph turns a side-wise imbalance of direction-selective DN activity into
   steering-MN asymmetry, largely independent of which families carry it. The sign is not set by the true wiring: the
   same-sign fraction across the random wirings is 0.46-0.62.
3. **Exploratory.** The true wiring matches the literature sign in 3 of 4 significant types (hg4 is reversed).
4. **Consequence for claims.** Direction information passes through the hybrid reliably. We do not claim that the
   specific DN wiring steers.
5. **Next: writing and data-only analysis, no new simulation** (branch `omeruk/summary-and-f4-design`):
   `docs/results/SUMMARY_F0_F3.md` (every number checked against its report), and `docs/08_CLOSED_LOOP_OPTOMOTOR.md`
   (design of the first closed-loop test, T2, with a data-only open-loop prediction from the F3b-3 records, labelled a
   design analysis and not a test).

**Finding from the T2 design analysis (2026-10-14; `docs/08_CLOSED_LOOP_OPTOMOTOR.md` section 8, post-hoc, not a test):**
- Table 1 of Cheong et al. gives activity by side of a turn and no kinematic effect; only b2 and b3 have a stated
  relation to stroke amplitude in the sources that could be read. The signs of the other mapped types are inferred.
- With the lateralisation signs, the open-loop F3b-3 records predict a weakly anti-directional turn for `ensemble` that
  is not distinguishable from zero over 20 seeds. The sign changes with the mapping of hg4, the random wirings are
  split between the two directions, and a constant yaw offset exceeds the direction-dependent term.
- So the closed-loop test is expected to show nothing reliable. The choice of the primary mapping set, the gain
  and the other design questions (section 9 of docs/08) are the project owner's decision.

**Decision after the T2 design (project owner, 2026-10-14, PR #16 merged):**
1. **Prediction.** The docs/08 open-loop prediction says the closed-loop test would show nothing: S = -1.00 +/- 3.39 Hz
   for `ensemble` (mapping set `lateral`, 20 seeds). Its sign is set by hg4 and by the inferred mapping. The constant yaw
   offset (-12 Hz) is larger than the direction term, and the random wirings are split (17 of 39 syndirectional).
2. **F4-1 (closed-loop optomotor run) is postponed** until the open-loop signal is syndirectional, significant and
   specific.
3. **Primary mapping set: `lateral`**, because it does not pick types by outcome. `direct` (b2, b3 only) is a
   registered secondary set.
4. **Literature.** Lindsay et al. 2017 (PDF in `literature/`, not committed) is to be read first, and the per-muscle
   signs and any relative effect sizes are to be updated from it.
5. **Lindsay rules (fixed before the v2 numbers).** The primary `lateral` set is every muscle whose sign Lindsay states
   directly and that passes the gate (b2, b3, i1, i2, hg1, hg2, hg3, hg4). iii1 and iii4 (text and Fig. 6 disagree)
   are sensitivity sets with both signs. Gain G1 is fixed at the v1 value. The closed-loop yaw trim subtracts the
   per-type (R - L) under grey of the same arm, calibrated on separate seeds (`E`). The other docs/08 questions are
   answered with the recommendations (docs/08 section 9.1).

**Outcome of the Lindsay update (2026-10-14; docs/08 section 10, `docs/results/f4_design_numbers_v2.json`, design
analysis, not a test):** every sign that carries weight in v1 is confirmed directly; the paper gives no per-muscle
effect sizes. The prediction is unchanged: weakly anti-directional, not distinguishable from zero, sign set by hg4,
and now also by the iii4 conflict. The grey trim removes nothing, because the mapped MNs are silent under grey; the
offset is evoked by the patterned stimulus. Changing the trim is open for the owner.

Task ladder (each with a pre-registered criterion):
T0 open-loop responses → T1 hover + perturbation → T2 optomotor yaw tracking → T3 GF take-off → T4 landing on the platform
→ T5 orienting to the target (visual target: the dark platform; olfaction only if the AL problem is solved) → T6 full task.

**Information-alignment rule (lesson from H3):** in the TRAINED arm the teacher signal must be derivable from the senses the
brain receives (e.g. a visually defined route). Pre-test: cross-validated R² > 0 in a brainless "bypass" arm (random projection).

## 5. Proposed directory layout

```
synaptera-banc-brain/
  connectome/   adapters (banc, flywire783, malecns), schema.py, crosswalk.csv, qa.py
  engine/       lif_brian2.py, lif_genn.py, rate_torch.py (+ sparse autograd), cond_adapt.py, protocol.py
  sense/        vision_flyvis.py, l3_luminance.py, haltere.py, wing_cs.py, jo.py, tarsal_grn.py, olfaction.py
  motor/        wing_muscle.py, haltere_mn.py, leg_mn.py, neck_mn.py, proboscis.py, legacy_bridge.py
  body/         flygym_body.py, flybody_body.py, quasi_steady.py, noise.py
  coupling/     scheduler.py (causal queue), recorder.py (HDF5/Zarr), replay.py
  tasks/        t0_openloop.py ... t6_full.py, criteria.py
  train/        bc.py, dagger.py, controls.py (rewired/random/linear/MLP)
  prereg/       OSF links, criteria files (immutable)
  scripts/      fetch_data.py, verify_report.py, figures/
  configs/      *.yaml (+ parameter origin register assumptions.json)
  tests/        unit + integration + "golden" open-loop tests
```

## 6. Separate sub-plan for olfaction (optional)

The structural analysis shows that the persistent state in the AL comes from the **sign rule** (77 % of the internal LN+PN
synapses excitatory; 54 aminergic LNs counted as excitatory; leading eigenvalue ~2,184 synapses → a self-sustaining mode
above ~2.3 Hz). Options (each labelled as a **model variant**): (i) remove aminergic neurons from the fast path (existing
`--nt-modulatory-silent narrow`), (ii) literature signs for the LNs (existing `--nt-literature`, 24 neurons were not enough),
(iii) a neuron model with adaptation (c), (iv) input normalisation in the rate model. Gate: the O1 criterion (< 0.1 Hz
100–200 ms after the cut).

## 7. Compute plan

| work | resource | estimated cost |
|---|---|---|
| Spiking arm, BANC ~160–190k neurons, CPU Brian2 | one process/core; 1 s simulated ≈ 20–40 s wall (scaled from BioReservoir measurements; to be verified) | 7.5 s flight ≈ 3–5 min; 5 seeds × 10 conditions ≈ a few hours |
| Spiking arm, GPU (GeNN/Brian2CUDA) | one GPU | close to real time (FastFly/Eon benchmarks) |
| Rate-arm training | A6000/A40-class GPU (~$0.5/hour) | flyhard pilot: 600 steps in 186 s, ~3 GB; hours per task |
| FlyVis + rendering | CPU/GPU | the cost of more sub-frames must be measured |

## 8. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| MN9 match in BANC uncertain | low (MN9 named in the Codex export, 03 appendix A) | RQ3, feeding | confirm by connectivity profile + NBLAST |
| BANC L/R proofreading asymmetry | medium–high | lateral readouts | symmetry gate; mirror-symmetric control; population-level readouts |
| VNC paths silent (no resting activity) | high | RQ1–RQ2 | variant (c); tonic drive (labelled); rate arm |
| Muscle-model parameters uncertain | high | RQ1 | open-loop literature only; sensitivity analysis; origin class `P/F` |
| Compute cost (5 ms coupling) | medium | schedule | GeNN/PyTorch; DEV subnet for development only |
| Training shows no "topology advantage" | medium | RQ4 | reported anyway; task choice pre-registered |
| Visual input on BANC (F1 QA, 2026-10-13): optic-lobe intrinsic types pass the symmetry gate at 16 % (median abs log2 1.40) | high | any visual task | visual input enters **only through visual projection neuron (VPN) types that pass the gate**, never through optic-lobe intrinsic types; 99 of 281 VPN types pass: H2, LC10c, LC11, LC12, LC14b, LC15, LC16, LC19, LC21, LC25, LC31c, LC33, LC4, LC41, LC43, LC9, LCe01a, LCe01b, LLPC2, LLPC3, LMTe01, LPLC2, LPLC4, LPT21, LPT28, LPT31, LPT45_dCal1, LPT48_vCal3, LPT49, LPT50, LPT54, LT1d, LT43, LT51, LT52, LT53, LT55, LT61b, LT62, LT63, LT73, LT82, LT84, LT86, LTe08, LTe12, LTe13, LTe15, LTe16, LTe18, LTe20, LTe24, LTe26, LTe27, LTe33, LTe43, LTe45, LTe51, LTe53, LTe54, LTe58, LTe60, LTe65, LTe69, LTe70, LTe72, MC65, MTe01b, MTe03, MTe06, MTe10, MTe23, MTe29, MTe30, MTe32, MTe34, MTe35, MTe37, MTe39, MTe41, MTe43, MTe46, MTe47, MTe51, MTe53, MeLp1, MeMe_e04, MeMe_e08, MeTu3c, OCG01f, aMe17a2, aMe17c, aMe19a, aMe20, aMe3, aMe4, aMe5, aMe8, s-LNv_a (`docs/results/f1_banc_qa.md` section 2) |
| HS cells fail the gate on BANC (HSE +0.82, HSS +0.59, HSN +0.54 log2 R/L, exc) | high | optomotor yaw | no HS-based L/R yaw readout under rule 5 unless waived and labelled; report the gate result with the experiment |
| TTMn fails the gate (log2 R/L +0.64) | medium | GF -> TTMn take-off path (B1-3) | report per side; treat any L/R-difference result on this path as gate-failed; DNp01 (GF) and the wing MNs are the readout alternatives |
| DNa01 fails the gate (+0.63); DNa02 (+0.14) and DNp15 (+0.23) pass on BANC (DNp15 fails on FlyWire, R/L 1.66x) | n/a | steering readout | DNa02 stays the primary steering drive; DNp15 needs no waiver on BANC (still needs one on FlyWire) |
| LB3b sugar GRNs are asymmetric (labellar sugar GRNs 41 L / 49 R; LB3b 20 L / 40 R in the neuron table) | high | sugar -> MN9 | sugar -> MN9 results are reported **per side** (drive left only / right only / both), never only pooled |
| The antennal lobe has the same self-excitatory structure on BANC (internal excitatory fraction 76.6 %) | medium | spontaneous ignition | the pre-registered ignition-frequency check (`prereg/02`) applies to BANC runs too, with its own seeds and the same definition |
| Structural w_syn scaling puts BANC in the self-sustaining regime (F2 prep calibration, `docs/results/f2_wsyn_calibration.md`) | high (observed) | every BANC closed-loop run | none of A / A2 / B passed the gates (ignition after the cut 18-20 / 20 at scales 2.4-3.0); the reference arm is silent at MN9. A new pre-registration is needed before any F2 test on BANC (e.g. input normalisation, inhibition scale); the reference arm result is reported as is |
| Licence | low | publication | BANC CC BY 4.0; FlyWire CC BY-NC; derived files in a separate bundle |

## 9. First concrete steps

1. F0: port the Synaptera code into the new layout (see `06_NEW_PROJECT_SETUP.md`), with the command queue, overwrite
   protection, conditional loading of the T4/T5 column file and amplitude clipping built in; tests.
2. F0: turn `scripts/dn_lr_asymmetry.py` into the readout-gate tool; publish the report for DNa02/DNp15.
3. F1: BANC adapter on the common schema; hash the data; QA report.
4. F1: first crosswalk: DNp01, DNg02 (a–h), DNa01/02, DNp15, DNp07/10, MN9, LB3, haltere afferents, wing MNs.
5. F2 pre-registration: criteria of the 5 open-loop tests (OSF + repository).

## 10. Why this order (link to the Synaptera results)

- DNg02 silence: 54 % of its input is inhibitory, 8 % from ascending neurons that are never driven → a power readout does
  not seem possible without VNC + AN feedback and resting activity → **F1–F3 first**.
- DNp15/DNp10 failures: consistent with the structural L/R asymmetry → **symmetry gate in F0**.
- Leg GRN → MN9 failure: the signed influence is negative in FlyWire and the VNC data are missing → **F2 in BANC**.
- Training ladder: information mismatch + stopped before DAgger → **information-alignment rule and DAgger mandatory in F5**.
- Olfactory lock-up: positive feedback caused by the sign rule → **separate sub-plan (§6)**, not on the critical path of flight.
