"""Consolidated summary F0 - F3b-3: docs/results/SUMMARY_F0_F3.md (analysis and writing only; no simulation).

    env -u PYTHONPATH python scripts/summary_f0_f3.py

Every number in the summary is a `{{fact.N}}` placeholder that is filled by a regular expression applied to an
existing report in docs/results (the FACTS table below). `scripts/verify_report.py` re-applies every expression to
the current reports, re-renders the template and compares the result with the committed summary; it also refuses a
template that contains a number of its own. The facts are written to docs/results/summary_f0_f3.json.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "docs" / "results"
OUT_MD = RES / "SUMMARY_F0_F3.md"
OUT_JSON = RES / "summary_f0_f3.json"

# id -> (report file, regular expression, mode). mode "search": groups of the first match; "count": number of matches.
FACTS = {
    # F0
    "f0_pass": ("f0_multiseed.md", r"\| pass \| pass \|", "count"),
    "f0_n": ("f0_multiseed.md", r"(?m)^\| \d \| \d \| (?:pass|FAIL) \|", "count"),
    "f0_posdiff": ("f0_acceptance.md", r"max position difference to v1 \| ([\d.]+) mm", "search"),
    "f0_ign": ("f0_ignition_frequency.md", r"estimated ignition probability of the new engine is\s+~(\d+ / \d+) \(([\d.]+) %\), v1 (\d+ / \d+)", "search"),
    "f0_ign100": ("f0_ignition_frequency.md", r"\*\*Primary result:\*\* (\d+ / \d+) ignitions in every arm.*?upper bound is ([\d.]+) %", "search"),
    "f0_lat": ("f0_latency_hover_yaw.md", r"\(mean ~(\d+) deg of total heading change, (\d+)-(\d+) deg\s+max difference\)", "search"),
    # F1
    "f1_counts": ("f1_banc_qa.md", r"neurons ([\d,]+) \(left ([\d,]+), right ([\d,]+)\); typed ([\d.]+)%", "search"),
    "f1_sym": ("f1_banc_qa.md", r"\| all \| (\d+) \| ([\d.]+) \| ([\d.]+) \|", "search"),
    "f1_smoke": ("f1_banc_smoke.md", r"MN9 is silent \(([\d.]+) Hz in every condition", "search"),
    # F2
    "f2_ignA": ("f2_wsyn_calibration.md", r"\| A_wsyn_struct \| [\d.]+ \| \d+ \| (\d+ / \d+) \(", "search"),
    "f2_ignA2": ("f2_wsyn_calibration.md", r"\| A2_wsyn_struct_class \| [\d.]+ \| \d+ \| (\d+ / \d+) \(", "search"),
    "f2_ignB": ("f2_wsyn_calibration.md", r"\| B_wsyn_fit_mn9 \| [\d.]+ \| \d+ \| (\d+ / \d+) \(", "search"),
    "f2_ignR": ("f2_wsyn_calibration.md", r"\| R \| 1\.000 \| \d+ \| (\d+ / \d+) \(", "search"),
    "f2_full": ("f2_full_graph.md", r"labellar mean ([\d.]+) Hz >= 10% of FlyWire's ([\d.]+) Hz", "search"),
    "f2_r3": ("f2_innorm.md", r"\| R3 \| banc888_full_v3 \| pass \| [^|]+\| [^|]+\| ([\d.]+) \| yes \| no \|", "search"),
    "f2_thr": ("f2_innorm.md", r"labellar mean [\d.]+ Hz >= ([\d.]+) Hz\)", "search"),
    # F3 crosswalk
    "x_shared": ("f3_dn_crosswalk.md", r"shared by name (\d+) \((\d+) FlyWire and (\d+) BANC neurons\)", "search"),
    "x_gate": ("f3_dn_crosswalk.md", r"Shared families passing the symmetry gate in both datasets: (\d+)", "search"),
    # F3a
    "a_gate2": ("f3a_banc_vnc.md", r"\| 2: ignition after the 200 ms cut, all DN groups 100 Hz, 50 seeds \| (\d+ / \d+)", "search"),
    "a_dlm": ("f3a_banc_vnc.md", r"DLM1-4 [\d.]+ / [\d.]+ / ([\d.]+) Hz; DLM5 [\d.]+ / [\d.]+ / ([\d.]+) Hz", "search"),
    # F3a-2
    "a2_c2": ("f3a2_checks.md", r"C2: (\d+) neurons carry activity after the cut; network rate in the last cut step \(mean over seeds\) ([\d.]+) Hz", "search"),
    "a2_dng02": ("f3a2_checks.md", r"\| DNg02 \| flight \| 13/12 \| ([\d.]+),", "search"),
    # F3a-3
    "a3_sub": ("f3a3_vnc_subgraph.md", r"Kept: ([\d,]+) neurons, ([\d,]+) pairs, ([\d,]+) synapses", "search"),
    "a3_r2": ("f3a3_vnc_subgraph.md", r"\*\*R2: (\d+ / \d+) seeds decay; pass", "search"),
    "a3_stop": ("f3a3_vnc_subgraph.md", r"Activity after the cut lasts \(last step with any spike, ms after the cut\), per seed: (\d+),", "search"),
    "a3_low": ("f3a3_vnc_subgraph.md", r"\| DNa04 \| (\d+) \| [\d, ]+ \| [\d.]+ / [\d.]+ \| ([\d.]+ / [\d.]+) \|", "search"),
    "a3_low10": ("f3a3_vnc_subgraph.md", r"\| DNa10 \| (\d+) \| [\d, ]+ \| [\d.]+ / [\d.]+ \| ([\d.]+ / [\d.]+) \|", "search"),
    # F3b-1
    "b1_sel": ("f3b1_optomotor.md", r"Direction-selective in the brain after Holm: ([A-Za-z0-9, ]+)\.", "search"),
    "b1_max": ("f3b1_optomotor.md", r"\| DNa04 \| ([\d.]+) \| ([\d.]+) \| (\d+) \| rot_left / left \|", "search"),
    # F3b-2
    "b2_brain": ("f3b2_dn_ensemble.md", r"Brain: (\d+) of (\d+) ensemble families carry", "search"),
    "b2_mean": ("f3b2_dn_ensemble.md", r"`ensemble` b3 ([+-]\d+\.\d+), i1 ([+-]\d+\.\d+), hg1 ([+-]\d+\.\d+), hg4 ([+-]\d+\.\d+); `ensemble_shuffled` b3 ([+-]\d+\.\d+), hg1 ([+-]\d+\.\d+)", "search"),
    # F3b-3
    "c_gate": ("f3b3_specificity.md", r"maximum ([\d.]+) Hz \(replays: ([\d.]+) Hz", "search"),
    "c_brain": ("f3b3_specificity.md", r"Direction-selective after Holm: \*\*(\d+)\*\* of (\d+)", "search"),
    "c_T": ("f3b3_specificity.md", r"\| ensemble \| b3 \| ([+-][\d.]+) \|.*?\| ensemble \| i1 \| ([+-][\d.]+) \|.*?\| ensemble \| hg1 \| ([+-][\d.]+) \|.*?\| ensemble \| hg4 \| ([+-][\d.]+) \|", "search"),
    "c_perm": ("f3b3_specificity.md", r"\| b3 \| \+[\d.]+ \| yes \| (\d+) / 39 \| ([\d.]+) \|.*?\| i1 \| [+-][\d.]+ \| yes \| (\d+) / 39 \| ([\d.]+) \|.*?\| hg1 \| [+-][\d.]+ \| yes \| (\d+) / 39 \| ([\d.]+) \|.*?\| hg4 \| [+-][\d.]+ \| yes \| (\d+) / 39 \| ([\d.]+) \|", "search"),
    "c_same": ("f3b3_specificity.md", r"b3 ([\d.]+), i1 ([\d.]+), i2 ([\d.]+), hg1 ([\d.]+), hg4 ([\d.]+) \(secondary", "search"),
    "c_expl": ("f3b3_specificity.md", r"Types with a fixed expected sign that pass the gate: (\d+); observed sign agrees in (\d+)\. Among those significant in `ensemble` \(secondary Holm\): (\d+), agreeing (\d+)\.", "search"),
    "c_checks": ("f3b3_specificity.md", r"Gate 2 \(DNg02 and all (\d+) ensemble neurons at 100 Hz, 200 ms cut, seeds [\d-]+\): (\d+ / \d+) ignited", "search"),
}

TEMPLATE = """# Summary F0 - F3b-3: what works, what does not, what we can claim

Written 2026-10-14 by `scripts/summary_f0_f3.py`. Analysis and writing only: nothing was simulated for this document.
Every number is read from the linked report by a regular expression (`scripts/summary_f0_f3.py`, table `FACTS`) and
re-checked against that report by `scripts/verify_report.py`; the facts are in `docs/results/summary_f0_f3.json`.
Plan and decisions: `docs/05_ARCHITECTURE_PLAN.md`. Next step (design only): `docs/08_CLOSED_LOOP_OPTOMOTOR.md`.

## 1. The path so far

| phase | question | result | report |
|---|---|---|---|
| F0 | does the ported v1 stack reproduce v1 (FlyWire brain + body)? | yes; one realisation shows a self-sustaining ignition | [f0_acceptance](f0_acceptance.md), [f0_multiseed](f0_multiseed.md), [f0_ignition_frequency](f0_ignition_frequency.md) |
| F1 | does BANC load, and does the reference arm run on it? | loads and runs; MN9 is silent | [f1_banc_qa](f1_banc_qa.md), [f1_banc_smoke](f1_banc_smoke.md) |
| F2 | can the BANC whole-brain graph replace the FlyWire brain? | no; the track stops and the hybrid starts | [f2_wsyn_calibration](f2_wsyn_calibration.md), [f2_full_graph](f2_full_graph.md), [f2_innorm](f2_innorm.md) |
| F3 prep | which DN families are shared and pass the symmetry gate? | {{x_gate.1}} of {{x_shared.1}} shared families | [f3_dn_crosswalk](f3_dn_crosswalk.md) |
| F3a, F3a-2, F3a-3 | does the BANC VNC (subgraph) turn DN activity into wing MN activity? | yes for wing power; steering by DNa04, DNa10, DNa15 | [f3a_banc_vnc](f3a_banc_vnc.md), [f3a2_checks](f3a2_checks.md), [f3a3_vnc_subgraph](f3a3_vnc_subgraph.md) |
| F3b-1 | does a rotating grating reach the VNC through three DN families? | no | [f3b1_optomotor](f3b1_optomotor.md) |
| F3b-2 | does a {{b2_brain.2}}-family DN ensemble carry it? | direction-selective, not specific | [f3b2_dn_ensemble](f3b2_dn_ensemble.md) |
| F3b-3 | is that specific to the true wiring (fresh data, permutation test against 39 random wirings)? | replicated, not specific | [f3b3_specificity](f3b3_specificity.md) |

## 2. What works

- **The v1 flight stack is ported and reproduces v1 at the acceptance criteria.** In the multi-seed acceptance,
  {{f0_pass}} of {{f0_n}} flights pass S1 and S2; the largest position difference to v1 is {{f0_posdiff.1}} mm
  ([f0_acceptance](f0_acceptance.md), [f0_multiseed](f0_multiseed.md)).
- **The BANC v3 VNC subgraph is a usable motor stage.** It keeps {{a3_sub.1}} of the BANC neurons
  ({{a3_sub.2}} pairs, {{a3_sub.3}} synapses). Persistence is solved: activity stops within {{a3_stop.1}} ms of the
  cut in {{a3_r2.1}} seeds, where the whole graph does not decay ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **Wing power comes from the VNC.** Bilateral DNg02 drive at 100 Hz drives DLM1-4 ({{a_dlm.1}} Hz) and DLM5
  ({{a_dlm.2}} Hz) ([f3a_banc_vnc](f3a_banc_vnc.md)). DNg02 itself is silent in the FlyWire brain
  ({{a2_dng02.1}} Hz in every F0 flight window), so in the hybrid it is a `HAND` baseline
  ([f3a2_checks](f3a2_checks.md)).
- **Three DN families steer ipsilaterally in the VNC.** DNa04 steers from {{a3_low.1}} Hz, which is inside its FlyWire
  flight range (L / R {{a3_low.2}} Hz) ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **The FlyWire brain turns grating direction into DN activity.** In F3b-1 the direction-selective families after Holm are {{b1_sel.1}}; in F3b-3
  {{c_brain.1}} of {{c_brain.2}} ensemble families are, on fresh seeds ([f3b1_optomotor](f3b1_optomotor.md),
  [f3b3_specificity](f3b3_specificity.md)).
- **Direction information passes through the hybrid reliably.** With the {{b2_brain.2}}-family ensemble bridge the
  steering MN asymmetry is direction-selective in b3 ({{c_T.1}} Hz), i1 ({{c_T.2}} Hz), hg1 ({{c_T.3}} Hz) and hg4
  ({{c_T.4}} Hz), replicated on fresh brain seeds with the same signs as F3b-2; the cut-bridge control is silent
  ([f3b3_specificity](f3b3_specificity.md) section 4).
- **The gates behave.** In F3b-3 gate 1 gave no spikes, gate 2 gave {{c_checks.2}} ignitions with all {{c_checks.1}}
  ensemble neurons and DNg02 at 100 Hz, and the network rate stays at or below {{c_gate.1}} Hz (gate-2 condition)
  ([f3b3_specificity](f3b3_specificity.md) section 2).

## 3. What does not work

- **The BANC whole-brain graph with the reference parameters does not drive MN9.** MN9 is {{f1_smoke.1}} Hz in every
  condition on the Codex export ([f1_banc_smoke](f1_banc_smoke.md)); on the full v888 pair-level graph it is still
  {{f2_full.1}} Hz against {{f2_full.2}} Hz in FlyWire ([f2_full_graph](f2_full_graph.md)).
- **Uniform w_syn scaling ignites the network before it makes MN9 respond.** Ignitions after the cut: {{f2_ignA.1}}
  (structural scale), {{f2_ignA2.1}} (class scale), {{f2_ignB.1}} (fitted to MN9), and {{f2_ignR.1}} in the unscaled
  reference arm, which is silent at MN9 ([f2_wsyn_calibration](f2_wsyn_calibration.md)).
- **The best BANC arm is lateralised and does not qualify.** On the v3 graph (unchanged parameters) MN9 reaches
  {{f2_r3.1}} Hz, just above the {{f2_thr.1}} Hz threshold, but through one neuron on one side, the leg path is silent
  and gate 3 fails ([f2_innorm](f2_innorm.md)).
- **Three DN families are too few.** In F3b-1 no arm is direction-selective in the VNC: the highest applied rate of
  DNa04 is {{b1_max.2}} Hz against its lowest steering rate of {{b1_max.3}} Hz ([f3b1_optomotor](f3b1_optomotor.md)).
- **The effect is not specific to the true wiring.** In F3b-3 the true wiring beats {{c_perm.1}} / 39 (b3),
  {{c_perm.3}} / 39 (i1), {{c_perm.5}} / 39 (hg1) and {{c_perm.7}} / 39 (hg4) of the random wirings, with permutation
  p of {{c_perm.2}}, {{c_perm.4}}, {{c_perm.6}} and {{c_perm.8}} against a criterion of 0.05
  ([f3b3_specificity](f3b3_specificity.md) section 4).

## 4. Negative and mixed results (reported with equal weight)

- **F0, ignition and latency** ([f0_ignition_frequency](f0_ignition_frequency.md),
  [f0_latency_hover_yaw](f0_latency_hover_yaw.md)): the new engine ignited the olfactory self-sustaining mode in
  {{f0_ign.1}} realisations ({{f0_ign.2}} %) against {{f0_ign.3}} for v1; the registered 100-seed test found
  {{f0_ign100.1}} ignitions per arm (95 % upper bound {{f0_ign100.2}} %), so equality is not demonstrated and not
  excluded. The causal-latency effect on a brain-closed hover flight (about {{f0_lat.1}} deg of heading, up to
  {{f0_lat.2}}-{{f0_lat.3}} deg) is not separable from seed-to-seed variability.
- **F2, BANC reference arm silent at MN9** ([f1_banc_smoke](f1_banc_smoke.md), [f2_full_graph](f2_full_graph.md),
  [f2_innorm](f2_innorm.md)): a data property (sparser graph, known proofreading asymmetry), not something tuned away.
- **F3a, expectations (b) and (c) missed** ([f3a_banc_vnc](f3a_banc_vnc.md)): steering MNs responded on one side only
  on the whole graph, and unilateral DNa02 did not steer. On the subgraph the one-sided MN responses are gone
  ([f3a3_vnc_subgraph](f3a3_vnc_subgraph.md)).
- **F3b-1, the three-family bridge** ([f3b1_optomotor](f3b1_optomotor.md)): not direction-selective in any arm.
- **F3b-2, outcome B** ([f3b2_dn_ensemble](f3b2_dn_ensemble.md)): `ensemble` is direction-selective, but
  so is the shuffled control, with b3 and hg1 of opposite signs ({{b2_mean.1}} and {{b2_mean.5}} Hz for b3;
  {{b2_mean.3}} and {{b2_mean.6}} Hz for hg1). That led to the F3b-3 hypothesis that the wiring sets the sign.
- **F3b-3, the sign hypothesis is not supported** ([f3b3_specificity](f3b3_specificity.md)): the fraction of random
  wirings with the same sign as `ensemble` is {{c_same.1}} (b3), {{c_same.2}} (i1), {{c_same.4}} (hg1) and {{c_same.5}}
  (hg4); the true wiring does not set the sign.
- **F3b-2 and F3b-3 used different brain seeds.** The F3b-2 ensemble had {{b2_brain.1}} direction-selective
  families and F3b-3 has {{c_brain.1}} (different seeds); the lists overlap but are not equal. Nothing downstream
  depends on the exact list.

## 5. Open items

1. **Which DN families carry steering in flight.** The bridge uses the shared, gate-passing families only; the
   real fly uses more DN types (BANC-only DNs are not bridged).
2. **The sign of the steering MN response.** It is set by the VNC and the ensemble as a whole, not by the specific
   DN wiring. Its relation to the literature sign is partial (below).
3. **The kinematic effect of each steering muscle.** Table 1 of Cheong et al. lists activity on the inner and outer side
   of a turn, not the effect on stroke amplitude; only two muscles have a direct statement in the sources read
   (`docs/08_CLOSED_LOOP_OPTOMOTOR.md` section 4.1).
4. **Neck and haltere MN responses** are lost on the subgraph (they depended on removed BANC brain / SEZ parts) and
   need DNs outside the current set.
5. **No body, no feedback so far.** Everything up to F3b-3 is open loop: no visual feedback, no muscle model, no
   VNC -> brain feedback. T2 (`docs/08_CLOSED_LOOP_OPTOMOTOR.md`) is a design, not a result.
6. **Neuron-model realism** (no adaptation or background; the LIF sits far from threshold) was listed in docs/05 as
   explanation 3 for the weak drive and has not been tested.

## 6. What we can and cannot claim

**We can claim**
- The FlyWire v783 brain model turns the direction of a rotating grating into direction-selective DN activity, and a
  DN ensemble bridge carries that into steering-MN asymmetry of the BANC v3 VNC subgraph, reproducibly across fresh
  seeds, and only when the bridge is on.
- The BANC VNC subgraph is a stable, non-self-sustaining motor stage for wing power and steering MN asymmetry under
  tonic wing-power drive.
- The BANC whole-brain graph with the unchanged reference parameters is not a usable brain replacement.

**We cannot claim**
- That the specific DN wiring steers: random derangements of the same rate streams give MN asymmetries of the same size
  and either sign.
- That the hybrid turns a fly with or against the drum: nothing has been closed in a loop, and the mapping from MN rate
  to wing kinematics has no direct literature support for most types.
- That the match to the literature sign is evidence of function: {{c_expl.2}} of {{c_expl.1}} comparable types agree and
  {{c_expl.4}} of the {{c_expl.3}} significant ones agree (hg4 is reversed), and this comparison was exploratory.
- Anything about BANC brain circuits, olfaction, landing, take-off or learning: none of that was run in the hybrid.
"""

STRAY = re.compile(r"\d")
ALLOWED = [r"\{\{[A-Za-z0-9_]+(?:\.\d)?\}\}", r"\[[^\]]*\]\([^)]*\)", r"`[^`]*`",
           r"F\d[a-z]?(?:-\d)?", r"prereg/\d+", r"docs/0\d", r"\bT2\b", r"\bv\d+\b", r"v783", r"v888", r"Table 1",
           r"2026-10-1\d", r"\bH\d\b", r"\bRQ\d\b", r"\bMN9\b", r"DLM1-4", r"DLM5", r"DNa\d+", r"DNg\d+", r"DNp\d+",
           r"\bb[123]\b", r"\bi[12]\b", r"\bhg\d\b", r"\biii\d\b", r"BANC v3", r"section \d(?:\.\d)?",
           r"explanation \d", r"gate-\d", r"\bGate \d", r"\bgate \d", r"\b0\.05\b", r"\b100 Hz\b", r"\b100-seed\b",
           r"\b95 %", r"\b39\b", r"\b[A-Z]\d\b", r"\bw_syn\b", r"\b\d\. ", r"\b\d\)", r"[a-z0-9_]+\.md",
           r"summary_f0_f3", r"\b1 -"]


def capture(facts=FACTS, res=RES):
    vals = {}
    for k, (fn, rx, mode) in facts.items():
        text = (res / fn).read_text()
        if mode == "count":
            vals[k] = [str(len(re.findall(rx, text)))]
        else:
            m = re.search(rx, text, re.S)
            if not m:
                raise SystemExit(f"fact {k}: no match in {fn}")
            vals[k] = list(m.groups())
    return vals


def render(template, vals):
    def sub(m):
        k, _, n = m.group(1).partition(".")
        v = vals[k]
        return v[int(n) - 1] if n else v[0]
    return re.sub(r"\{\{([A-Za-z0-9_]+(?:\.\d)?)\}\}", sub, template)


def stray_numbers(template):
    """Digits left in the template after removing placeholders, links, code and whitelisted tokens."""
    t = template
    for pat in ALLOWED:
        t = re.sub(pat, " ", t)
    return sorted(set(re.findall(r"\S*\d\S*", t)))


def main():
    vals = capture()
    text = render(TEMPLATE, vals)
    OUT_MD.write_text(text)
    OUT_JSON.write_text(json.dumps(
        {k: dict(file=FACTS[k][0], regex=FACTS[k][1], mode=FACTS[k][2], values=vals[k]) for k in FACTS}, indent=1))
    bad = stray_numbers(TEMPLATE)
    if bad:
        print("numbers typed in the template:", bad)
        return 1
    print("facts", len(FACTS), "-> ", OUT_MD.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
