"""Render docs/results/f3b2_dn_ensemble.{md,json} from runs/f3b2/f3b2_v3.json (prereg/10; numbers by script).

    env -u PYTHONPATH python scripts/f3b2_report.py [--overwrite]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from f3a2_report import ALPHA, cp, p_t, p_w  # noqa: E402
from f3b1_report import lr, ms, test_block  # noqa: E402
import f3b2_common as C  # noqa: E402

REC = ROOT / "runs" / "f3b2" / "f3b2_v3.json"
F3B1 = ROOT / "runs" / "f3b1" / "f3b1_v3.json"
F3B1_REPORT = ROOT / "docs" / "results" / "f3b1_optomotor.json"
OUT = ROOT / "docs" / "results" / "f3b2_dn_ensemble"
STIMS = tuple(C.STIMULI)
SIDES = ("left", "right")


def runs(rp, a, st):
    return sorted([r for r in rp if r["arm"] == a and r["stimulus"] == st], key=lambda r: r["seed"])


def setup_section(d, out):
    e, s = d["bridges"]["ensemble"], d["bridges"]["ensemble_shuffled"]
    br = d["brain_records"]
    match = {a: {m: sum(u["match"] == m for u in d["bridges"][a]["units"]) for m in ("type", "family", "shuffled")}
             for a in ("ensemble", "ensemble_shuffled")}
    fam_units = sorted(f"{u['target_family']} {u['side']} ({', '.join(u['fw_types'])} -> {', '.join(sorted(set(u['banc_types'])))})"
                       for u in e["units"] if u["match"] == "family")
    n_fw = {a: int(sum(u["n_fw"] for u in d["bridges"][a]["units"])) for a in ("ensemble", "ensemble_shuffled")}
    out["setup"] = dict(units={a: len(d["bridges"][a]["units"]) for a in match}, match=match,
                        n_banc=d["ensemble_banc"]["n"], by_side=d["ensemble_banc"]["by_side"],
                        unmatched_banc={a: len(d["bridges"][a]["unmatched_banc"]) for a in match},
                        unmatched_fw={a: len(d["bridges"][a]["unmatched_fw"]) for a in match},
                        n_fw_unit_sum=n_fw, brain_files=len(br["sha256"]), n_dn=br["n_dn"])
    return ["## 1. Setup (prereg/10 sections 2-5)", "",
            f"- Brain records reused from F3b-1 (`{br['dir']}`): {len(br['sha256'])} files, {br['n_dn']:,} FlyWire DNs "
            "each; SHA-256 of every file recorded and checked by `scripts/verify_report.py`. No new brain run.",
            f"- BANC VNC subgraph {d['subgraph']['n_neurons']:,} neurons, {d['subgraph']['pairs']:,} pairs, "
            f"{d['subgraph']['synapses']:,} synapses; reference arm. Bridge latency {e['config']['latency_intervals']} "
            f"interval ({C.INTERVAL_MS:g} ms), window {e['config']['window_intervals']} intervals, per-neuron mean rate, "
            "independent Poisson re-synthesis (spike-timing correlations are lost), gain 1.",
            f"- Ensemble: {len(C.ENSEMBLE)} families, {d['ensemble_banc']['n']} BANC neurons "
            f"({d['ensemble_banc']['by_side'].get('left', 0)} left, {d['ensemble_banc']['by_side'].get('right', 0)} right). "
            f"DNg02 `HAND` {C.BASELINE_HZ:g} Hz both sides ({d['groups']['DNg02_L']['n']} L / {d['groups']['DNg02_R']['n']} R "
            "neurons), not in the brain-driven set; every other BANC DN silent.",
            f"- `ensemble`: {len(e['units'])} units ({match['ensemble']['type']} by exact type, {match['ensemble']['family']} by "
            f"family); unmatched BANC neurons {len(e['unmatched_banc'])}, unmatched FlyWire neurons {len(e['unmatched_fw'])}.",
            f"- `ensemble_shuffled`: {len(s['units'])} units (by source family); unmatched BANC neurons "
            f"{len(s['unmatched_banc'])}, unmatched FlyWire neurons {len(s['unmatched_fw'])}. Fixed permutation "
            f"(Sattolo, seed {C.PERMUTATION_SEED}) as listed in prereg/10 section 5.",
            "- Units matched by family: " + ("; ".join(fam_units) if fam_units else "none") + ".",
            f"- Peak RAM of the VNC stage {d['peak_rss_gb']:.2f} GB, wall {d['total_wall_s']:.0f} s.", ""]


def gates_section(d, out):
    g1 = sum(r["spikes"] for r in d["gate1"])
    ig = sum(r["ignited"] for r in d["gate2_runs"])
    n2 = len(d["gate2_runs"])
    lo, hi = cp(ig, n2)
    net = {f"{a}|{s}": float(np.mean([r["network_rate_hz"] for r in runs(d["replay"], a, s)])) for a in C.ARMS for s in STIMS}
    g2net = float(np.mean([r["network_rate_hz"] for r in d["gate2_runs"]]))
    mx_key = max(net, key=net.get)
    mx = max(max(net.values()), g2net)
    ok = g1 == 0 and ig == 0 and mx < 5.0
    out["gates"] = dict(gate1_spikes=g1, gate2_ignited=ig, gate2_n=n2, gate2_ci=[lo, hi], network=net,
                        gate2_network=g2net, gate3_max=mx, passed=ok)
    return ["## 2. Gates (VNC side; prereg/10 section 10)", "",
            f"- Gate 1 (no input, seeds 0-9): {g1} spikes: **{'pass' if g1 == 0 else 'FAIL'}**.",
            f"- Gate 2 (DNg02 and all {d['ensemble_banc']['n']} ensemble neurons at 100 Hz, 200 ms cut, seeds 200-249): "
            f"{ig} / {n2} ignited (Clopper-Pearson 95 % CI {lo:.3f}-{hi:.3f}): **{'pass' if ig == 0 else 'FAIL'}**.",
            f"- Gate 3' (network rate < 5 Hz): maximum {mx:.4f} Hz (replays: {net[mx_key]:.4f} Hz in `{mx_key}`; "
            f"gate-2 condition {g2net:.4f} Hz): **{'pass' if mx < 5.0 else 'FAIL'}**.", ""]


def brain_rates(d):
    br = d["brain"]
    return {f"{st}|{f}|{s}": [x["rate_hz"][f"{f}|{s}"] for x in sorted(br, key=lambda x: x["seed"]) if x["stimulus"] == st]
            for st in STIMS for f in C.ENSEMBLE for s in SIDES}


def brain_section(d, out):
    rates = brain_rates(d)

    def dbr(f):
        return [(a - b) - (c - e) for a, b, c, e in zip(rates[f"rot_right|{f}|right"], rates[f"rot_right|{f}|left"],
                                                        rates[f"rot_left|{f}|right"], rates[f"rot_left|{f}|left"])]
    t = test_block({f: dbr(f) for f in C.ENSEMBLE})
    sel = [f for f in C.ENSEMBLE if t[f]["sig"]]
    prev = json.loads(F3B1_REPORT.read_text())["brain"]
    prev_p = {**{f: v["p_t"] for f, v in prev["tests"].items()}, "DNp31": prev["dnp31"]["p_t"]}
    same = {f: bool(abs(t[f]["p_t"] - p) <= 1e-12 * max(1.0, abs(p))) for f, p in prev_p.items()}
    silent = [f for f in C.ENSEMBLE if all(max(rates[f"{st}|{f}|{s}"]) == 0 for st in STIMS for s in SIDES)]
    out["brain"] = dict(rates={k: [float(x) for x in v] for k, v in rates.items()}, tests=t, selective=sel,
                        f3b1_p_t=prev_p, f3b1_same_p=same, silent=silent)
    L = ["## 3. Brain side: which ensemble families carry a direction-selective signal (`BRAIN`; prereg/10 section 9)", "",
         "D_brain = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the FlyWire per-neuron rate (1 s stimulus window), "
         f"paired over seeds 0-9; t-test, Holm over the {len(C.ENSEMBLE)} families.", "",
         f"- Direction-selective after Holm: **{len(sel)}** of {len(C.ENSEMBLE)}: "
         + (", ".join(f"{f} ({'+' if t[f]['mean'] > 0 else '-'})" for f in sel) if sel else "none") + ".",
         f"- Families silent on both sides under every stimulus (all seeds): {len(silent)}.",
         "- Consistency with F3b-1 (uncorrected t-test p equal): "
         + ", ".join(f"{f} {'yes' if v else 'NO'}" for f, v in same.items()) + ".", "",
         "Families with p (t, uncorrected) < 0.05 (the full 112-row table is in the JSON):", "",
         "| family | rot_left L / R (Hz) | rot_right L / R (Hz) | mean D (Hz) | s.d. | p (t) | p Holm | selective |",
         "|---|---|---|---|---|---|---|---|"]
    for f in sorted([f for f in C.ENSEMBLE if t[f]["p_t"] < 0.05], key=lambda f: t[f]["p_t"]):
        v = t[f]
        rl = "/".join(f"{np.mean(rates[f'rot_left|{f}|{s}']):.2f}" for s in SIDES)
        rr = "/".join(f"{np.mean(rates[f'rot_right|{f}|{s}']):.2f}" for s in SIDES)
        L.append(f"| {f} | {rl} | {rr} | {v['mean']:.2f} | {v['sd']:.2f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} | "
                 f"{'yes' if v['sig'] else 'no'} |")
    L.append("")
    return L


def mn_tests(d, types):
    rp = d["replay"]
    tests, verdict = {}, {}
    for a in C.ARMS:
        dv = {m: [lr(x["rate_hz"], m) - lr(y["rate_hz"], m) for x, y in zip(runs(rp, a, "rot_right"), runs(rp, a, "rot_left"))]
              for m in types}
        tests[a] = test_block(dv)
        verdict[a] = any(v["sig"] for v in tests[a].values())
    spec = {a: dict(selective=verdict[a], controls={c: verdict[c] for c in C.CONTROLS[a]},
                    specific=bool(verdict[a] and not any(verdict[c] for c in C.CONTROLS[a]))) for a in C.CONTROLS}
    return tests, verdict, spec


def test_table(tests):
    L = ["| arm | MN | mean D (Hz) | s.d. | p (t) | p Holm | significant | p (Wilcoxon) | p Wilcoxon Holm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, tt in tests.items():
        for m, v in tt.items():
            L.append(f"| {a} | {m} | {v['mean']:.2f} | {v['sd']:.2f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} | "
                     f"{'yes' if v['sig'] else 'no'} | {v['p_w']:.4g} | {v['p_w_holm']:.4g} |")
    return L


def verdict_lines(tests, verdict, spec):
    L = ["| arm | direction-selective steering | significant types (sign of mean D) |", "|---|---|---|"]
    for a in C.ARMS:
        sig = [f"{m} ({'+' if v['mean'] > 0 else '-'})" for m, v in tests[a].items() if v["sig"]]
        L.append(f"| {a} | {'**yes**' if verdict[a] else 'no'} | {', '.join(sig) if sig else '-'} |")
    L += [""]
    for a, v in spec.items():
        ctl = ", ".join(f"`{c}` {'selective' if s else 'not selective'}" for c, s in v["controls"].items())
        res = ("direction-selective and **specific**" if v["specific"] else
               "direction-selective but **not specific**" if v["selective"] else "**not direction-selective**")
        L.append(f"- `{a}`: {res} ({ctl}).")
    return L


def vnc_section(d, out):
    tests, verdict, spec = mn_tests(d, C.PRIMARY_MN)
    rp = d["replay"]
    base = {f"{a}|{st}|{m}": [lr(x["rate_hz"], m) for x in runs(rp, a, st)] for a in C.ARMS for st in STIMS
            for m in C.PRIMARY_MN}
    out["vnc"] = dict(tests=tests, selective=verdict, specificity=spec,
                      r_minus_l={k: [float(x) for x in v] for k, v in base.items()})
    L = ["## 4. Primary test: direction-selective steering in the BANC VNC (`VNC`; prereg/10 section 7)", "",
         "D = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the per-neuron MN rate (1 s stimulus window), paired over seeds "
         "0-9; two-sided t-test, Holm over the 5 MN types per arm (alpha 0.05). Wilcoxon (Holm) is the sensitivity "
         "analysis.", ""]
    L += test_table(tests) + [""] + verdict_lines(tests, verdict, spec) + [""]
    L += ["(R - L) per stimulus (Hz, mean ± s.d. over seeds; `static` and `grey` are the baseline, descriptive):", "",
          "| arm | MN | " + " | ".join(STIMS) + " |", "|---|---|" + "---|" * len(STIMS)]
    for a in C.ARMS:
        for m in C.PRIMARY_MN:
            L.append(f"| {a} | {m} | " + " | ".join(ms(base[f"{a}|{st}|{m}"]) for st in STIMS) + " |")
    L.append("")
    return L


def secondary_section(d, out):
    gate = [m for m in C.STEER_MN if d["symmetry"].get(m, {}).get("passed")]
    failing = [m for m in C.STEER_MN if m not in gate]
    tests, verdict, spec = mn_tests(d, gate)
    out["secondary"] = dict(types=gate, gate_failing=failing, tests=tests, selective=verdict, specificity=spec)
    L = ["## 5. Secondary test: all gate-passing steering MN types (prereg/10 section 8; not used for the decision)", "",
         f"Types passing the symmetry gate on the subgraph ({len(gate)}): {', '.join(gate)}; failing: "
         f"{', '.join(failing) if failing else 'none'}. Same D and t-test, Holm over the {len(gate)} types per arm.", ""]
    L += verdict_lines(tests, verdict, spec) + ["", "Types with p (t, uncorrected) < 0.05:", ""]
    small = {a: {m: v for m, v in tt.items() if v["p_t"] < 0.05} for a, tt in tests.items()}
    rows = test_table({a: tt for a, tt in small.items() if tt})
    L += (rows if len(rows) > 2 else ["none"]) + [""]
    return L


def drive_section(d, out):
    rp = d["replay"]
    keys = sorted(d["ensemble_banc"]["family_side_n"])
    arms = ("ensemble", "ensemble_shuffled")

    def m(a, st, field, k):
        return float(np.mean([r[field][k] for r in runs(rp, a, st)]))
    app = {f"{a}|{st}|{k}": m(a, st, "dn_applied_hz", k) for a in arms for st in STIMS for k in keys}
    rea = {f"{a}|{st}|{k}": m(a, st, "dn_realised_hz", k) for a in arms for st in STIMS for k in keys}
    mx = {f"{a}|{k}": max(app[f"{a}|{st}|{k}"] for st in STIMS) for a in arms for k in keys}
    used = {st: [r["transfer"]["fraction_all_dn_used"] for r in runs(rp, "ensemble", st)] for st in STIMS}
    n_dn_spk = {st: int(sum(r["transfer"]["spikes_all_dn"] for r in runs(rp, "ensemble", st))) for st in STIMS}
    lost = max((r["transfer"]["fraction_lost"] for r in rp if r["transfer"]), default=0.0)
    sel = set(out["brain"]["selective"])
    fam_max = {f: max(mx.get(f"ensemble|{f}|{s}", 0.0) for s in SIDES) for f in C.ENSEMBLE}
    bands = {"0": sum(v == 0 for v in fam_max.values()), "(0, 10)": sum(0 < v < 10 for v in fam_max.values()),
             "[10, 25)": sum(10 <= v < 25 for v in fam_max.values()), "[25, 50)": sum(25 <= v < 50 for v in fam_max.values()),
             ">= 50": sum(v >= 50 for v in fam_max.values())}
    tot = {f"{a}|{st}": float(np.mean([sum(r["dn_applied_hz"][k] * d["ensemble_banc"]["family_side_n"][k] for k in keys)
                                        for r in runs(rp, a, st)])) for a in arms for st in STIMS}
    out["drive"] = dict(applied=app, realised=rea, max_applied=mx, family_max=fam_max, bands=bands,
                        total_applied_hz=tot, fraction_all_dn_used={st: [float(min(v)), float(max(v))] for st, v in used.items()},
                        fw_dn_spikes=n_dn_spk,
                        max_fraction_lost=lost)
    L = ["## 6. Applied DN rates per ensemble family (prereg/10 section 6)", "",
         "- Fraction of all FlyWire DN spikes (whole 1.5 s record) passed to the VNC (arm `ensemble`, range over seeds): "
         + "; ".join(f"{st} {min(used[st]):.4f}-{max(used[st]):.4f} ({n_dn_spk[st]} DN spikes over 10 seeds)" for st in STIMS)
         + f". FlyWire spikes of the ensemble families without a BANC unit: maximum fraction {lost:.3f}.",
         "- Summed applied input over all ensemble neurons (per-neuron rate x neuron count, mean over seeds, Hz): "
         + "; ".join(f"`{a}` " + ", ".join(f"{st} {tot[f'{a}|{st}']:.0f}" for st in STIMS) for a in arms) + ".",
         "- Families by their highest applied per-neuron rate in `ensemble` (over sides and stimuli, mean over seeds): "
         + ", ".join(f"{k} Hz: {v}" for k, v in bands.items()) + ".", "",
         "Per family and side (arm `ensemble`; Hz, mean over seeds; `applied / realised`; * = direction-selective in the "
         "brain after Holm):", "",
         "| family | side | n BANC | rot_left | rot_right | static | grey | max applied (`ensemble_shuffled`) |",
         "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        f, s = k.split("|")
        L.append(f"| {f}{' *' if f in sel else ''} | {s} | {d['ensemble_banc']['family_side_n'][k]} | "
                 + " | ".join(f"{app[f'ensemble|{st}|{k}']:.1f} / {rea[f'ensemble|{st}|{k}']:.1f}" for st in STIMS)
                 + f" | {mx[f'ensemble_shuffled|{k}']:.1f} |")
    L.append("")
    return L


def bridge_off_section(d, out):
    """Descriptive: bridge_off here against the F3b-1 bridge_off (prereg/10 section 5)."""
    old = [r for r in json.loads(F3B1.read_text())["replay"] if r["arm"] == "bridge_off"]
    new = [r for r in d["replay"] if r["arm"] == "bridge_off"]
    rows = {}
    for mn in C.PRIMARY_MN:
        for s in SIDES:
            k = f"{mn}|{s}"
            rows[k] = dict(f3b2=float(np.mean([r["rate_hz"][k] for r in new])), f3b1=float(np.mean([r["rate_hz"][k] for r in old])))
    out["bridge_off_vs_f3b1"] = rows
    return ["## 7. `bridge_off` against F3b-1 `bridge_off` (descriptive)", "",
            "Per-neuron rate of the primary MN types, mean over the 4 stimuli x 10 seeds (Hz, F3b-2 / F3b-1): "
            + "; ".join(f"{k} {v['f3b2']:.2f} / {v['f3b1']:.2f}" for k, v in rows.items()) + ".", ""]


def post_hoc(out):
    """Post-hoc (not registered): sign pattern of the significant primary types in `ensemble` and `ensemble_shuffled`,
    and the (R - L) baseline under `static`."""
    t = out["vnc"]["tests"]
    sig = {a: {m: v["mean"] for m, v in t[a].items() if v["sig"]} for a in ("ensemble", "ensemble_shuffled")}
    both = [m for m in C.PRIMARY_MN if m in sig["ensemble"] and m in sig["ensemble_shuffled"]]
    opp = [m for m in both if np.sign(sig["ensemble"][m]) != np.sign(sig["ensemble_shuffled"][m])]
    rl = out["vnc"]["r_minus_l"]
    stat = {m: float(np.mean(rl[f"ensemble|static|{m}"])) for m in C.PRIMARY_MN}
    grey_zero = all(not any(rl[f"{a}|grey|{m}"]) for a in C.ARMS for m in C.PRIMARY_MN)
    out["post_hoc"] = dict(significant=sig, both=both, opposite_sign=opp, static_r_minus_l=stat, grey_all_zero=grey_zero)
    fmt = lambda a: ", ".join(f"{m} {sig[a][m]:+.2f}" for m in sig[a]) or "none"  # noqa: E731
    return ["## 9. Post-hoc (not registered in prereg/10)", "",
            f"- Mean D of the significant primary types (Hz): `ensemble` {fmt('ensemble')}; `ensemble_shuffled` "
            f"{fmt('ensemble_shuffled')}. Significant in both arms: {', '.join(both) if both else 'none'}; with opposite "
            f"signs: {', '.join(opp) if opp else 'none'}.",
            "- Reading: the shuffled control keeps the brain's direction-selective rate streams (section 3) and only moves "
            "them to other BANC families. So it tests whether the effect needs the correct family-to-family wiring, and "
            "it does not test whether direction information reaches the VNC at all. Under the registered rule, a "
            "direction-selective shuffled arm makes the `ensemble` result not specific; the rule is not changed here.",
            "- (R - L) of the primary types under `static` in `ensemble` (mean over seeds, Hz): "
            + ", ".join(f"{m} {v:+.2f}" for m, v in stat.items())
            + f". Under `grey` every primary type is silent in every arm: {'yes' if grey_zero else 'no'} (the FlyWire DNs "
            "fire no spikes under `grey`).", ""]


def summary(out):
    b, v, g, s2 = out["brain"], out["vnc"], out["gates"], out["secondary"]
    sp = v["specificity"]["ensemble"]
    outcome = "A" if sp["specific"] else "B" if sp["selective"] else "C"
    out["outcome"] = outcome
    text = {"A": "`ensemble` is direction-selective and specific: explanation 1 is supported at gain 1.",
            "B": "`ensemble` is direction-selective but not specific: this does not support explanation 1.",
            "C": "`ensemble` is not direction-selective: explanation 1 (too few bridged DNs) is not supported at gain 1 "
                 "with these bridge rules. As instructed, the work stops here; the next step (neuron model v2) is the "
                 "project owner's decision."}[outcome]
    return ["## 8. Summary", "",
            f"- Gates: {'all pass' if g['passed'] else 'a gate FAILS (tests not used)'}.",
            f"- Brain: {len(b['selective'])} of {len(C.ENSEMBLE)} ensemble families carry a direction-selective signal "
            "after Holm" + (f" ({', '.join(b['selective'])})" if b["selective"] else "") + ".",
            f"- Primary test, outcome **{outcome}** (prereg/10 section 11): {text}",
            "- Secondary test (all gate-passing steering types, not used for the decision): "
            + "; ".join(f"`{a}` {'direction-selective' if s2['selective'][a] else 'not direction-selective'}" for a in C.ARMS) + ".",
            "- Limits stated in prereg/10: open loop, no body, no VNC -> brain feedback; rate coding with Poisson "
            "re-synthesis; DNs without a gate-passing shared family are not bridged; no degree-matched random VNC in "
            "this step.", ""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    md, js = OUT.with_suffix(".md"), OUT.with_suffix(".json")
    if (md.exists() or js.exists()) and not a.overwrite:
        raise SystemExit(f"{md} exists (use --overwrite)")
    d = json.loads(REC.read_text())
    out = {}
    L = ["# F3b-2: DN ensemble bridge, FlyWire brain -> BANC VNC (prereg/10)", "",
         "Generated by `scripts/f3b2_report.py` from `runs/f3b2/f3b2_v3.json` (and, for the comparisons, "
         "`runs/f3b1/f3b1_v3.json` and `docs/results/f3b1_optomotor.json`); checked by `scripts/verify_report.py`. "
         "Pre-registration: `prereg/10_f3b2_dn_ensemble.md`.", ""]
    L += setup_section(d, out)
    L += gates_section(d, out)
    L += brain_section(d, out)
    L += vnc_section(d, out)
    L += secondary_section(d, out)
    L += drive_section(d, out)
    L += bridge_off_section(d, out)
    L += summary(out)
    L += post_hoc(out)
    md.write_text("\n".join(L) + "\n")
    js.write_text(json.dumps(out, indent=1) + "\n")
    print("->", md, js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
