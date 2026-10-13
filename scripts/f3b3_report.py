"""Render docs/results/f3b3_specificity.{md,json} from runs/f3b3/f3b3_v3.json (prereg/11; numbers by script).

    env -u PYTHONPATH python scripts/f3b3_report.py [--overwrite]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from f3a2_report import cp  # noqa: E402
from f3b1_report import lr, ms, test_block  # noqa: E402
import f3b3_common as C  # noqa: E402

REC = ROOT / "runs" / "f3b3" / "f3b3_v3.json"
INPUT_CHECK = ROOT / "runs" / "f3b3" / "input_check.json"
OUT = ROOT / "docs" / "results" / "f3b3_specificity"
STIMS = tuple(C.STIMULI)
SIDES = ("left", "right")
# prereg/11 section 9 (exploratory): expected sign of D under syndirectional steering, from Table 1 of Cheong et al.,
# eLife, doi 10.7554/eLife.96084 (inner side of turn active -> +, outer side -> -; None = not fixed by the literature)
EXPECTED_SIGN = {"b3": 1, "i1": 1, "i2": None, "hg1": 1, "hg4": 1, "b1": None, "b2": -1, "iii1": -1, "iii3": -1,
                 "iii4": None, "hg2": None, "hg3": 1, "tp1": None, "tp2": None, "tpn": None, "ps1": None, "ps2": None}
EXPECTED_BASIS = {"b3": "inner: activity increased / outer: decreased", "i1": "inner: active / outer: inactive",
                  "i2": "minor change on both sides", "hg1": "inner: active / outer: inactive",
                  "hg4": "inner: activity increased / outer: decreased", "b1": "phase delay / phase advance (timing)",
                  "b2": "inner: inactive / outer: active", "iii1": "inner: inactive / outer: active",
                  "iii3": "inner: activity decreased / outer: increased", "iii4": "minor or inactive on both sides",
                  "hg2": "minor or inactive on both sides", "hg3": "inner: active / outer: inactive",
                  "tp1": "not in Table 1", "tp2": "not in Table 1", "tpn": "not in Table 1", "ps1": "not in Table 1",
                  "ps2": "not in Table 1"}


def runs(rp, a, st):
    return sorted([r for r in rp if r["arm"] == a and r["stimulus"] == st], key=lambda r: r["seed"])


def d_values(rp, a, m):
    return [lr(x["rate_hz"], m) - lr(y["rate_hz"], m) for x, y in zip(runs(rp, a, "rot_right"), runs(rp, a, "rot_left"))]


def perm_stats(T_ens, T_null):
    """Permutation p (ties against `ensemble`), count of random wirings with smaller |T|, and same-sign fraction."""
    ge = sum(abs(t) >= abs(T_ens) for t in T_null)
    smaller = sum(abs(t) < abs(T_ens) for t in T_null)
    same = sum(np.sign(t) == np.sign(T_ens) and t != 0 for t in T_null) if T_ens != 0 else 0
    return dict(p_perm=(1 + ge) / (len(T_null) + 1), n_smaller=int(smaller), n_ge=int(ge),
                same_sign_fraction=float(same / len(T_null)))


def setup_section(d, chk, out):
    e = d["bridges"]["ensemble"]
    br = d["brain_records"]
    rnd = {a: d["bridges"][a] for a in C.RANDOM_ARMS}
    match = {m: sum(u["match"] == m for u in e["units"]) for m in ("type", "family")}
    r_units = sorted({len(b["units"]) for b in rnd.values()})
    r_unm = max(max(len(b["unmatched_banc"]), len(b["unmatched_fw"])) for b in rnd.values())
    out["setup"] = dict(units_ensemble=len(e["units"]), match=match, random_units=r_units, random_max_unmatched=r_unm,
                        n_banc=d["ensemble_banc"]["n"], by_side=d["ensemble_banc"]["by_side"],
                        unmatched_banc=len(e["unmatched_banc"]), unmatched_fw=len(e["unmatched_fw"]),
                        brain_files=len(br["sha256"]), n_dn=br["n_dn"], permutations_sha256=d["constants"]["permutations_sha256"],
                        input_check_passed=chk["passed"], equals_f3b1_render=chk.get("equals_f3b1_render"),
                        a_in={k: v["A_in_T4aT5a"] for k, v in chk["per_stimulus"].items()})
    return ["## 1. Setup (prereg/11 sections 2-5)", "",
            f"- Fresh brain runs: {len(br['sha256'])} records (4 stimuli x seeds {C.SEEDS[0]}-{C.SEEDS[-1]}), "
            f"{br['n_dn']:,} FlyWire DNs each, in `{br['dir']}`; SHA-256 of every file recorded and checked by "
            "`scripts/verify_report.py`. No F3b-1 record is used.",
            "- Input check (prereg/09 section 3): " + ("passed" if chk["passed"] else "FAILED") + "; A_in (T4a/T5a) "
            + ", ".join(f"{k} {v['A_in_T4aT5a']:+.4f}" for k, v in chk["per_stimulus"].items())
            + f". The new render equals the F3b-1 render bit for bit: {'yes' if chk.get('equals_f3b1_render') else 'no'}.",
            f"- BANC VNC subgraph {d['subgraph']['n_neurons']:,} neurons, {d['subgraph']['pairs']:,} pairs, "
            f"{d['subgraph']['synapses']:,} synapses; reference arm. Bridge latency {e['config']['latency_intervals']} "
            f"interval ({C.INTERVAL_MS:g} ms), window {e['config']['window_intervals']} intervals, per-neuron mean rate, "
            "independent Poisson re-synthesis, gain 1.",
            f"- Ensemble: {len(C.ENSEMBLE)} families, {d['ensemble_banc']['n']} BANC neurons "
            f"({d['ensemble_banc']['by_side'].get('left', 0)} left, {d['ensemble_banc']['by_side'].get('right', 0)} right); "
            f"DNg02 `HAND` {C.BASELINE_HZ:g} Hz both sides; every other BANC DN silent.",
            f"- `ensemble`: {len(e['units'])} units ({match['type']} by exact type, {match['family']} by family); unmatched "
            f"BANC {len(e['unmatched_banc'])}, FlyWire {len(e['unmatched_fw'])}.",
            f"- `random_1` ... `random_{C.N_RANDOM}`: units per wiring {', '.join(map(str, r_units))}; at most {r_unm} "
            f"unmatched neurons. Permutation SHA-256 `{d['constants']['permutations_sha256']}` (prereg/11 section 5).",
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
                        gate2_network=g2net, gate3_max=mx, gate3_max_key=mx_key, passed=ok)
    return ["## 2. Gates (VNC side; prereg/11 section 10)", "",
            f"- Gate 1 (no input, seeds {C.GATE1_SEEDS[0]}-{C.GATE1_SEEDS[-1]}): {g1} spikes: "
            f"**{'pass' if g1 == 0 else 'FAIL'}**.",
            f"- Gate 2 (DNg02 and all {d['ensemble_banc']['n']} ensemble neurons at 100 Hz, 200 ms cut, seeds "
            f"{C.GATE2_SEEDS[0]}-{C.GATE2_SEEDS[-1]}): {ig} / {n2} ignited (Clopper-Pearson 95 % CI {lo:.3f}-{hi:.3f}): "
            f"**{'pass' if ig == 0 else 'FAIL'}**.",
            f"- Gate 3' (network rate < 5 Hz, {len(C.ARMS)} arms x {len(STIMS)} stimuli and the gate-2 condition): maximum "
            f"{mx:.4f} Hz (replays: {net[mx_key]:.4f} Hz in `{mx_key}`; gate-2 condition {g2net:.4f} Hz): "
            f"**{'pass' if mx < 5.0 else 'FAIL'}**.", ""]


def brain_section(d, out):
    br = sorted(d["brain"], key=lambda x: x["seed"])
    rates = {f"{st}|{f}|{s}": [x["rate_hz"][f"{f}|{s}"] for x in br if x["stimulus"] == st]
             for st in STIMS for f in C.ENSEMBLE for s in SIDES}

    def dbr(f):
        return [(a - b) - (c - e) for a, b, c, e in zip(rates[f"rot_right|{f}|right"], rates[f"rot_right|{f}|left"],
                                                        rates[f"rot_left|{f}|right"], rates[f"rot_left|{f}|left"])]
    t = test_block({f: dbr(f) for f in C.ENSEMBLE})
    sel = [f for f in C.ENSEMBLE if t[f]["sig"]]
    silent = [f for f in C.ENSEMBLE if all(max(rates[f"{st}|{f}|{s}"]) == 0 for st in STIMS for s in SIDES)]
    out["brain"] = dict(rates={k: [float(x) for x in v] for k, v in rates.items()}, tests=t, selective=sel, silent=silent)
    L = ["## 3. Brain side on the fresh seeds (`BRAIN`; prereg/11 section 8.3, descriptive)", "",
         "D_brain = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the FlyWire per-neuron rate (1 s stimulus window), "
         f"paired over seeds {C.SEEDS[0]}-{C.SEEDS[-1]}; t-test, Holm over the {len(C.ENSEMBLE)} families.", "",
         f"- Direction-selective after Holm: **{len(sel)}** of {len(C.ENSEMBLE)}: "
         + (", ".join(f"{f} ({'+' if t[f]['mean'] > 0 else '-'})" for f in sel) if sel else "none") + ".",
         f"- Families silent on both sides under every stimulus (all seeds): {len(silent)}.", "",
         "Direction-selective families (the full 112-row table is in the JSON):", "",
         "| family | rot_left L / R (Hz) | rot_right L / R (Hz) | mean D (Hz) | s.d. | p (t) | p Holm |", "|---|---|---|---|---|---|---|"]
    for f in sorted(sel, key=lambda f: t[f]["p_t"]):
        v = t[f]
        rl = "/".join(f"{np.mean(rates[f'rot_left|{f}|{s}']):.2f}" for s in SIDES)
        rr = "/".join(f"{np.mean(rates[f'rot_right|{f}|{s}']):.2f}" for s in SIDES)
        L.append(f"| {f} | {rl} | {rr} | {v['mean']:.2f} | {v['sd']:.2f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} |")
    L.append("")
    return L


def mn_block(d, types):
    """Per type: t-tests of `ensemble` and `bridge_off` (Holm over ``types``), T per arm, permutation statistics."""
    rp = d["replay"]
    tests = {a: test_block({m: d_values(rp, a, m) for m in types}) for a in ("ensemble", "bridge_off")}
    T = {a: {m: float(np.mean(d_values(rp, a, m))) for m in types} for a in C.ARMS}
    perm = {m: perm_stats(T["ensemble"][m], [T[a][m] for a in C.RANDOM_ARMS]) for m in types}
    return tests, T, perm


def primary_section(d, out):
    tests, T, perm = mn_block(d, C.PRIMARY_MN)
    te, tb = tests["ensemble"], tests["bridge_off"]
    a_sig = [m for m in C.PRIMARY_MN if te[m]["sig"]]
    b_ok = [m for m in a_sig if perm[m]["p_perm"] <= C.PERM_ALPHA]
    bo_sel = any(v["sig"] for v in tb.values())
    bo_zero = all(not any(v["d"]) for v in tb.values())
    rule_a, rule_b = bool(a_sig), bool(b_ok)
    gates = out["gates"]["passed"]
    if not rule_a:
        outcome = "not replicated"
    elif rule_b and not bo_sel and gates:
        outcome = "specific"
    else:
        outcome = "replicated, not specific"
    reason = ("" if outcome != "replicated, not specific" else
              "(b) fails" if not rule_b else "`bridge_off` is direction-selective" if bo_sel else "a gate fails")
    out["primary"] = dict(tests=tests, T=T, perm=perm, rule_a=rule_a, rule_a_types=a_sig, rule_b=rule_b,
                          rule_b_types=b_ok, bridge_off_selective=bo_sel, bridge_off_all_zero=bo_zero,
                          outcome=outcome, reason=reason)
    L = ["## 4. Primary test (`VNC`; prereg/11 sections 6-7)", "",
         f"D_m = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the per-neuron MN rate (1 s stimulus window), per seed "
         f"({len(C.SEEDS)} seeds); T_m = mean of D_m over the seeds.", "",
         "**(a) Replication** (two-sided one-sample t-test, Holm over the 5 types; Wilcoxon (Holm) as sensitivity "
         "analysis, not used):", "",
         "| arm | MN | T (Hz) | s.d. of D | p (t) | p Holm | significant | p (Wilcoxon) | p Wilcoxon Holm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, tt in tests.items():
        for m, v in tt.items():
            L.append(f"| {a} | {m} | {v['mean']:+.3f} | {v['sd']:.3f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} | "
                     f"{'yes' if v['sig'] else 'no'} | {v['p_w']:.4g} | {v['p_w_holm']:.4g} |")
    sig_txt = ", ".join(f"{m} ({te[m]['mean']:+.3f} Hz)" for m in a_sig) if a_sig else "none"
    L += ["", f"- (a) holds: **{'yes' if rule_a else 'no'}** (significant in `ensemble`: {sig_txt}).",
          f"- `bridge_off` direction-selective: **{'yes' if bo_sel else 'no'}** (D identically 0 for every type: "
          f"{'yes' if bo_zero else 'no'}).", "",
          f"**(b) Specificity (magnitude)** against the {C.N_RANDOM} random wirings; p_perm = (1 + #{{k: |T_m(random_k)| >= "
          f"|T_m(ensemble)|}}) / {C.N_RANDOM + 1}, criterion <= {C.PERM_ALPHA:g}; **(c) sign specificity** (secondary): "
          "fraction of random wirings with the same sign of T_m as `ensemble`.", "",
          "| MN | T (ensemble, Hz) | significant in (a) | random wirings with smaller abs(T) | p_perm | (b) met | same-sign fraction (c) |",
          "|---|---|---|---|---|---|---|"]
    for m in C.PRIMARY_MN:
        p = perm[m]
        L.append(f"| {m} | {T['ensemble'][m]:+.3f} | {'yes' if te[m]['sig'] else 'no'} | {p['n_smaller']} / {C.N_RANDOM} | "
                 f"{p['p_perm']:.3f} | {'yes' if m in b_ok else 'no'} | {p['same_sign_fraction']:.3f} |")
    L += ["", f"- (b) holds: **{'yes' if rule_b else 'no'}** (types significant in (a) with p_perm <= {C.PERM_ALPHA:g}: "
          f"{', '.join(b_ok) if b_ok else 'none'}).",
          f"- **Outcome: {outcome}**" + (f" ({reason})" if reason else "") + ".", ""]
    return L


def null_section(out):
    T, perm = out["primary"]["T"], out["primary"]["perm"]
    null = {m: sorted(T[a][m] for a in C.RANDOM_ARMS) for m in C.PRIMARY_MN}
    q = {m: dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                     [float(x) for x in np.quantile(null[m], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])])) for m in C.PRIMARY_MN}
    out["null"] = dict(sorted=null, quantiles=q)
    L = ["## 5. Null distribution of T_m over the 39 random wirings (prereg/11 section 8.1)", "",
         "Quantiles (Hz; numpy linear interpolation) and the position of `ensemble`:", "",
         "| MN | min | 5 % | 25 % | median | 75 % | 95 % | max | T (ensemble) | random wirings with smaller abs(T) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in C.PRIMARY_MN:
        v = q[m]
        L.append(f"| {m} | " + " | ".join(f"{v[k]:+.3f}" for k in ("min", "q05", "q25", "median", "q75", "q95", "max"))
                 + f" | {T['ensemble'][m]:+.3f} | {perm[m]['n_smaller']} / {C.N_RANDOM} |")
    L += ["", "All 39 values per type, sorted (Hz; rank 1 = most negative):", "",
          "| rank | " + " | ".join(C.PRIMARY_MN) + " |", "|---|" + "---|" * len(C.PRIMARY_MN)]
    for i in range(C.N_RANDOM):
        L.append(f"| {i + 1} | " + " | ".join(f"{null[m][i]:+.3f}" for m in C.PRIMARY_MN) + " |")
    L.append("")
    return L


def secondary_section(d, out):
    gate = [m for m in C.STEER_MN if d["symmetry"].get(m, {}).get("passed")]
    failing = [m for m in C.STEER_MN if m not in gate]
    tests, T, perm = mn_block(d, gate)
    out["secondary"] = dict(types=gate, gate_failing=failing, tests=tests, T=T, perm=perm)
    te = tests["ensemble"]
    L = ["## 6. Secondary set: gate-passing steering MN types (prereg/11 section 8.2; not used for the decision)", "",
         f"Types passing the symmetry gate on the subgraph ({len(gate)}): {', '.join(gate)}; failing: "
         f"{', '.join(failing) if failing else 'none'}. Same D, T, t-test (Holm over the {len(gate)} types) and "
         "permutation statistics.", "",
         "| MN | T (ensemble, Hz) | p Holm (ensemble) | significant | p Holm (bridge_off) | p_perm | same-sign fraction |",
         "|---|---|---|---|---|---|---|"]
    for m in gate:
        L.append(f"| {m} | {T['ensemble'][m]:+.3f} | {te[m]['p_t_holm']:.4g} | {'yes' if te[m]['sig'] else 'no'} | "
                 f"{tests['bridge_off'][m]['p_t_holm']:.4g} | {perm[m]['p_perm']:.3f} | {perm[m]['same_sign_fraction']:.3f} |")
    L.append("")
    return L


def exploratory_section(out):
    sec = out["secondary"]
    te, T = sec["tests"]["ensemble"], sec["T"]["ensemble"]
    rows = {}
    for m in C.STEER_MN:
        exp = EXPECTED_SIGN[m]
        if m not in sec["types"]:
            rows[m] = dict(expected=exp, observed=None, significant=None, agree=None, gate=False)
            continue
        obs = int(np.sign(T[m]))
        rows[m] = dict(expected=exp, observed=obs, significant=bool(te[m]["sig"]),
                       agree=(None if exp is None or obs == 0 else bool(obs == exp)), gate=True)
    cmp = [m for m, r in rows.items() if r["agree"] is not None]
    n_agree = sum(rows[m]["agree"] for m in cmp)
    sig_cmp = [m for m in cmp if rows[m]["significant"]]
    n_agree_sig = sum(rows[m]["agree"] for m in sig_cmp)
    out["exploratory"] = dict(rows=rows, compared=cmp, n_agree=n_agree, significant_compared=sig_cmp,
                              n_agree_significant=n_agree_sig)
    sg = {1: "+", -1: "-", 0: "0", None: "-"}
    L = ["## 7. Exploratory: expected sign of syndirectional steering (prereg/11 section 9; not used for any decision)", "",
         "Expected sign of D from Table 1 of Cheong et al., eLife, doi 10.7554/eLife.96084 (compiled from Egelhaaf 1989; "
         "Heide 1975; Heide & Götz 1996; Lehmann & Götz 1996; Lindsay, Sustar & Dickinson 2017; Nachtigall & Wilson "
         "1967): a muscle more active on the inner side of the turn predicts D > 0, one more active on the outer side "
         "predicts D < 0. Assumptions: syndirectional turning (`rot_right` -> right turn), `side` is the fly's own side, "
         "MN rate as a proxy for muscle activation.", "",
         "| MN | Table 1 (inner / outer side of turn) | expected sign | observed sign of T (ensemble) | significant (secondary Holm) | agree |",
         "|---|---|---|---|---|---|"]
    for m, r in rows.items():
        exp = "not fixed" if r["expected"] is None else sg[r["expected"]]
        if not r["gate"]:
            L.append(f"| {m} | {EXPECTED_BASIS[m]} | {exp} | not tested (fails the symmetry gate) | - | - |")
            continue
        ag = "-" if r["agree"] is None else ("yes" if r["agree"] else "no")
        L.append(f"| {m} | {EXPECTED_BASIS[m]} | {exp} | {sg[r['observed']]} ({T[m]:+.3f} Hz) | "
                 f"{'yes' if r['significant'] else 'no'} | {ag} |")
    L += ["", f"- Types with a fixed expected sign that pass the gate: {len(cmp)}; observed sign agrees in {n_agree}. "
          f"Among those significant in `ensemble` (secondary Holm): {len(sig_cmp)}, agreeing {n_agree_sig}.",
          "- Limits: Table 1 compiles results across dipterans and methods, mostly not open-loop rotating gratings in "
          "*Drosophila*; the MN rate is only a proxy (b1 acts through firing phase). This is a plausibility check, not "
          "a test.", ""]
    return L


def drive_section(d, out):
    rp = d["replay"]
    keys = sorted(d["ensemble_banc"]["family_side_n"])

    def m(st, field, k):
        return float(np.mean([r[field][k] for r in runs(rp, "ensemble", st)]))
    app = {f"{st}|{k}": m(st, "dn_applied_hz", k) for st in STIMS for k in keys}
    rea = {f"{st}|{k}": m(st, "dn_realised_hz", k) for st in STIMS for k in keys}
    fam_max = {f: max(app.get(f"{st}|{f}|{s}", 0.0) for st in STIMS for s in SIDES) for f in C.ENSEMBLE}
    bands = {"0": sum(v == 0 for v in fam_max.values()), "(0, 10)": sum(0 < v < 10 for v in fam_max.values()),
             "[10, 25)": sum(10 <= v < 25 for v in fam_max.values()), "[25, 50)": sum(25 <= v < 50 for v in fam_max.values()),
             ">= 50": sum(v >= 50 for v in fam_max.values())}
    steer = {f"{a}|{st}": float(np.mean([r["steer_applied_mean_hz"] for r in runs(rp, a, st)])) for a in C.ARMS for st in STIMS}
    rnd = {st: [steer[f"{a}|{st}"] for a in C.RANDOM_ARMS] for st in STIMS}
    lost = max((r["transfer"]["fraction_lost"] for r in rp if r["transfer"]), default=0.0)
    sel = set(out["brain"]["selective"])
    out["drive"] = dict(applied=app, realised=rea, family_max=fam_max, bands=bands, steer_applied_mean=steer,
                        max_fraction_lost=lost)
    L = ["## 8. Applied DN rates (prereg/11 section 8.4)", "",
         "- Mean applied rate over the 305 bridge targets (stimulus window, mean over seeds, Hz): `ensemble` "
         + ", ".join(f"{st} {steer[f'ensemble|{st}']:.3f}" for st in STIMS) + "; random wirings (min-max) "
         + ", ".join(f"{st} {min(rnd[st]):.3f}-{max(rnd[st]):.3f}" for st in STIMS) + ".",
         f"- FlyWire spikes of the source families without a BANC unit: maximum fraction {lost:.3f}.",
         "- Families by their highest applied per-neuron rate in `ensemble` (over sides and stimuli, mean over seeds): "
         + ", ".join(f"{k} Hz: {v}" for k, v in bands.items()) + ".", "",
         "Per family and side (arm `ensemble`; Hz, mean over seeds; `applied / realised`; * = direction-selective in the "
         "brain after Holm); families with a non-zero applied rate:", "",
         "| family | side | n BANC | rot_left | rot_right | static | grey |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        f, s = k.split("|")
        if fam_max[f] == 0:
            continue
        L.append(f"| {f}{' *' if f in sel else ''} | {s} | {d['ensemble_banc']['family_side_n'][k]} | "
                 + " | ".join(f"{app[f'{st}|{k}']:.1f} / {rea[f'{st}|{k}']:.1f}" for st in STIMS) + " |")
    L.append("")
    return L


def network_section(d, out):
    net = out["gates"]["network"]
    nd = {f"{a}|{st}": float(np.mean([r["nondriven_rate_hz"] for r in runs(d["replay"], a, st)])) for a in C.ARMS for st in STIMS}
    out["nondriven"] = nd
    rn = {st: [net[f"{a}|{st}"] for a in C.RANDOM_ARMS] for st in STIMS}
    L = ["## 9. Network and non-driven rates (prereg/11 section 8.5)", "",
         "| stimulus | network `ensemble` | network `bridge_off` | network random (min-max) | non-driven `ensemble` | non-driven `bridge_off` |",
         "|---|---|---|---|---|---|"]
    for st in STIMS:
        L.append(f"| {st} | {net[f'ensemble|{st}']:.3f} | {net[f'bridge_off|{st}']:.3f} | {min(rn[st]):.3f}-{max(rn[st]):.3f} | "
                 f"{nd[f'ensemble|{st}']:.3f} | {nd[f'bridge_off|{st}']:.3f} |")
    L += ["", "(R - L) of the primary types per stimulus in `ensemble` (Hz, mean ± s.d. over seeds; descriptive):", "",
          "| MN | " + " | ".join(STIMS) + " |", "|---|" + "---|" * len(STIMS)]
    rl = {f"{st}|{m}": [lr(x["rate_hz"], m) for x in runs(d["replay"], "ensemble", st)] for st in STIMS for m in C.PRIMARY_MN}
    out["r_minus_l_ensemble"] = rl
    for m in C.PRIMARY_MN:
        L.append(f"| {m} | " + " | ".join(ms(rl[f"{st}|{m}"]) for st in STIMS) + " |")
    L.append("")
    return L


def summary(out):
    p, g, b, ex = out["primary"], out["gates"], out["brain"], out["exploratory"]
    text = {"specific": "the true wiring replicates a direction-selective response and its magnitude exceeds the random "
                        "wirings ((a) and (b) hold, `bridge_off` silent).",
            "replicated, not specific": "the true wiring replicates a direction-selective response, but the registered "
                                        f"specificity rule is not met ({p['reason']}).",
            "not replicated": "no primary MN type is direction-selective in `ensemble` after Holm on the fresh seeds."}[p["outcome"]]
    return ["## 10. Summary", "",
            f"- Gates: {'all pass' if g['passed'] else 'a gate FAILS (tests not used)'}.",
            f"- Brain (fresh seeds): {len(b['selective'])} of {len(C.ENSEMBLE)} ensemble families direction-selective after Holm.",
            f"- (a) replication: {'yes' if p['rule_a'] else 'no'}"
            + (f" ({', '.join(p['rule_a_types'])})" if p["rule_a_types"] else "") + f"; (b) magnitude specificity: "
            f"{'yes' if p['rule_b'] else 'no'}" + (f" ({', '.join(p['rule_b_types'])})" if p["rule_b_types"] else "")
            + f"; `bridge_off` direction-selective: {'yes' if p['bridge_off_selective'] else 'no'}.",
            f"- **Outcome (prereg/11 section 7): {p['outcome']}**: {text}",
            "- (c) same-sign fraction per primary type: "
            + ", ".join(f"{m} {p['perm'][m]['same_sign_fraction']:.3f}" for m in C.PRIMARY_MN) + " (secondary, not used).",
            f"- Exploratory sign check: {ex['n_agree']} of {len(ex['compared'])} comparable types agree with the "
            "literature-expected sign (not used for any decision).",
            "- Limits stated in prereg/09-11: open loop, no body, no VNC -> brain feedback; rate coding with Poisson "
            "re-synthesis; only the 112 gate-passing shared DN families are bridged; no degree-matched random VNC in "
            "this step.", ""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    md, js = OUT.with_suffix(".md"), OUT.with_suffix(".json")
    if (md.exists() or js.exists()) and not a.overwrite:
        raise SystemExit(f"{md} exists (use --overwrite)")
    d = json.loads(REC.read_text())
    chk = json.loads(INPUT_CHECK.read_text())
    out = {}
    L = ["# F3b-3: confirmatory specificity test of the DN ensemble (prereg/11)", "",
         "Generated by `scripts/f3b3_report.py` from `runs/f3b3/f3b3_v3.json` and `runs/f3b3/input_check.json`; checked "
         "by `scripts/verify_report.py`. Pre-registration: `prereg/11_f3b3_specificity.md`.", ""]
    L += setup_section(d, chk, out)
    L += gates_section(d, out)
    L += brain_section(d, out)
    L += primary_section(d, out)
    L += null_section(out)
    L += secondary_section(d, out)
    L += exploratory_section(out)
    L += drive_section(d, out)
    L += network_section(d, out)
    L += summary(out)
    md.write_text("\n".join(L) + "\n")
    js.write_text(json.dumps(out, indent=1) + "\n")
    print("->", md, js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
