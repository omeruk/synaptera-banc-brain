"""Render docs/results/f3b1_optomotor.{md,json} from runs/f3b1/f3b1_v3.json and runs/f3b1/input_check.json
(prereg/09; numbers by script). Gains recomputed from docs/results/f3a3_vnc_subgraph.json and f3a2_checks.json.

    env -u PYTHONPATH python scripts/f3b1_report.py [--overwrite]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from f3a2_report import ALPHA, cp, holm, p_t, p_w  # noqa: E402
import f3b1_common as C  # noqa: E402

REC = ROOT / "runs" / "f3b1" / "f3b1_v3.json"
CHK = ROOT / "runs" / "f3b1" / "input_check.json"
F3A3 = ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json"
C1 = ROOT / "docs" / "results" / "f3a2_checks.json"
OUT = ROOT / "docs" / "results" / "f3b1_optomotor"
POWER = ("DLM1-4", "DVM1a-c", "tp2", "ps1")
STIMS = tuple(C.STIMULI)


def f2(x):
    return f"{x:.2f}"


def ms(v):
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


def gains():
    lo = json.loads(F3A3.read_text())["r3"]["lowest_rate"]
    c1 = json.loads(C1.read_text())["c1"]["f0"]
    return {f: dict(lowest=lo[f], median=c1[f]["flight"]["median"], g=lo[f] / c1[f]["flight"]["median"])
            for f in C.STEER_FAMILIES}


def test_block(dvals):
    """{key: d list} -> {key: dict(mean, sd, p_t, p_t_holm, p_w, p_w_holm, sig)} with Holm over the keys."""
    keys = list(dvals)
    pt = [p_t(dvals[k]) for k in keys]
    pw = [p_w(dvals[k]) for k in keys]
    ht, hw = holm(pt), holm(pw)
    return {k: dict(d=[float(x) for x in dvals[k]], mean=float(np.mean(dvals[k])), sd=float(np.std(dvals[k], ddof=1)),
                    p_t=pt[i], p_t_holm=float(ht[i]), p_w=pw[i], p_w_holm=float(hw[i]),
                    sig=bool(ht[i] < ALPHA), sig_w=bool(hw[i] < ALPHA)) for i, k in enumerate(keys)}


def lr(rate, m):
    return rate[f"{m}|right"] - rate[f"{m}|left"]


def setup_section(d, out):
    g = gains()
    out["gains"] = g
    same = all(abs(g[f]["g"] - C.RATE_GAIN[f]) < 1e-12 for f in g)
    b = d["bridges"]["ref"]
    L = ["## 1. Setup (prereg/09 sections 2-6)", "",
         f"- Feed-forward replay: the FlyWire brain ran once per stimulus and seed ({len(STIMS)} x {len(C.SEEDS)} runs); "
         f"its recorded DN spikes drove the BANC VNC subgraph ({d['subgraph']['n_neurons']:,} neurons, "
         f"{d['subgraph']['pairs']:,} pairs, {d['subgraph']['synapses']:,} synapses) in every arm. Bridge latency "
         f"{b['config']['latency_intervals']} interval ({C.INTERVAL_MS:g} ms), window {b['config']['window_intervals']} "
         f"intervals, per-neuron mean rate, independent Poisson re-synthesis (spike-timing correlations are lost).",
         f"- DNg02 `HAND` baseline {C.BASELINE_HZ:g} Hz both sides in every arm ({d['groups']['DNg02_L']['n']} L / "
         f"{d['groups']['DNg02_R']['n']} R BANC neurons); DNp31 recorded on the brain side only; every other BANC DN silent.",
         f"- Peak RAM of the VNC stage {d['peak_rss_gb']:.2f} GB, wall {d['total_wall_s']:.0f} s.", "",
         "Mapping table (arm `ref`; labels: DN rates `BRAIN`, MN rates `VNC`):", "",
         "| BANC family | side | match | FlyWire types (n) | BANC neurons |", "|---|---|---|---|---|"]
    for u in sorted(b["units"], key=lambda u: (u["target_family"], u["side"])):
        L.append(f"| {u['target_family']} | {u['side']} | {u['match']} | {', '.join(u['fw_types'])} ({u['n_fw']}) | {u['n_banc']} |")
    L += ["", f"Unmatched BANC neurons of the families: {len(b['unmatched_banc'])}; unmatched FlyWire neurons: "
          f"{len(b['unmatched_fw'])}.", "",
          "Gains of `rate_gain` (recomputed from `f3a3_vnc_subgraph.json` and `f3a2_checks.json`; equal to the registered "
          f"values: {'yes' if same else 'NO'}):", "",
          "| family | lowest steering rate (Hz) | FlyWire flight median (Hz) | g |", "|---|---|---|---|"]
    for f, v in g.items():
        L.append(f"| {f} | {v['lowest']} | {v['median']:.2f} | {v['g']:.3f} |")
    L += ["", "Shuffled controls (side kept): " + ", ".join(f"BANC {t} <- FlyWire {s}" for t, s in C.PERMUTATION.items())
          + "; `shuffled` gain 1, `shuffled_gain` gain of the source family.", ""]
    return L


def input_section(chk, out):
    out["input_check"] = chk
    L = ["## 2. Input check (prereg/09 section 3)", "",
         "A_in = (L - R) / (L + R + 1) of the boundary-layer rates over the 1 s stimulus window.", "",
         "| stimulus | A_in T4a+T5a | A_in all | mean L (Hz) | mean R (Hz) |", "|---|---|---|---|---|"]
    for k, v in chk["per_stimulus"].items():
        L.append(f"| {k} | {v['A_in_T4aT5a']:.4f} | {v['A_in_all']:.4f} | {v['mean_L']:.2f} | {v['mean_R']:.2f} |")
    L += ["", f"Input check: **{'passed' if chk['passed'] else 'FAILED'}** (opposite signs under `rot_left` / "
          f"`rot_right`, |A_in| >= {chk['min_abs']})." , ""]
    return L


def gates_section(d, out):
    g1 = sum(r["spikes"] for r in d["gate1"])
    ig = sum(r["ignited"] for r in d["gate2_runs"])
    n2 = len(d["gate2_runs"])
    lo, hi = cp(ig, n2)
    net = {f"{a}|{s}": float(np.mean([r["network_rate_hz"] for r in d["replay"] if r["arm"] == a and r["stimulus"] == s]))
           for a in C.ARMS for s in STIMS}
    g2net = float(np.mean([r["network_rate_hz"] for r in d["gate2_runs"]]))
    mx_key = max(net, key=net.get)
    mx = max(max(net.values()), g2net)
    ok = g1 == 0 and ig == 0 and mx < 5.0
    out["gates"] = dict(gate1_spikes=g1, gate2_ignited=ig, gate2_n=n2, gate2_ci=[lo, hi], network=net,
                        gate2_network=g2net, gate3_max=mx, passed=ok)
    return ["## 3. Gates (VNC side)", "",
            f"- Gate 1 (no input, seeds 0-9): {g1} spikes: **{'pass' if g1 == 0 else 'FAIL'}**.",
            f"- Gate 2 (DNg02, DNa04, DNa10, DNa15 both sides 100 Hz, 200 ms cut, seeds 200-249): {ig} / {n2} ignited "
            f"(Clopper-Pearson 95 % CI {lo:.3f}-{hi:.3f}): **{'pass' if ig == 0 else 'FAIL'}**.",
            f"- Gate 3' (network rate < 5 Hz): maximum {mx:.4f} Hz (replays: {net[mx_key]:.4f} Hz in `{mx_key}`; "
            f"gate-2 condition {g2net:.4f} Hz): **{'pass' if mx < 5.0 else 'FAIL'}**.", ""]


def brain_section(d, out):
    br = d["brain"]
    fams = C.BRAIN_FAMILIES

    def vals(st, key):
        return [x["rate_hz"][key] for x in sorted(br, key=lambda x: x["seed"]) if x["stimulus"] == st]
    rates = {f"{st}|{f}|{s}": vals(st, f"{f}|{s}") for st in STIMS for f in fams for s in ("left", "right")}
    L = ["## 4. Brain side: FlyWire DN rates (`BRAIN`; 1 s stimulus window, mean ± s.d. over seeds 0-9, Hz)", "",
         "| family | side | " + " | ".join(STIMS) + " |", "|---|---|" + "---|" * len(STIMS)]
    for f in fams:
        for s in ("left", "right"):
            L.append(f"| {f} | {s} | " + " | ".join(ms(rates[f"{st}|{f}|{s}"]) for st in STIMS) + " |")
    netb = {st: [x["net_rate_hz"] for x in br if x["stimulus"] == st] for st in STIMS}
    inp = {st: (np.mean([x["input_mean_L"] for x in br if x["stimulus"] == st]),
                np.mean([x["input_mean_R"] for x in br if x["stimulus"] == st])) for st in STIMS}
    L += ["", "FlyWire network rate (Hz): " + ", ".join(f"{st} {ms(netb[st])}" for st in STIMS) + ". Boundary input "
          "mean L / R (Hz): " + ", ".join(f"{st} {inp[st][0]:.2f} / {inp[st][1]:.2f}" for st in STIMS) + ".", ""]

    def dbr(f):
        rr = [a - b for a, b in zip(rates[f"rot_right|{f}|right"], rates[f"rot_right|{f}|left"])]
        rl = [a - b for a, b in zip(rates[f"rot_left|{f}|right"], rates[f"rot_left|{f}|left"])]
        return [a - b for a, b in zip(rr, rl)]
    t = test_block({f: dbr(f) for f in C.STEER_FAMILIES})
    p31 = dbr("DNp31")
    t31 = dict(d=p31, mean=float(np.mean(p31)), sd=float(np.std(p31, ddof=1)), p_t=p_t(p31), p_w=p_w(p31))
    out["brain"] = dict(rates={k: [float(x) for x in v] for k, v in rates.items()}, tests=t, dnp31=t31,
                        network={k: [float(x) for x in v] for k, v in netb.items()},
                        selective=[f for f in t if t[f]["sig"]])
    L += ["Brain-side check (section 9): D_brain = (R - L)(`rot_right`) - (R - L)(`rot_left`), paired over seeds; "
          "t-test, Holm over the 3 steering families (DNp31 uncorrected, not in the Holm family).", "",
          "| family | mean D (Hz) | s.d. | p (t) | p Holm | direction-selective | p (Wilcoxon) |", "|---|---|---|---|---|---|---|"]
    for f, v in t.items():
        L.append(f"| {f} | {v['mean']:.2f} | {v['sd']:.2f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} | "
                 f"{'yes' if v['sig'] else 'no'} | {v['p_w']:.4g} |")
    L.append(f"| DNp31 | {t31['mean']:.2f} | {t31['sd']:.2f} | {t31['p_t']:.4g} | - | (uncorrected) | {t31['p_w']:.4g} |")
    sel = out["brain"]["selective"]
    L += ["", "Direction-selective in the brain after Holm: " + (", ".join(sel) if sel else "none") + ".", ""]
    return L


def vnc_section(d, out):
    rp = d["replay"]

    def runs(a, st):
        return sorted([r for r in rp if r["arm"] == a and r["stimulus"] == st], key=lambda r: r["seed"])
    tests, verdict = {}, {}
    for a in C.ARMS:
        dv = {m: [lr(x["rate_hz"], m) - lr(y["rate_hz"], m) for x, y in zip(runs(a, "rot_right"), runs(a, "rot_left"))]
              for m in C.PRIMARY_MN}
        tests[a] = test_block(dv)
        verdict[a] = any(v["sig"] for v in tests[a].values())
    spec = {a: dict(selective=verdict[a], controls={c: verdict[c] for c in C.CONTROLS[a]},
                    specific=bool(verdict[a] and not any(verdict[c] for c in C.CONTROLS[a])))
            for a in C.CONTROLS}
    out["vnc"] = dict(tests=tests, selective=verdict, specificity=spec)
    L = ["## 5. Primary test: direction-selective steering in the BANC VNC (`VNC`; prereg/09 section 8)", "",
         "D = (R - L)(`rot_right`) - (R - L)(`rot_left`) of the per-neuron MN rate (1 s stimulus window), paired over seeds "
         "0-9; two-sided t-test, Holm over the 5 MN types per arm (alpha 0.05). Wilcoxon (Holm) is the sensitivity "
         "analysis.", "",
         "| arm | MN | mean D (Hz) | s.d. | p (t) | p Holm | significant | p (Wilcoxon) | p Wilcoxon Holm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in C.ARMS:
        for m, v in tests[a].items():
            L.append(f"| {a} | {m} | {v['mean']:.2f} | {v['sd']:.2f} | {v['p_t']:.4g} | {v['p_t_holm']:.4g} | "
                     f"{'yes' if v['sig'] else 'no'} | {v['p_w']:.4g} | {v['p_w_holm']:.4g} |")
    L += ["", "| arm | direction-selective steering | significant types (sign of mean D) |", "|---|---|---|"]
    for a in C.ARMS:
        sig = [f"{m} ({'+' if v['mean'] > 0 else '-'})" for m, v in tests[a].items() if v["sig"]]
        L.append(f"| {a} | {'**yes**' if verdict[a] else 'no'} | {', '.join(sig) if sig else '-'} |")
    L += ["", "Specificity (the arm's controls must not show it):", ""]
    for a, v in spec.items():
        ctl = ", ".join(f"`{c}` {'selective' if s else 'not selective'}" for c, s in v["controls"].items())
        res = ("direction-selective and **specific**" if v["specific"] else
               "direction-selective but **not specific**" if v["selective"] else "**not direction-selective**")
        L.append(f"- `{a}`: {res} ({ctl}).")
    L.append("")
    # baseline (R - L)
    base = {f"{a}|{st}|{m}": [lr(x["rate_hz"], m) for x in runs(a, st)] for a in C.ARMS for st in STIMS
            for m in C.PRIMARY_MN}
    out["vnc"]["r_minus_l"] = {k: [float(x) for x in v] for k, v in base.items()}
    L += ["(R - L) per stimulus (Hz, mean ± s.d. over seeds; `static` and `grey` are the baseline, descriptive):", "",
          "| arm | MN | " + " | ".join(STIMS) + " |", "|---|---|" + "---|" * len(STIMS)]
    for a in C.ARMS:
        for m in C.PRIMARY_MN:
            L.append(f"| {a} | {m} | " + " | ".join(ms(base[f"{a}|{st}|{m}"]) for st in STIMS) + " |")
    L.append("")
    return L


def drive_section(d, out):
    rp = d["replay"]
    keys = [f"{f}|{s}" for f in C.STEER_FAMILIES for s in ("left", "right")]

    def m(a, st, field, k):
        return float(np.mean([r[field][k] for r in rp if r["arm"] == a and r["stimulus"] == st]))
    app = {f"{a}|{st}|{k}": m(a, st, "dn_applied_hz", k) for a in C.ARMS for st in STIMS for k in keys}
    rea = {f"{a}|{st}|{k}": m(a, st, "dn_realised_hz", k) for a in C.ARMS for st in STIMS for k in keys}
    lost = max((r["transfer"]["fraction_lost"] for r in rp if r["transfer"]), default=0.0)
    used = [r["transfer"]["fraction_all_dn_used"] for r in rp if r["transfer"] and r["arm"] == "ref"]
    out["drive"] = dict(applied=app, realised=rea, max_fraction_lost=lost,
                        fraction_all_dn_used=[float(np.min(used)), float(np.max(used))])
    L = ["## 6. Applied and realised DN rates in the VNC (1 s stimulus window, mean over seeds, Hz; `applied / realised`)", "",
         "| arm | stimulus | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for a in C.ARMS:
        for st in STIMS:
            L.append(f"| {a} | {st} | " + " | ".join(f"{app[f'{a}|{st}|{k}']:.1f} / {rea[f'{a}|{st}|{k}']:.1f}" for k in keys) + " |")
    L += ["", f"- FlyWire spikes of the source families without a BANC unit: maximum fraction {lost:.3f} over all runs.",
          f"- Fraction of all FlyWire DN spikes passed to the VNC (arm `ref`): {min(used):.4f}-{max(used):.4f} (only "
          "the three steering families are bridged).", ""]
    # power output and network
    rp_ = d["replay"]
    pw = {f"{a}|{st}|{t}|{s}": float(np.mean([r["rate_hz"][f"{t}|{s}"] for r in rp_ if r["arm"] == a and r["stimulus"] == st]))
          for a in C.ARMS for st in STIMS for t in POWER for s in ("left", "right")}
    out["power"] = pw
    L += ["Wing power output under the `HAND` baseline (arm `ref`, Hz, L / R): " + "; ".join(
        f"{t} " + ", ".join(f"{st} {pw[f'ref|{st}|{t}|left']:.1f} / {pw[f'ref|{st}|{t}|right']:.1f}" for st in STIMS)
        for t in POWER) + ".", ""]
    return L


def post_hoc(d, out):
    """Post-hoc (not registered): applied DN rates of the test arms against the F3a-3 lowest steering rate."""
    lo = json.loads(F3A3.read_text())["r3"]["lowest_rate"]
    app = out["drive"]["applied"]
    rows = {}
    L = ["## 8. Post-hoc (not registered in prereg/09)", "",
         "Highest applied input rate per family (over sides and stimuli, mean over seeds) against the lowest input rate "
         "at which the family steered in F3a-3 (Hz):", "",
         "| family | `ref` max applied | `rate_gain` max applied | F3a-3 lowest steering rate | stimulus / side of the `rate_gain` maximum |",
         "|---|---|---|---|---|"]
    for f in C.STEER_FAMILIES:
        mx = {}
        for a in ("ref", "rate_gain"):
            ks = [k for k in app if k.startswith(f"{a}|") and f"|{f}|" in k]
            k = max(ks, key=app.get)
            mx[a] = (app[k], k)
        rows[f] = dict(ref=mx["ref"][0], rate_gain=mx["rate_gain"][0], lowest=lo[f], where=mx["rate_gain"][1])
        _, st, _, sd = mx["rate_gain"][1].split("|")
        L.append(f"| {f} | {mx['ref'][0]:.1f} | {mx['rate_gain'][0]:.1f} | {lo[f]} | {st} / {sd} |")
    out["post_hoc"] = rows
    reach = [f for f, r in rows.items() if r["rate_gain"] >= r["lowest"]]
    L += ["", "- Families whose highest applied rate in `rate_gain` reaches their F3a-3 lowest steering rate: "
          + (", ".join(reach) if reach else "none") + ". In this open-loop visual protocol (no flight, no sugar input) the "
          "FlyWire steering DNs fire below their F0 flight rates. So the direction-selective brain signal (section 4) "
          "reaches the VNC at rates below those at which these families steered in F3a-3. A gain fitted to the flight "
          "rates does not bring it into range.", ""]
    return L


def summary(out):
    b, v, g = out["brain"], out["vnc"], out["gates"]
    L = ["## 7. Summary", "",
         f"- Gates: {'all pass' if g['passed'] else 'a gate FAILS (tests not used)'}; input check "
         f"{'passed' if out['input_check']['passed'] else 'FAILED'}.",
         "- Brain: direction-selective FlyWire steering families after Holm: "
         + (", ".join(b["selective"]) if b["selective"] else "none") + "."]
    for a, s in v["specificity"].items():
        L.append(f"- `{a}`: " + ("direction-selective and specific" if s["specific"] else
                                 "direction-selective but not specific" if s["selective"] else "not direction-selective") + ".")
    L += ["- Limits stated in prereg/09: open loop, no body, no VNC -> brain feedback; rate coding with Poisson "
          "re-synthesis; only DNa04 / DNa10 / DNa15 bridged; no degree-matched random VNC in this step.", ""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    md, js = OUT.with_suffix(".md"), OUT.with_suffix(".json")
    if (md.exists() or js.exists()) and not a.overwrite:
        raise SystemExit(f"{md} exists (use --overwrite)")
    d = json.loads(REC.read_text())
    chk = json.loads(CHK.read_text())
    out = {}
    L = ["# F3b-1: hybrid optomotor test, FlyWire brain -> BANC VNC (prereg/09)", "",
         "Generated by `scripts/f3b1_report.py` from `runs/f3b1/f3b1_v3.json` and `runs/f3b1/input_check.json`; checked "
         "by `scripts/verify_report.py`. Pre-registration: `prereg/09_f3b1_optomotor.md`.", ""]
    L += setup_section(d, out)
    L += input_section(chk, out)
    L += gates_section(d, out)
    L += brain_section(d, out)
    L += vnc_section(d, out)
    L += drive_section(d, out)
    L += summary(out)
    L += post_hoc(d, out)
    md.write_text("\n".join(L) + "\n")
    js.write_text(json.dumps(out, indent=1) + "\n")
    print("->", md, js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
