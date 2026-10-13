"""Render docs/results/f3a3_vnc_subgraph.{md,json} from runs/f3a3/f3a3_v3.json (prereg/08; numbers by script).
Whole-graph comparisons from runs/f3a/f3a_v3.json (F3a) and runs/f3a2/f3a2_v3.json (C2); FlyWire rates from
docs/results/f3a2_checks.json (C1).

    env -u PYTHONPATH python scripts/f3a3_report.py [--overwrite]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from f3a2_report import ALPHA, cp, holm, p_t, p_w  # noqa: E402

REC = ROOT / "runs" / "f3a3" / "f3a3_v3.json"
F3A_REC = ROOT / "runs" / "f3a" / "f3a_v3.json"
F3A2_REC = ROOT / "runs" / "f3a2" / "f3a2_v3.json"
C1 = ROOT / "docs" / "results" / "f3a2_checks.json"
SUB = ROOT / "docs" / "results" / "f3a3_subgraph.json"
OUT = ROOT / "docs" / "results" / "f3a3_vnc_subgraph"
POWER = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b", "tp2", "ps1")
FLOOR = 1.0          # evaluation floor of prereg/06 (Hz)
BLK = 4              # 100 ms = 4 steps of 25 ms


def mean_rate(runs, cond, key, field="condition"):
    return float(np.mean([x["rate_hz"][key] for x in runs if x[field] == cond]))


def subgraph_section(d, out):
    s = d["subgraph"]
    committed = json.loads(SUB.read_text())
    same = all(committed[k] == s[k] for k in s)
    out["subgraph"] = dict(s, matches_committed=same)
    a, dn, mo, rm = s["ascending"], s["descending"], s["motor"], s["removed"]
    return ["## 1. The subgraph (prereg/08 section 1; data only)", "",
            f"- Source `{d['graph_meta']['data_source']}`: {s['whole']['neurons']:,} neurons, {s['whole']['pairs']:,} "
            f"pairs, {s['whole']['synapses']:,} synapses.",
            f"- Kept: {s['kept']['neurons']:,} neurons, {s['kept']['pairs']:,} pairs, {s['kept']['synapses']:,} synapses. "
            f"By reason: region VNC/neck {s['kept_by_reason']['region']:,}, motor {s['kept_by_reason']['motor']:,}, "
            f"descending {s['kept_by_reason']['descending']:,}, ascending {s['kept_by_reason']['ascending']:,}. By region: "
            + ", ".join(f"{k} {v:,}" for k, v in s["kept_by_region"].items()) + ".",
            f"- Removed: {rm['neurons']:,} neurons (" + ", ".join(f"{k} {v:,}" for k, v in rm["by_region"].items() if v)
            + f"); edges onto DNs from kept neurons {rm['edges_onto_dn_from_kept']['pairs']:,} pairs / "
            f"{rm['edges_onto_dn_from_kept']['synapses']:,} synapses (DNs are input nodes).",
            f"- Ascending neurons (outputs, recorded): {a['neurons']:,} (" + ", ".join(f"{k} {v:,}" for k, v in a["by_region"].items() if v)
            + f"); inputs kept {a['inputs_kept']['synapses']:,} synapses, inputs from removed neurons removed "
            f"{a['inputs_removed']['synapses']:,}; outputs kept {a['outputs_kept']['synapses']:,}, outputs to removed "
            f"neurons {a['outputs_removed']['synapses']:,}.",
            f"- DNs (input nodes): {dn['neurons']:,} (" + ", ".join(f"{k} {v:,}" for k, v in dn["by_region"].items() if v)
            + f"); outputs kept {dn['outputs_kept']['synapses']:,} synapses, outputs to removed neurons "
            f"{dn['outputs_removed']['synapses']:,}.",
            f"- Motor neurons: {mo['neurons']:,} (" + ", ".join(f"{k} {v:,}" for k, v in mo["by_region"].items() if v)
            + f"); inputs kept {mo['inputs_kept']['synapses']:,} synapses, inputs from removed neurons "
            f"{mo['inputs_removed']['synapses']:,}.",
            f"- The run's subgraph equals the committed `docs/results/f3a3_subgraph.json`: {'yes' if same else 'NO'}. "
            f"Peak RAM {d['peak_rss_gb']:.2f} GB, wall {d['total_wall_s']:.0f} s.",
            "- DN groups (neurons): " + ", ".join(f"{k} {v['n']}" for k, v in d["dn_groups"].items()) + ".", ""]


def r1_section(d, out):
    f3a = json.loads(F3A_REC.read_text())
    runs, wr = d["r1"], f3a["runs"]
    drives = list(d["r1_drives"])
    keys = list(runs[0]["rate_hz"])
    tys = list(dict.fromkeys(k.split("|")[0] for k in keys))
    sub = {(c, k): mean_rate(runs, c, k) for c in drives for k in keys}
    whl = {(c, k): mean_rate(wr, c, k, "drive") for c in drives for k in keys if k in wr[0]["rate_hz"]}
    show = [t for t in tys if any(sub.get((c, f"{t}|{s}"), 0) > 0 or whl.get((c, f"{t}|{s}"), 0) > 0
                                  for c in drives for s in ("left", "right"))]
    out["r1"] = dict(sub={f"{c}|{k}": v for (c, k), v in sub.items()}, whole={f"{c}|{k}": v for (c, k), v in whl.items()})
    L = ["## 3. R1: recruitment on the subgraph against F3a on the whole graph (seeds 0-9)", "",
         "Per-neuron rate over the 1 s drive, mean over seeds, Hz, `subgraph L / R (whole L / R)`; readout types with any "
         "spike in either graph under any drive.", "",
         "| category | type | " + " | ".join(drives) + " |", "|---|---|" + "---|" * len(drives)]

    def cell(c, t):
        def v(m, s):
            x = m.get((c, f"{t}|{s}"))
            return "-" if x is None else f"{x:.1f}"
        return f"{v(sub, 'left')} / {v(sub, 'right')} ({v(whl, 'left')} / {v(whl, 'right')})"
    for t in show:
        L.append(f"| {d['types'][t]['category']} | {t} | " + " | ".join(cell(c, t) for c in drives) + " |")
    L += ["", f"Active (seed mean >= {FLOOR:.0f} Hz) type/side per drive:", "",
          "| drive | network sub / whole (Hz) | active in both | subgraph only | whole graph only |", "|---|---|---|---|---|"]
    out["r1"]["active"] = {}
    for c in drives:
        a_s = {k for k in keys if sub[(c, k)] >= FLOOR}
        a_w = {k for k in keys if whl.get((c, k), 0) >= FLOOR}
        out["r1"]["active"][c] = dict(both=sorted(a_s & a_w), sub_only=sorted(a_s - a_w), whole_only=sorted(a_w - a_s))
        ns = float(np.mean([x["network_rate_hz"] for x in runs if x["condition"] == c]))
        nw = float(np.mean([x["network_rate_hz"] for x in wr if x["drive"] == c]))
        L.append(f"| {c} | {ns:.4f} / {nw:.4f} | {len(a_s & a_w)} | {', '.join(sorted(a_s - a_w)) or '-'} | "
                 f"{', '.join(sorted(a_w - a_s)) or '-'} |")
    L.append("")
    return L


def decay(net, n_on):
    """(A, B): mean network rate in the last 100 ms of the cut and of the drive."""
    return float(np.mean(net[-BLK:])), float(np.mean(net[n_on - BLK:n_on]))


def r2_section(d, out):
    r2 = d["r2"]
    runs = r2["runs"]
    n_on, n_cut, st = d["n_on"], d["n_cut_r2"], d["step_ms"]
    net = np.mean([r["network_step_hz"] for r in runs], axis=0)
    ndn = np.mean([r["nondriven_step_hz"] for r in runs], axis=0)
    keys = list(runs[0]["mn_step_hz"])
    mn = {k: np.mean([r["mn_step_hz"][k] for r in runs], axis=0) for k in keys}
    act = [k for k in keys if mn[k][n_on:].max() > 0]
    nb = (n_on + n_cut) // BLK
    ab = [decay(r["network_step_hz"], n_on) for r in runs]
    abn = [decay(r["nondriven_step_hz"], n_on) for r in runs]
    dec = [a < 0.1 * b for a, b in ab]
    whole = json.loads(F3A2_REC.read_text())["c2"]["runs"]
    abw = [decay(r["network_step_hz"], n_on) for r in whole]
    ok = sum(dec) >= 9
    out["r2"] = dict(A=[a for a, _ in ab], B=[b for _, b in ab], decays=dec, n_decay=sum(dec), passed=ok,
                     A_nondriven=[a for a, _ in abn], B_nondriven=[b for _, b in abn],
                     whole_A=[a for a, _ in abw], whole_B=[b for _, b in abw],
                     network_block_hz=[float(net[b * BLK:(b + 1) * BLK].mean()) for b in range(nb)])
    L = ["## 4. R2: persistence on the subgraph (DNg02 both sides 100 Hz for 1 s, then a 1 s cut; seeds 0-9)", "",
         "### 4.1 Criterion (prereg/08 section 4)", "",
         "A = network rate in the last 100 ms of the cut, B = in the last 100 ms of the drive (all subgraph neurons). A seed "
         "decays if A < 0.1 B; R2 passes if >= 9 of 10 seeds decay. Whole graph = prereg/07 C2, same seeds (descriptive).", "",
         "| seed | A (Hz) | B (Hz) | A / B | decays | non-driven A / B | whole graph A / B |", "|---|---|---|---|---|---|---|"]
    for r, (a, b), (an, bn), (aw, bw), ok_s in zip(runs, ab, abn, abw, dec):
        L.append(f"| {r['seed']} | {a:.4f} | {b:.4f} | {a / b:.3f} | {'yes' if ok_s else 'no'} | {an / bn:.3f} | "
                 f"{aw / bw:.3f} |")
    L += ["", f"**R2: {sum(dec)} / {len(dec)} seeds decay; {'pass' if ok else 'FAIL'}.**", "",
          "### 4.2 Time course", "",
          f"Network rate, non-driven rate and every MN type/side with any spike after the cut, per 100 ms block (mean "
          f"over seeds). Drive = blocks 1-{n_on // BLK}, cut = blocks {n_on // BLK + 1}-{nb}.", "",
          "| block (ms) | network (Hz) | non-driven (Hz) | " + " | ".join(act) + " |", "|---|---|---|" + "---|" * len(act)]
    for b in range(nb):
        sl = slice(b * BLK, (b + 1) * BLK)
        L.append(f"| {b * BLK * st:.0f}-{(b + 1) * BLK * st:.0f} | {net[sl].mean():.4f} | {ndn[sl].mean():.4f} | "
                 + " | ".join(f"{mn[k][sl].mean():.1f}" for k in act) + " |")
    last = [r["last_active_step"] for r in runs]
    after = [(s - n_on + 1) * st if s is not None and s >= n_on else 0.0 for s in last]
    car = r2["carriers"]
    out["r2"].update(active_ms_after_cut=after, cut_spikes=[r["cut_spikes"] for r in runs],
                     n_cut_neurons=[r["n_cut_neurons"] for r in runs], n_carriers=len(car),
                     n_carriers_all_seeds=sum(c["n_seeds"] == len(runs) for c in car))
    L += ["", "- Activity after the cut lasts (last step with any spike, ms after the cut), per seed: "
          + ", ".join(f"{v:.0f}" for v in after) + ".",
          f"- Cut spikes per seed: {', '.join(str(v) for v in out['r2']['cut_spikes'])}; neurons spiking during the cut "
          f"per seed: {', '.join(str(v) for v in out['r2']['n_cut_neurons'])}.",
          f"- Carriers (>= 1 spike during the cut in any seed): {len(car)} neurons; in all {len(runs)} seeds: "
          f"{out['r2']['n_carriers_all_seeds']}.", ""]
    for col in ("region", "super_class", "cell_class"):
        cnt, sp = Counter(c[col] for c in car), Counter()
        for c in car:
            sp[c[col]] += c["cut_spikes"]
        out["r2"][f"by_{col}"] = {k: dict(neurons=v, spikes=sp[k]) for k, v in cnt.items()}
        if not car:
            continue
        L += [f"Carriers by `{col}`:", "", f"| {col} | neurons | cut spikes (all seeds) |", "|---|---|---|"]
        for k, v in sorted(cnt.items(), key=lambda x: -sp[x[0]]):
            L.append(f"| {k} | {v} | {sp[k]} |")
        L.append("")
    if car:
        L += ["Top 30 carriers by cut spikes (all seeds):", "",
              "| root id | super_class | cell_class | cell_type | side | region | seeds | cut spikes |",
              "|---|---|---|---|---|---|---|---|"]
        for c in car[:30]:
            L.append(f"| {c['root_id']} | {c['super_class']} | {c['cell_class']} | {c['cell_type']} | {c['side']} | "
                     f"{c['region']} | {c['n_seeds']} | {c['cut_spikes']} |")
        L.append("")
    out["r2"]["top_carriers"] = car[:30]
    return L


def r3_section(d, out):
    runs = d["r3"]
    sym, symw = d["symmetry"], d["symmetry_whole"]
    steer = d["steer_mn"]
    passed = [m for m in steer if sym.get(m, {}).get("passed")]
    failed = [m for m in steer if m not in passed]
    fams, rates, seeds = d["steer_families"], d["steer_rates"], d["seeds"]["run"]
    by = {(x["condition"], x["seed"]): x for x in runs}

    def D(c, m):
        return np.array([by[(c, s)]["rate_hz"][f"{m}|right"] - by[(c, s)]["rate_hz"][f"{m}|left"] for s in seeds])

    tests, res = [], {}
    for r in rates:
        tr = []
        for f in fams:
            for m in passed:
                for s in ("L", "R"):
                    dd = D(f"{f}_{s}_{r}", m) - D("B", m)
                    tr.append(dict(rate=r, family=f, mn=m, side=s, d=dd.tolist(), mean=float(dd.mean()),
                                   p_t=p_t(dd), p_w=p_w(dd)))
        for t, a, b in zip(tr, holm([t["p_t"] for t in tr]), holm([t["p_w"] for t in tr])):
            t["p_t_holm"], t["p_w_holm"] = float(a), float(b)
        tests += tr
    idx = {(t["rate"], t["family"], t["mn"], t["side"]): t for t in tests}
    for f in fams:
        res[f] = {}
        for r in rates:
            hits, hits_w = [], []
            for m in passed:
                a, b = idx[(r, f, m, "L")], idx[(r, f, m, "R")]
                for key, lst in (("p_t_holm", hits), ("p_w_holm", hits_w)):
                    if a[key] < ALPHA and b[key] < ALPHA and np.sign(a["mean"]) * np.sign(b["mean"]) < 0:
                        lst.append(dict(mn=m, sign="ipsilateral" if a["mean"] < 0 else "contralateral"))
            own = [float(np.mean([by[(f"{f}_{s}_{r}", x)]["group_rate_hz"][f"{f}_{s}"] for x in seeds])) for s in ("L", "R")]
            res[f][str(r)] = dict(steers=bool(hits), mn=hits, steers_wilcoxon=bool(hits_w), mn_wilcoxon=hits_w,
                                  own_rate_hz=own)
    c1 = json.loads(C1.read_text())["c1"]["f0"]
    lowest = {f: next((r for r in rates if res[f][str(r)]["steers"]), None) for f in fams}
    n_per_rate = len(fams) * 2 * len(passed)
    out["r3"] = dict(gate_passing=passed, gate_failing=failed, n_tests_per_rate=n_per_rate, tests=tests, families=res,
                     lowest_rate=lowest)
    L = ["## 5. R3: steering dose-response on the subgraph (baseline DNg02 both sides 100 Hz, label HAND; seeds 0-9)", "",
         "### 5.1 Symmetry gate of the steering MN types (input synapses, |log2(R/L)| < 0.5; subgraph decides)", "",
         "| MN type | n L/R | subgraph log2 total / exc | pass | whole graph log2 total / exc | pass |", "|---|---|---|---|---|---|"]
    for m in steer:
        a, b = sym.get(m), symw.get(m)
        if a is None:
            L.append(f"| {m} | - | - | not in graph | - | - |")
            continue
        L.append(f"| {m} | {a['n_left']}/{a['n_right']} | {a['log2_total']:+.2f} / {a['log2_exc']:+.2f} | "
                 f"{'pass' if a['passed'] else 'FAIL'} | {b['log2_total']:+.2f} / {b['log2_exc']:+.2f} | "
                 f"{'pass' if b['passed'] else 'FAIL'} |")
    L += ["", "### 5.2 Decision per family and rate (prereg/08 section 5)", "",
          f"Gate-passing types used ({len(passed)}): {', '.join(passed)}. Failing, reported separately ({len(failed)}): "
          f"{', '.join(failed) if failed else 'none'}. Holm per rate over {n_per_rate} tests ({len(fams)} families x 2 "
          f"sides x {len(passed)} types). Primary: paired t-test; sensitivity: Wilcoxon signed-rank (smallest exact p "
          "0.00195 at 10 seeds, so it cannot pass Holm at this test count; uncorrected p in 5.4). Own rate = realised "
          "rate of the driven DN (L drive / R drive), mean over seeds.", "",
          "| family | rate (Hz) | steers (t, Holm) | MN types and sign | steers (Wilcoxon, Holm) | own rate L / R (Hz) |",
          "|---|---|---|---|---|---|"]
    for f in fams:
        for r in rates:
            x = res[f][str(r)]
            L.append(f"| {f} | {r} | {'yes' if x['steers'] else 'no'} | "
                     f"{', '.join(h['mn'] + ' ' + h['sign'] for h in x['mn']) if x['mn'] else '-'} | "
                     f"{'yes' if x['steers_wilcoxon'] else 'no'} | {x['own_rate_hz'][0]:.1f} / {x['own_rate_hz'][1]:.1f} |")
    L += ["", "### 5.3 Lowest steering rate against the FlyWire rate (C1, F0 flight window, per neuron, seed mean)", "",
          "| family | lowest steering input rate (Hz) | rates at which it steers | own rate there L / R (Hz) | "
          "FlyWire flight L / R (Hz) |", "|---|---|---|---|---|"]
    out["r3"]["flywire_flight"] = {}
    for f in fams:
        fw = c1[f]["flight"]
        out["r3"]["flywire_flight"][f] = [fw["rates_left"], fw["rates_right"]]
        lo = lowest[f]
        ownc = (f"{res[f][str(lo)]['own_rate_hz'][0]:.1f} / {res[f][str(lo)]['own_rate_hz'][1]:.1f}" if lo else "-")
        L.append(f"| {f} | {lo if lo else 'none'} | {', '.join(str(r) for r in rates if res[f][str(r)]['steers']) or 'none'} | "
                 f"{ownc} | {', '.join(f'{v:.1f}' for v in fw['rates_left'])} / "
                 f"{', '.join(f'{v:.1f}' for v in fw['rates_right'])} |")
    L += ["", "### 5.4 Tests with a non-zero mean change (d = (R - L) under steering minus (R - L) under baseline, Hz; "
          "rows where both sides' d are identically 0 are omitted)", "",
          "| family | rate | MN type | left drive: mean d | p (t) | p Holm (t) | p (Wilcoxon) | right drive: mean d | "
          "p (t) | p Holm (t) | p (Wilcoxon) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_zero = 0
    for f in fams:
        for r in rates:
            for m in passed:
                a, b = idx[(r, f, m, "L")], idx[(r, f, m, "R")]
                if not any(a["d"]) and not any(b["d"]):
                    n_zero += 1
                    continue
                L.append(f"| {f} | {r} | {m} | {a['mean']:+.2f} | {a['p_t']:.2g} | {a['p_t_holm']:.2g} | {a['p_w']:.2g} | "
                         f"{b['mean']:+.2f} | {b['p_t']:.2g} | {b['p_t_holm']:.2g} | {b['p_w']:.2g} |")
    out["r3"]["n_rows_all_zero"] = n_zero
    L += ["", f"Family x rate x MN-type rows with d identically 0 under both drive sides (omitted): {n_zero} of "
          f"{len(fams) * len(rates) * len(passed)}.", "",
          "### 5.5 Gate-failing steering MN types (descriptive, not used)", ""]
    out["r3"]["gate_failing_d"] = {}
    if failed:
        L += ["| family | rate | MN type | left drive: mean d (Hz) | right drive: mean d (Hz) |", "|---|---|---|---|---|"]
        for f in fams:
            for r in rates:
                for m in failed:
                    if f"{m}|left" not in runs[0]["rate_hz"] or f"{m}|right" not in runs[0]["rate_hz"]:
                        continue
                    a, b = (float((D(f"{f}_{s}_{r}", m) - D("B", m)).mean()) for s in ("L", "R"))
                    out["r3"]["gate_failing_d"][f"{f}|{r}|{m}"] = [a, b]
                    L.append(f"| {f} | {r} | {m} | {a:+.2f} | {b:+.2f} |")
    else:
        L.append("None.")
    # baseline power and network
    r1 = out["r1"]["sub"]
    L += ["", "### 5.6 Baseline power output (`B`) against R1 G100 and F3a G100 (Hz, L / R)", "",
          "| MN type | R3 baseline B | R1 G100 (subgraph) | F3a G100 (whole) |", "|---|---|---|---|"]
    out["r3"]["baseline_power"] = {}
    for m in POWER:
        bl, br = mean_rate(runs, "B", f"{m}|left"), mean_rate(runs, "B", f"{m}|right")
        out["r3"]["baseline_power"][m] = [bl, br]
        L.append(f"| {m} | {bl:.1f} / {br:.1f} | {r1['G100|' + m + '|left']:.1f} / {r1['G100|' + m + '|right']:.1f} | "
                 f"{out['r1']['whole']['G100|' + m + '|left']:.1f} / {out['r1']['whole']['G100|' + m + '|right']:.1f} |")
    L += ["", "### 5.7 Network rate per condition (mean over seeds)", "",
          "| condition | network (Hz) | non-driven (Hz) | DNg02 L / R (Hz) |", "|---|---|---|---|"]
    out["r3"]["network"] = {}
    for c in d["r3_conditions"]:
        r = [x for x in runs if x["condition"] == c]
        net = float(np.mean([x["network_rate_hz"] for x in r]))
        out["r3"]["network"][c] = net
        L.append(f"| {c} | {net:.4f} | {np.mean([x['nondriven_rate_hz'] for x in r]):.4f} | "
                 f"{np.mean([x['group_rate_hz']['DNg02_L'] for x in r]):.1f} / "
                 f"{np.mean([x['group_rate_hz']['DNg02_R'] for x in r]):.1f} |")
    L.append("")
    return L


def post_hoc(out):
    """Observations not registered in prereg/08 (labelled post-hoc; not used for any decision)."""
    g = out["r1"]["active"]["G100"]
    wo = g["whole_only"]
    cats = Counter(k.split("|")[0] for k in wo)
    c3w = json.loads(C1.read_text())["c3"]["families"]
    r3 = out["r3"]
    cmp_ = {f: dict(whole=[h["mn"] for h in c3w[f]["mn"]], sub=[h["mn"] for h in r3["families"][f]["100"]["mn"]])
            for f in r3["families"]}
    above = {}
    for f, lo in r3["lowest_rate"].items():
        if lo is None:
            continue
        own = r3["families"][f][str(lo)]["own_rate_hz"]
        fw = r3["flywire_flight"][f]
        above[f] = dict(own=own, flywire_max=max(fw[0] + fw[1]), above=min(own) > max(fw[0] + fw[1]))
    out["post_hoc"] = dict(g100_whole_only=wo, c3_100hz_whole_vs_sub=cmp_, lowest_vs_flywire=above)
    return ["## 7. Post-hoc observations (not registered in prereg/08; not used for any decision)", "",
            f"- **What the BANC brain added to F3a at G100.** {len(wo)} type/sides are active (>= 1 Hz) on the whole graph "
            f"but not on the subgraph: " + ", ".join(f"{t}" for t in sorted(cats)) + ". This includes the right-only "
            "b3, hg1 and hg4 of F3a (expectation (b)) and the neck MNs. They depend on what the subgraph removes (BANC brain/SEZ neurons, or the synapses onto DNs); "
            "this run does not separate the two. The wing power output (DLM, DVM1, tp2, ps1) and the right-only DVM3a-b "
            "are within about 1-5 Hz on both graphs (R1 table).",
            "- **R3 at 100 Hz against prereg/07 C3 (whole graph), gate-passing MN types that steer:** "
            + "; ".join(f"{f} whole {', '.join(v['whole']) or '-'}, subgraph {', '.join(v['sub']) or '-'}"
                        for f, v in cmp_.items()) + ".",
            "- **Lowest steering rate against FlyWire.** Realised own rate of the driven DN at the lowest steering rate "
            "against the largest FlyWire F0 flight-window rate of the family's neurons: "
            + "; ".join(f"{f} {v['own'][0]:.1f} / {v['own'][1]:.1f} Hz vs {v['flywire_max']:.1f} Hz "
                        f"({'above' if v['above'] else 'overlapping'})" for f, v in above.items())
            + ".", ""]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if (OUT.with_suffix(".md").exists() or OUT.with_suffix(".json").exists()) and "--overwrite" not in argv:
        raise SystemExit("output exists; use --overwrite")
    d = json.loads(REC.read_text())
    out = {}
    g1 = sum(r["spikes"] for r in d["gate1"])
    g2 = d["gate2_runs"]
    ign = sum(r["ignited"] for r in g2)
    lo, hi = cp(ign, len(g2))
    nets = {f"R1 {c}": float(np.mean([x["network_rate_hz"] for x in d["r1"] if x["condition"] == c])) for c in d["r1_drives"]}
    nets["R2"] = float(np.mean([r["network_rate_hz"] for r in d["r2"]["runs"]]))
    for c in d["r3_conditions"]:
        nets[f"R3 {c}"] = float(np.mean([x["network_rate_hz"] for x in d["r3"] if x["condition"] == c]))
    nets["gate2"] = float(np.mean([r["network_rate_hz"] for r in g2]))
    mx = max(nets, key=nets.get)
    gates = dict(gate1_spikes=g1, gate1_pass=g1 == 0, gate2_ignited=ign, gate2_n=len(g2), gate2_ci=[lo, hi],
                 gate2_pass=ign == 0, gate2_mean_last_cut_step_hz=float(np.mean([r["last_cut_step_hz"] for r in g2])),
                 gate3_max_network_hz=nets[mx], gate3_max_condition=mx, gate3_pass=nets[mx] < 5.0)
    out["gates"] = gates
    ok = gates["gate1_pass"] and gates["gate2_pass"] and gates["gate3_pass"]
    L = ["# F3a-3: BANC VNC subgraph, persistence and steering dose-response (prereg/08_f3a3_vnc_subgraph.md)", "",
         "Generated by `scripts/f3a3_subgraph.py`, `scripts/f3a3_vnc_run.py` and `scripts/f3a3_report.py`; record "
         "`runs/f3a3/f3a3_v3.json` (not committed); whole-graph comparisons from `runs/f3a/f3a_v3.json` (F3a) and "
         "`runs/f3a2/f3a2_v3.json` (prereg/07 C2). Open loop on the BANC VNC subgraph only (no brain, no body, no "
         "hybrid). Criteria, seeds and windows are those of the committed pre-registration.", ""]
    L += subgraph_section(d, out)
    L += ["## 2. Gates", "", "| gate | result | pass |", "|---|---|---|",
          f"| 1: no input, seeds {d['seeds']['gate1'][0]}-{d['seeds']['gate1'][-1]} (spikes) | {g1} | "
          f"{'pass' if gates['gate1_pass'] else 'FAIL'} |",
          f"| 2: ignition after the 200 ms cut, {len(d['dn_groups'])} groups at 100 Hz, {len(g2)} seeds | {ign} / {len(g2)} "
          f"(95 % CI {lo:.3f}-{hi:.3f}); mean last-cut-step rate {gates['gate2_mean_last_cut_step_hz']:.4f} Hz | "
          f"{'pass' if gates['gate2_pass'] else 'FAIL'} |",
          f"| 3': network rate under drive < 5 Hz in every condition | max {nets[mx]:.4f} Hz ({mx}) | "
          f"{'pass' if gates['gate3_pass'] else 'FAIL'} |", "",
          ("All gates pass; R1-R3 are usable for the interface pre-registration." if ok else
           "**A gate failed: per prereg/08 section 7, R1-R3 are reported but not used for the interface.**"), ""]
    L += r1_section(d, out)
    L += r2_section(d, out)
    L += r3_section(d, out)
    st = {f: v for f, v in out["r3"]["lowest_rate"].items()}
    L += ["## 6. Summary (prereg/08 section 7)", "",
          f"- Gates: {'all pass' if ok else 'a gate failed'}.",
          f"- Subgraph: {out['subgraph']['kept']['neurons']:,} neurons, {out['subgraph']['kept']['pairs']:,} pairs, "
          f"{out['subgraph']['kept']['synapses']:,} synapses.",
          f"- R2 (decay A < 0.1 B in >= 9 / 10 seeds): {out['r2']['n_decay']} / {len(out['r2']['decays'])}, "
          f"**{'pass' if out['r2']['passed'] else 'FAIL'}**.",
          "- R3, lowest steering input rate: " + ", ".join(f"{f} {v if v else 'none'}" + (" Hz" if v else "")
                                                          for f, v in st.items()) + ".",
          "- What follows for F3b is the project owner's decision (prereg/08 section 7).", ""]
    L += post_hoc(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1))
    OUT.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
