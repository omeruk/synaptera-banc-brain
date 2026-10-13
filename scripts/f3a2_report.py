"""Render docs/results/f3a2_checks.{md,json} from runs/f3a2/{c1_rates,f3a2_v3}.json (prereg/07; numbers by script).

    env -u PYTHONPATH python scripts/f3a2_report.py [--overwrite]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "runs" / "f3a2"
OUT = ROOT / "docs" / "results" / "f3a2_checks"
F3A = ROOT / "docs" / "results" / "f3a_banc_vnc.json"
ALPHA = 0.05
BINS = ((0, 20, "< 20"), (20, 50, "20-50"), (50, 100, "50-100"), (100, np.inf, ">= 100"))
POWER = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b", "tp2", "ps1")


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def p_t(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    if np.std(d) == 0:
        return 0.0
    return float(ttest_1samp(d, 0.0).pvalue)


def p_w(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, k in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[k]))
        adj[k] = run
    return adj


def bins(x):
    x = np.asarray(x, float)
    return [int(((x >= lo) & (x < hi)).sum()) for lo, hi, _ in BINS]


def c1_section(c1, out):
    N = c1["neurons"]
    fams = c1["families"]
    L = ["## 3. C1: FlyWire per-neuron rates of the seven families (existing records, no simulation)", "",
         "Sources: F0 records `runs/f0/f0_5ms_seed{0..4}_data.h5` (coupling 5 ms, latency 1; all spikes), windows "
         "perch (calibration, wings off), flight (takeoff, cruise, approach, descend), landing (touchdown, landed); and "
         "the v1 visual DN screen records (seeds 601-605, 10 conditions, 200-1000 ms after onset; v1 engine and input "
         "path, reported as v1 records). Rate = mean per neuron over the window; seed mean shown. Bins = BANC "
         "recruitment range from F3a (no wing MN at 20 Hz DNg02 drive; DLM/DVM/tp2 recruited at 50-100 Hz).", "",
         "### 3.1 F0 (per neuron, mean over 5 seeds; L / R listed per neuron)", "",
         "| family | window | n L/R | rates L (Hz) | rates R (Hz) | median (Hz) | max 25 ms bin (Hz) | "
         + " | ".join(f"{b[2]} Hz" for b in BINS) + " |", "|---|---|---|---|---|---|---|" + "---|" * len(BINS)]
    out["c1"] = dict(f0={}, vd={})
    seeds = sorted(c1["f0"])
    for f in fams:
        ks = [k for k, x in enumerate(N) if x["family"] == f]
        kl = [k for k in ks if N[k]["side"] == "left"]
        kr = [k for k in ks if N[k]["side"] == "right"]
        for w in ("perch", "flight", "landing"):
            r = np.mean([c1["f0"][s][w]["rate_hz"] for s in seeds], axis=0)
            pk = np.max([c1["f0"][s][w]["peak25_hz"] for s in seeds], axis=0)
            b = bins(r[ks])
            med = float(np.median(r[ks]))
            out["c1"]["f0"].setdefault(f, {})[w] = dict(rates_left=r[kl].tolist(), rates_right=r[kr].tolist(), median=med,
                                                        max=float(r[ks].max()), peak25=float(pk[ks].max()), bins=b)
            L.append(f"| {f} | {w} | {len(kl)}/{len(kr)} | {', '.join(f'{v:.1f}' for v in r[kl])} | "
                     f"{', '.join(f'{v:.1f}' for v in r[kr])} | {med:.1f} | {pk[ks].max():.0f} | "
                     + " | ".join(str(v) for v in b) + " |")
    # DNg02 statement
    g = out["c1"]["f0"]["DNg02"]
    ok_f0 = all(g[w]["median"] < 1.0 and g[w]["max"] < 20.0 for w in g)
    vs = sorted(c1["vd"])
    conds = list(c1["vd"][vs[0]])
    kg = [k for k, x in enumerate(N) if x["family"] == "DNg02"]
    vd_g = {c: np.mean([c1["vd"][s][c]["rate_hz"] for s in vs], axis=0)[kg] for c in conds}
    ok_vd = all(float(np.median(v)) < 1.0 and float(v.max()) < 20.0 for v in vd_g.values())
    out["c1"]["dng02_nearly_silent"] = dict(f0=ok_f0, v1_screen=ok_vd)
    L += ["", "### 3.2 v1 visual DN screen (per neuron, mean over 5 seeds; range over the 10 conditions)", "",
          "| family | n L/R | per-neuron rate, min-max over conditions (Hz), L then R | median over neurons x conditions (Hz) "
          "| max 25 ms bin (Hz) | " + " | ".join(f"{b[2]} Hz" for b in BINS) + " (neuron x condition) |",
          "|---|---|---|---|---|" + "---|" * len(BINS)]
    for f in fams:
        ks = [k for k, x in enumerate(N) if x["family"] == f]
        kl = [k for k in ks if N[k]["side"] == "left"]
        kr = [k for k in ks if N[k]["side"] == "right"]
        M = np.array([np.mean([c1["vd"][s][c]["rate_hz"] for s in vs], axis=0) for c in conds])   # cond x neuron
        P = np.max([[c1["vd"][s][c]["peak25_hz"] for c in conds] for s in vs], axis=(0, 1))
        cells = [f"{M[:, k].min():.1f}-{M[:, k].max():.1f}" for k in kl + kr]
        b = bins(M[:, ks].ravel())
        med = float(np.median(M[:, ks]))
        out["c1"]["vd"][f] = dict(per_neuron_min=[float(M[:, k].min()) for k in kl + kr],
                                  per_neuron_max=[float(M[:, k].max()) for k in kl + kr], median=med,
                                  peak25=float(P[ks].max()), bins=b)
        L.append(f"| {f} | {len(kl)}/{len(kr)} | {', '.join(cells)} | {med:.1f} | {P[ks].max():.0f} | "
                 + " | ".join(str(v) for v in b) + " |")
    L += ["", "### 3.3 DNg02 \"nearly silent\" (docs/02 §1.5; prereg/07 section 1)", "",
          f"- F0 (median < 1 Hz in every window and no DNg02 neuron >= 20 Hz): **{'confirmed' if ok_f0 else 'refuted'}** "
          f"(medians perch {g['perch']['median']:.1f}, flight {g['flight']['median']:.1f}, landing "
          f"{g['landing']['median']:.1f} Hz; maxima {g['perch']['max']:.1f}, {g['flight']['max']:.1f}, "
          f"{g['landing']['max']:.1f} Hz; largest 25 ms bin {max(g[w]['peak25'] for w in g):.0f} Hz).",
          f"- v1 visual screen (same statement over the 10 conditions): **{'confirmed' if ok_vd else 'refuted'}** "
          f"(largest condition median {max(float(np.median(v)) for v in vd_g.values()):.1f} Hz, largest neuron "
          f"{max(float(v.max()) for v in vd_g.values()):.1f} Hz).", ""]
    return L


def c2_section(d, out):
    c2 = d["c2"]
    runs = c2["runs"]
    n_on, n_cut, st = d["n_on"], d["n_cut_c2"], d["step_ms"]
    net = np.mean([r["network_step_hz"] for r in runs], axis=0)
    keys = list(runs[0]["mn_step_hz"])
    mn = {k: np.mean([r["mn_step_hz"][k] for r in runs], axis=0) for k in keys}
    act = [k for k in keys if mn[k][n_on:].max() > 0]
    blk = 4
    nb = (n_on + n_cut) // blk
    L = ["## 4. C2: persistence after the cut (DNg02 both sides 100 Hz for 1 s, then a 1 s cut; seeds 0-9)", "",
         f"Network rate and every MN type/side with any spike after the cut, per 100 ms block (mean of {blk} steps of "
         f"{st:.0f} ms, mean over seeds). Drive = blocks 1-{n_on // blk}, cut = blocks {n_on // blk + 1}-{nb}.", "",
         "| block (ms) | network (Hz) | " + " | ".join(act) + " |", "|---|---|" + "---|" * len(act)]
    for b in range(nb):
        sl = slice(b * blk, (b + 1) * blk)
        L.append(f"| {b * blk * st:.0f}-{(b + 1) * blk * st:.0f} | {net[sl].mean():.4f} | "
                 + " | ".join(f"{mn[k][sl].mean():.1f}" for k in act) + " |")
    last = [r["last_active_step"] for r in runs]
    after = [(s - n_on + 1) * st if s is not None and s >= n_on else 0.0 for s in last]
    car = c2["carriers"]
    n_seeds = len(runs)
    out["c2"] = dict(network_block_hz=[float(net[b * blk:(b + 1) * blk].mean()) for b in range(nb)],
                     last_cut_step_hz=[r["network_step_hz"][-1] for r in runs], active_ms_after_cut=after,
                     cut_spikes=[r["cut_spikes"] for r in runs], n_cut_neurons=[r["n_cut_neurons"] for r in runs],
                     n_carriers=len(car), n_carriers_all_seeds=sum(c["n_seeds"] == n_seeds for c in car))
    L += ["", f"- Network rate in the last cut step (1 s after the cut), per seed: "
          + ", ".join(f"{v:.4f}" for v in out["c2"]["last_cut_step_hz"]) + " Hz.",
          f"- Activity after the cut lasts (last step with any spike, ms after the cut), per seed: "
          + ", ".join(f"{v:.0f}" for v in after) + ".",
          f"- Cut spikes per seed: {', '.join(str(v) for v in out['c2']['cut_spikes'])}; neurons spiking during the cut "
          f"per seed: {', '.join(str(v) for v in out['c2']['n_cut_neurons'])}.",
          f"- Carriers (>= 1 spike during the cut in any seed): {len(car)} neurons; in all {n_seeds} seeds: "
          f"{out['c2']['n_carriers_all_seeds']}.", ""]
    for col in ("super_class", "cell_class"):
        cnt = Counter(c[col] for c in car)
        sp = Counter()
        for c in car:
            sp[c[col]] += c["cut_spikes"]
        out["c2"][f"by_{col}"] = {k: dict(neurons=v, spikes=sp[k]) for k, v in cnt.items()}
        L += [f"Carriers by `{col}`:", "", f"| {col} | neurons | cut spikes (all seeds) |", "|---|---|---|"]
        for k, v in sorted(cnt.items(), key=lambda x: -sp[x[0]]):
            L.append(f"| {k} | {v} | {sp[k]} |")
        L.append("")
    L += ["Top 30 carriers by cut spikes (all seeds):", "",
          "| root id | super_class | cell_class | cell_type | side | seeds | cut spikes |", "|---|---|---|---|---|---|---|"]
    for c in car[:30]:
        L.append(f"| {c['root_id']} | {c['super_class']} | {c['cell_class']} | {c['cell_type']} | {c['side']} | "
                 f"{c['n_seeds']} | {c['cut_spikes']} |")
    out["c2"]["top_carriers"] = car[:30]
    L.append("")
    return L


def c3_section(d, out):
    runs = d["c3"]
    sym = d["symmetry"]
    steer = d["steer_mn"]
    passed = [m for m in steer if sym.get(m, {}).get("passed")]
    failed = [m for m in steer if m not in passed]
    fams = sorted({c.rsplit("_", 1)[0] for c in d["c3_conditions"] if c != "B"},
                  key=lambda f: [c.rsplit("_", 1)[0] for c in d["c3_conditions"]].index(f))
    seeds = d["seeds"]["run"]

    def D(cond, m):
        r = {x["seed"]: x for x in runs if x["condition"] == cond}
        return np.array([r[s]["rate_hz"][f"{m}|right"] - r[s]["rate_hz"][f"{m}|left"] for s in seeds])

    def mean_rate(cond, key):
        return float(np.mean([x["rate_hz"][key] for x in runs if x["condition"] == cond]))

    tests = []
    for f in fams:
        for m in passed:
            for s in ("L", "R"):
                dd = D(f"{f}_{s}", m) - D("B", m)
                tests.append(dict(family=f, mn=m, side=s, d=dd.tolist(), mean=float(dd.mean()), p_t=p_t(dd), p_w=p_w(dd)))
    pt = holm([t["p_t"] for t in tests])
    pw = holm([t["p_w"] for t in tests])
    for t, a, b in zip(tests, pt, pw):
        t["p_t_holm"], t["p_w_holm"] = float(a), float(b)
    idx = {(t["family"], t["mn"], t["side"]): t for t in tests}
    res = {}
    for f in fams:
        hits, hits_w = [], []
        for m in passed:
            a, b = idx[(f, m, "L")], idx[(f, m, "R")]
            for key, lst in (("p_t_holm", hits), ("p_w_holm", hits_w)):
                if a[key] < ALPHA and b[key] < ALPHA and np.sign(a["mean"]) * np.sign(b["mean"]) < 0:
                    lst.append(dict(mn=m, sign="ipsilateral" if a["mean"] < 0 else "contralateral"))
        res[f] = dict(steers=bool(hits), mn=hits, steers_wilcoxon=bool(hits_w), mn_wilcoxon=hits_w)
    out["c3"] = dict(gate_passing=passed, gate_failing=failed, n_tests=len(tests), tests=tests, families=res)
    # baseline power vs F3a G100
    f3a = json.loads(F3A.read_text())
    L = ["## 5. C3: steering on top of power (baseline DNg02 both sides 100 Hz, label HAND; seeds 0-9)", "",
         "### 5.1 Baseline power output (`B`) against F3a G100 (Hz, L / R)", "",
         "| MN type | C3 baseline B | F3a G100 |", "|---|---|---|"]
    out["c3"]["baseline_power"] = {}
    for m in POWER:
        bl, br = mean_rate("B", f"{m}|left"), mean_rate("B", f"{m}|right")
        g = f3a["rates"][m]["by_drive"]["G100"]
        out["c3"]["baseline_power"][m] = [bl, br]
        L.append(f"| {m} | {bl:.1f} / {br:.1f} | {g['left'][0]:.1f} / {g['right'][0]:.1f} |")
    L += ["", "### 5.2 Network rate and driven-neuron rate per condition (mean over seeds)", "",
          "| condition | network (Hz) | non-driven (Hz) | DNg02 L / R (Hz) | steering DN (Hz) |", "|---|---|---|---|---|"]
    out["c3"]["network"] = {}
    for c in d["c3_conditions"]:
        r = [x for x in runs if x["condition"] == c]
        net = float(np.mean([x["network_rate_hz"] for x in r]))
        out["c3"]["network"][c] = net
        st = [g for g in d["c3_conditions"][c] if not g.startswith("DNg02")]
        sv = f"{np.mean([x['group_rate_hz'][st[0]] for x in r]):.1f}" if st else "-"
        L.append(f"| {c} | {net:.4f} | {np.mean([x['nondriven_rate_hz'] for x in r]):.4f} | "
                 f"{np.mean([x['group_rate_hz']['DNg02_L'] for x in r]):.1f} / "
                 f"{np.mean([x['group_rate_hz']['DNg02_R'] for x in r]):.1f} | {sv} |")
    L += ["", "### 5.3 Decision per family (prereg/07 section 3)", "",
          f"Gate-passing steering MN types used ({len(passed)}): {', '.join(passed)}. Failing, reported separately "
          f"({len(failed)}): {', '.join(failed) if failed else 'none'}. Tests in the Holm set: {len(tests)} "
          f"(6 families x 2 drive sides x {len(passed)} MN types). Primary: paired t-test; sensitivity: Wilcoxon "
          "signed-rank (with 10 seeds its smallest exact p is 0.00195, so it cannot pass Holm at this test count; "
          "uncorrected p is shown in 5.4).", "",
          "| family | steers (t-test, Holm) | MN types and sign | steers (Wilcoxon, Holm) |", "|---|---|---|---|"]
    for f in fams:
        r = res[f]
        L.append(f"| {f} | {'yes' if r['steers'] else 'no'} | "
                 f"{', '.join(h['mn'] + ' ' + h['sign'] for h in r['mn']) if r['mn'] else '-'} | "
                 f"{'yes' if r['steers_wilcoxon'] else 'no'} |")
    L += ["", "### 5.4 Tests with a non-zero mean change (d = (R - L) under steering minus (R - L) under baseline, Hz; "
          "rows where both sides' d are identically 0 are omitted)", "",
          "| family | MN type | left drive: mean d | p (t) | p Holm (t) | p (Wilcoxon) | right drive: mean d | p (t) | "
          "p Holm (t) | p (Wilcoxon) |", "|---|---|---|---|---|---|---|---|---|---|"]
    n_zero = 0
    for f in fams:
        for m in passed:
            a, b = idx[(f, m, "L")], idx[(f, m, "R")]
            if not any(a["d"]) and not any(b["d"]):
                n_zero += 1
                continue
            L.append(f"| {f} | {m} | {a['mean']:+.2f} | {a['p_t']:.2g} | {a['p_t_holm']:.2g} | {a['p_w']:.2g} | "
                     f"{b['mean']:+.2f} | {b['p_t']:.2g} | {b['p_t_holm']:.2g} | {b['p_w']:.2g} |")
    out["c3"]["n_pairs_all_zero"] = n_zero
    L += ["", f"Family x MN-type pairs with d identically 0 under both drive sides (omitted): {n_zero} of "
          f"{len(fams) * len(passed)}.", "",
          "### 5.5 Gate-failing steering MN types (descriptive, not used)", "",
          "| family | MN type | left drive: mean d (Hz) | right drive: mean d (Hz) |", "|---|---|---|---|"]
    out["c3"]["gate_failing_d"] = {}
    for f in fams:
        for m in failed:
            a, b = (float((D(f"{f}_{s}", m) - D("B", m)).mean()) for s in ("L", "R"))
            out["c3"]["gate_failing_d"][f"{f}|{m}"] = [a, b]
            L.append(f"| {f} | {m} | {a:+.2f} | {b:+.2f} |")
    L += ["", "### 5.6 Steering MN rates per condition (Hz, L / R, mean over seeds; gate-passing types with any spike)", ""]
    act = [m for m in passed if any(mean_rate(c, f"{m}|{s}") > 0 for c in d["c3_conditions"] for s in ("left", "right"))]
    L += ["| condition | " + " | ".join(act) + " |", "|---|" + "---|" * len(act)]
    for c in d["c3_conditions"]:
        L.append(f"| {c} | " + " | ".join(f"{mean_rate(c, m + '|left'):.1f} / {mean_rate(c, m + '|right'):.1f}"
                                         for m in act) + " |")
    L.append("")
    return L


def post_hoc(c1, d, out):
    """Observations not registered in prereg/07 (labelled post-hoc; not used for any decision)."""
    car = d["c2"]["carriers"]
    tot = sum(c["cut_spikes"] for c in car)
    reg = Counter(c["region"] for c in car)
    sp = Counter()
    for c in car:
        sp[c["region"]] += c["cut_spikes"]
    runs = d["c2"]["runs"]
    n_on = d["n_on"]
    net = np.mean([r["network_step_hz"] for r in runs], axis=0)
    drive_end, cut_end = float(net[n_on - 4:n_on].mean()), float(net[-4:].mean())
    N = c1["neurons"]
    seeds = sorted(c1["f0"])
    fl = np.mean([c1["f0"][s]["flight"]["rate_hz"] for s in seeds], axis=0)
    rng = {f: (float(fl[[k for k, x in enumerate(N) if x["family"] == f]].min()),
               float(fl[[k for k, x in enumerate(N) if x["family"] == f]].max())) for f in c1["families"]}
    own = {c: float(np.mean([x["group_rate_hz"][g] for x in d["c3"] if x["condition"] == c
                             for g in d["c3_conditions"][c] if not g.startswith("DNg02")]))
           for c in d["c3_conditions"] if c != "B"}
    out["post_hoc"] = dict(c2_region={k: dict(neurons=reg[k], spikes=sp[k]) for k in reg}, c2_cut_spikes_total=tot,
                           c2_net_last100_drive=drive_end, c2_net_last100_cut=cut_end, f0_flight_range=rng,
                           c3_steering_dn_own_rate=own)
    L = ["## 7. Post-hoc observations (not registered in prereg/07; not used for any decision)", "",
         f"- **C2, where the persistent activity sits.** Network rate in the last 100 ms of the drive "
         f"{drive_end:.4f} Hz, in the last 100 ms of the 1 s cut {cut_end:.4f} Hz: the activity does not decay within "
         "1 s. Wing power MNs (DLM, DVM, tp2, ps1) stop within 100-200 ms of the cut; neck MNs and right b3/hg4 keep "
         "firing. Carriers by BANC `region` (cut spikes, all seeds): "
         + ", ".join(f"{k} {reg[k]} neurons / {sp[k]} spikes ({100 * sp[k] / tot:.0f} %)"
                     for k in sorted(reg, key=lambda k: -sp[k]))
         + ". The BANC graph contains its own brain; most of the persistent activity is carried by BANC brain-region "
         "neurons reached from the DNg02 drive, not by VNC neurons. Gate 2 passes because its threshold is 1 Hz.",
         "- **C3 drive rate against the FlyWire rate range.** C3 drove each steering family at 100 Hz. The FlyWire F0 "
         "flight-window rates of the same families (seed mean, min-max over the family's neurons): "
         + ", ".join(f"{f} {lo:.1f}-{hi:.1f} Hz" for f, (lo, hi) in rng.items() if f != "DNg02")
         + ". Whether the steering effects found at 100 Hz persist at these rates was not tested.",
         "- **Driven-DN own rate in C3** (Poisson drive 100 Hz plus network input; mean over seeds): "
         + ", ".join(f"{c} {v:.1f} Hz" for c, v in own.items()) + ".", ""]
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if (OUT.with_suffix(".md").exists() or OUT.with_suffix(".json").exists()) and "--overwrite" not in argv:
        raise SystemExit("output exists; use --overwrite")
    c1 = json.loads((R / "c1_rates.json").read_text())
    d = json.loads((R / "f3a2_v3.json").read_text())
    out = {}
    g1 = sum(r["spikes"] for r in d["gate1"])
    g2 = d["gate2_runs"]
    ign = sum(r["ignited"] for r in g2)
    lo, hi = cp(ign, len(g2))
    nets = {"C2": float(np.mean([r["network_rate_hz"] for r in d["c2"]["runs"]]))}
    for c in d["c3_conditions"]:
        nets[c] = float(np.mean([x["network_rate_hz"] for x in d["c3"] if x["condition"] == c]))
    nets["gate2"] = float(np.mean([r["network_rate_hz"] for r in g2]))
    mx = max(nets, key=nets.get)
    gates = dict(gate1_spikes=g1, gate1_pass=g1 == 0, gate2_ignited=ign, gate2_n=len(g2), gate2_ci=[lo, hi],
                 gate2_pass=ign == 0, gate2_mean_last_cut_step_hz=float(np.mean([r["last_cut_step_hz"] for r in g2])),
                 gate3_max_network_hz=nets[mx], gate3_max_condition=mx, gate3_pass=nets[mx] < 5.0)
    out["gates"] = gates
    ok = gates["gate1_pass"] and gates["gate2_pass"] and gates["gate3_pass"]
    L = ["# F3a-2: DN rate range, persistence and steering on top of power (prereg/07_f3a2_rate_persistence_steering.md)",
         "", "Generated by `scripts/f3a2_c1_rates.py`, `scripts/f3a2_vnc_run.py` and `scripts/f3a2_report.py`; records "
         "`runs/f3a2/c1_rates.json` and `runs/f3a2/f3a2_v3.json` (not committed). C1 is data only (existing records); "
         "C2 and C3 are open-loop runs on the BANC v3 VNC only (no brain, no body, no hybrid). Criteria, seeds and "
         "windows are those of the committed pre-registration.", "",
         "## 1. Graph and driven groups", "",
         f"- `{d['graph_meta']['data_source']}`: {d['n_neurons']:,} neurons, {d['pairs']:,} pairs, {d['synapses']:,} "
         f"synapses; reference arm (w_syn 0.275 mV, scale 1.0); peak RAM {d['peak_rss_gb']:.2f} GB, wall "
         f"{d['total_wall_s']:.0f} s.",
         "- DN groups (neurons): " + ", ".join(f"{k} {v['n']}" for k, v in d["dn_groups"].items()) + ".",
         f"- C3 baseline: DNg02 both sides 100 Hz, label `{d['baseline_label']}`.", "",
         "## 2. Gates", "", "| gate | result | pass |", "|---|---|---|",
         f"| 1: no input, seeds 0-4 (spikes) | {g1} | {'pass' if gates['gate1_pass'] else 'FAIL'} |",
         f"| 2: ignition after the 200 ms cut, all 14 groups 100 Hz, {len(g2)} seeds | {ign} / {len(g2)} (95 % CI "
         f"{lo:.3f}-{hi:.3f}); mean last-cut-step rate {gates['gate2_mean_last_cut_step_hz']:.3f} Hz | "
         f"{'pass' if gates['gate2_pass'] else 'FAIL'} |",
         f"| 3': network rate under drive < 5 Hz in every condition | max {nets[mx]:.3f} Hz ({mx}) | "
         f"{'pass' if gates['gate3_pass'] else 'FAIL'} |", "",
         ("All gates pass; C2 and C3 are usable for the interface pre-registration." if ok else
          "**A gate failed: per prereg/07 section 5, C2 and C3 are reported but not used for the interface.**"), ""]
    L += c1_section(c1, out)
    L += c2_section(d, out)
    L += c3_section(d, out)
    st = [f for f, r in out["c3"]["families"].items() if r["steers"]]
    L += ["## 6. Summary (prereg/07 section 5)", "",
          f"- Gates: {'all pass' if ok else 'a gate failed'}.",
          f"- C1, DNg02 nearly silent in the FlyWire brain: F0 "
          f"{'confirmed' if out['c1']['dng02_nearly_silent']['f0'] else 'refuted'}, v1 screen "
          f"{'confirmed' if out['c1']['dng02_nearly_silent']['v1_screen'] else 'refuted'}.",
          f"- C2: {out['c2']['n_carriers']} neurons carry activity after the cut; network rate in the last cut step "
          f"(mean over seeds) {np.mean(out['c2']['last_cut_step_hz']):.4f} Hz.",
          f"- C3, families that steer (t-test, Holm): {', '.join(st) if st else 'none'}; do not steer: "
          f"{', '.join(f for f in out['c3']['families'] if f not in st) or 'none'}.",
          "- Which families go into F3b is the project owner's decision (prereg/07 section 5).", ""]
    L += post_hoc(c1, d, out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1))
    OUT.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
