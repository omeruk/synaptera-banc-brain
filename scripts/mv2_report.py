"""Render docs/results/mv2_neuron_model.{md,json} from the prereg/12 run records (numbers by script, rule 11).

    env -u PYTHONPATH python scripts/mv2_report.py [--overwrite]

Records (not committed): runs/mv2/t1_<arm>.json (test i), t2_<arm>.json (test ii), t3_<arm>/brain + t3_<arm>.json
(test iii), t4_<arm>.json (test iv). The arms reported are those that ran (prereg/12 section 2: R and ACG).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import f4_design_numbers as F4  # noqa: E402
import mv2_common as M  # noqa: E402
from f3a2_report import ALPHA, cp, holm, p_t  # noqa: E402

RUNS = M.RUNS
OUT = ROOT / "docs" / "results" / "mv2_neuron_model"
F3A3_STORED = ROOT / "runs" / "f3a3" / "f3a3_v3.json"
F4_V1 = ROOT / "docs" / "results" / "f4_design_numbers.json"
SETS = {k: F4.SETS_V2[k] for k in (M.T3_PRIMARY_SET,) + M.T3_SECONDARY_SETS}
STIMS = ("rot_left", "rot_right", "static", "grey")
OTHER_MN = ("TTMn", "PSI", "DLM1-4", "DLM5")


def arms_run():
    return [a for a in M.ARMS if (RUNS / f"t1_{a}.json").exists() or (a == "R" and (RUNS / "t2_R.json").exists())]


def p_one_sided(x):
    """H1: mean > 0. All zero: 1; zero variance: 0 if the mean is positive, else 1."""
    x = np.asarray(x, float)
    if not x.any():
        return 1.0
    if np.std(x) == 0:
        return 0.0 if x.mean() > 0 else 1.0
    return float(ttest_1samp(x, 0.0, alternative="greater").pvalue)


def load(p):
    return json.loads(Path(p).read_text())


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── test (i) ────────────────────────────────────────────────────────────────────────────────────────────────────
def test1(arms):
    out = {}
    for a in arms:
        p = RUNS / f"t1_{a}.json"
        if not p.exists():
            continue
        d = load(p)
        mn9 = np.array([np.mean(r["mn9_hz"]) for r in d["runs"]])
        ign = int(sum(r["ignited"] for r in d["runs"]))
        out[a] = dict(record=rel(p), sha256=sha(p), seeds=d["seeds"], n=len(d["runs"]),
                      mn9_mean_hz=float(mn9.mean()), mn9_sd_hz=float(mn9.std(ddof=1)),
                      mn9_per_seed_hz=mn9.tolist(), n_ignited=ign, ignition_ci=list(cp(ign, len(d["runs"]))),
                      no_input_spikes=int(sum(r["spikes"] for r in d["no_input"])),
                      network_on_hz=float(np.mean([r["network_on_hz"] for r in d["runs"]])),
                      n_active_on=float(np.mean([r["n_active_on"] for r in d["runs"]])),
                      sugar_hz=float(np.mean([r["sugar_hz"] for r in d["runs"]])),
                      last_cut_max_hz=float(max(r["last_cut_hz"] for r in d["runs"])),
                      gap_pairs=len(d["gap_pairs"]), wall_s=d["total_wall_s"])
    r = out.get("R")
    for a, o in out.items():
        if a == "R" or r is None:
            continue
        o["mn9_ratio_to_R"] = o["mn9_mean_hz"] / r["mn9_mean_hz"] if r["mn9_mean_hz"] > 0 else None
        o["fisher_p_vs_R"] = float(fisher_exact([[o["n_ignited"], o["n"] - o["n_ignited"]],
                                                 [r["n_ignited"], r["n"] - r["n_ignited"]]])[1])
        o["pass_mn9"] = o["mn9_ratio_to_R"] is not None and o["mn9_ratio_to_R"] >= M.T1_MN9_MIN_FRACTION
        o["pass_ignition"] = o["n_ignited"] <= r["n_ignited"]
        o["pass_no_input"] = o["no_input_spikes"] == 0
        o["passed"] = bool(o["pass_mn9"] and o["pass_ignition"] and o["pass_no_input"])
    return out


# ── test (ii) ───────────────────────────────────────────────────────────────────────────────────────────────────
def gates_f3a3(d):
    g1 = sum(r["spikes"] for r in d["gate1"])
    g2 = d["gate2_runs"]
    ign = sum(r["ignited"] for r in g2)
    nets = {f"R1 {c}": float(np.mean([x["network_rate_hz"] for x in d["r1"] if x["condition"] == c])) for c in d["r1_drives"]}
    nets["R2"] = float(np.mean([r["network_rate_hz"] for r in d["r2"]["runs"]]))
    for c in d["r3_conditions"]:
        nets[f"R3 {c}"] = float(np.mean([x["network_rate_hz"] for x in d["r3"] if x["condition"] == c]))
    nets["gate2"] = float(np.mean([r["network_rate_hz"] for r in g2]))
    mx = max(nets, key=nets.get)
    return dict(gate1_spikes=int(g1), gate2_ignited=int(ign), gate2_n=len(g2), gate3_max_network_hz=nets[mx],
                gate3_max_condition=mx, passed=bool(g1 == 0 and ign == 0 and nets[mx] < 5.0))


def same_as_stored(d):
    s = load(F3A3_STORED)
    keys = ("gate1", "r1", "r3", "gate2_runs")
    eq = {k: json.dumps(d[k], sort_keys=True) == json.dumps(s[k], sort_keys=True) for k in keys}
    eq["r2_runs"] = json.dumps(d["r2"]["runs"], sort_keys=True) == json.dumps(s["r2"]["runs"], sort_keys=True)
    return eq


def test2(arms):
    import f3a3_report as R3
    out = {}
    for a in arms:
        p = RUNS / f"t2_{a}.json"
        if not p.exists():
            continue
        d = load(p)
        o = {}
        R3.r1_section(d, o)
        R3.r2_section(d, o)
        R3.r3_section(d, o)
        g = gates_f3a3(d)
        r1 = {}
        for c in d["r1_drives"]:
            for k in sorted(d["r1"][0]["rate_hz"]):
                v = float(np.mean([x["rate_hz"][k] for x in d["r1"] if x["condition"] == c]))
                r1.setdefault(c, {})[k] = v
        out[a] = dict(record=rel(p), sha256=sha(p), gates=g, r2_n_decay=o["r2"]["n_decay"], r2_passed=o["r2"]["passed"],
                      r2_A=o["r2"]["A"], r2_B=o["r2"]["B"], lowest_rate=o["r3"]["lowest_rate"],
                      gate_passing=o["r3"]["gate_passing"],
                      steers={f: {r: v["steers"] for r, v in fam.items()} for f, fam in o["r3"]["families"].items()},
                      steer_mn={f: {r: v["mn"] for r, v in fam.items()} for f, fam in o["r3"]["families"].items()},
                      own_rate_hz={f: {r: v["own_rate_hz"] for r, v in fam.items()} for f, fam in o["r3"]["families"].items()},
                      r1_mean_hz=r1, passed=bool(g["passed"] and o["r2"]["passed"]), wall_s=d["total_wall_s"])
        if a == "R":
            out[a]["equals_prereg08_record"] = same_as_stored(d)
    r = out.get("R")
    if r is not None:
        for a, o in out.items():
            cmp = {}
            for f, v in o["lowest_rate"].items():
                rv = r["lowest_rate"][f]
                cmp[f] = ("none" if v is None else "equal" if v == rv else "new" if rv is None
                          else "lower" if v < rv else "higher")
            o["lowest_vs_R"] = cmp
    return out


# ── test (iii) ──────────────────────────────────────────────────────────────────────────────────────────────────
def s_values(rates, arm, seeds, signs):
    return F4.s_index(rates, arm, seeds, signs)


def offset(rates, arm, seeds, signs):
    return (F4.dA_stim(rates, arm, "rot_right", seeds, signs) + F4.dA_stim(rates, arm, "rot_left", seeds, signs)) / 2.0


def test3(arms):
    g1 = load(F4_V1)["gain"]["G1_deg_per_hz"]
    out, s_ens = {}, {}
    for a in arms:
        p = RUNS / f"t3_{a}.json"
        if not p.exists():
            continue
        d = load(p)
        rates, seeds = F4.load_rates(d)
        bridge_arms = list(d["constants"]["arms"])
        randoms = [x for x in bridge_arms if x.startswith("random_")]
        sets = {}
        for name, signs in SETS.items():
            arm_stats = {}
            for b in bridge_arms:
                s = s_values(rates, b, seeds, signs)
                arm_stats[b] = dict(mean=float(s.mean()), sd=float(s.std(ddof=1)), values=s.tolist(),
                                    p_one_sided=p_one_sided(s), p_two_sided=p_t(s), n_pos=int((s > 0).sum()),
                                    n_neg=int((s < 0).sum()),
                                    offset_mean=float(offset(rates, b, seeds, signs).mean()))
            e = abs(arm_stats["ensemble"]["mean"])
            n_ge = sum(abs(arm_stats[k]["mean"]) >= e for k in randoms)
            rnd = [arm_stats[k]["mean"] for k in randoms]
            sets[name] = dict(signs=signs, arms=arm_stats, perm_p=(1 + n_ge) / (len(randoms) + 1),
                              n_random_abs_ge=int(n_ge), n_random=len(randoms),
                              random_n_syndirectional=int(sum(v > 0 for v in rnd)),
                              random_n_anti=int(sum(v < 0 for v in rnd)),
                              turn_deg_G1=float(g1 * arm_stats["ensemble"]["mean"]))
        s_ens[a] = np.array(sets[M.T3_PRIMARY_SET]["arms"]["ensemble"]["values"])
        # per MN type (gate-passing steering types on the subgraph)
        types = [m for m in d["constants"]["steer_mn"] if d["symmetry"].get(m, {}).get("passed")]
        D = F4.d_per_type(rates, "ensemble", seeds, types)
        ph = holm([p_t(D[m]) for m in types])
        lat = SETS[M.T3_PRIMARY_SET]
        per_type = {m: dict(T=float(D[m].mean()), sd=float(D[m].std(ddof=1)), p=p_t(D[m]), p_holm=float(h),
                            contribution_to_S=(float(-lat[m] * D[m].mean() / 2.0) if m in lat else None))
                    for m, h in zip(types, ph)}
        # gates
        g1s = int(sum(r["spikes"] for r in d["gate1"]))
        ign = int(sum(r["ignited"] for r in d["gate2_runs"]))
        nets = {}
        for b in bridge_arms:
            for st in STIMS:
                nets[f"{b}|{st}"] = float(np.mean([r["network_rate_hz"] for r in d["replay"]
                                                   if r["arm"] == b and r["stimulus"] == st]))
        nets["gate2"] = float(np.mean([r["network_rate_hz"] for r in d["gate2_runs"]]))
        mx = max(nets, key=nets.get)
        gates = dict(gate1_spikes=g1s, gate2_ignited=ign, gate2_n=len(d["gate2_runs"]), gate2_ci=list(cp(ign, len(d["gate2_runs"]))),
                     gate3_max_network_hz=nets[mx], gate3_max_condition=mx,
                     passed=bool(g1s == 0 and ign == 0 and nets[mx] < 5.0))
        ens = [r for r in d["replay"] if r["arm"] == "ensemble"]
        other = {st: {f"{m}|{sd}": float(np.mean([r["rate_hz"].get(f"{m}|{sd}", 0.0) for r in ens if r["stimulus"] == st]))
                      for m in OTHER_MN for sd in ("left", "right")} for st in STIMS}
        gf = {st: {sd: float(np.mean([r["dn_realised_hz"].get(f"DNp01|{sd}", 0.0) for r in ens if r["stimulus"] == st]))
                   for sd in ("left", "right")} for st in STIMS}
        brain = {st: float(np.mean([b["net_rate_hz"] for b in d["brain"] if b["stimulus"] == st])) for st in STIMS}
        out[a] = dict(record=rel(p), sha256=sha(p), brain_records=d["brain_records"]["dir"],
                      n_brain_records=len(d["brain_records"]["sha256"]), seeds=seeds, bridge_arms=bridge_arms,
                      sets=sets, per_type=per_type, gates=gates, mn_rates_ensemble=other, gf_realised_hz=gf,
                      brain_net_rate_hz=brain, applied_max_hz=float(max(r["steer_applied_mean_hz"] for r in ens)),
                      wall_s=d["total_wall_s"])
    # Holm over the arms that ran (primary, one-sided)
    names = list(out)
    ph = holm([out[a]["sets"][M.T3_PRIMARY_SET]["arms"]["ensemble"]["p_one_sided"] for a in names]) if names else []
    for a, h in zip(names, ph):
        o = out[a]
        prim = o["sets"][M.T3_PRIMARY_SET]
        e = prim["arms"]["ensemble"]
        o["p_one_sided_holm"] = float(h)
        c1 = e["mean"] > 0 and h < M.T3_ALPHA
        c2 = prim["perm_p"] <= M.T3_ALPHA and prim["n_random_abs_ge"] == 0
        c3 = prim["arms"]["bridge_off"]["p_two_sided"] >= M.T3_ALPHA
        c4 = o["gates"]["passed"]
        o["criteria"] = dict(syndirectional_significant=bool(c1), specific=bool(c2), bridge_off_silent=bool(c3),
                             gates=bool(c4))
        if c1 and c2 and c3 and c4:
            outcome = "pass"
        elif c1 and not c2:
            outcome = "syndirectional, significant, not specific"
        elif e["mean"] < 0 and e["p_two_sided"] < M.T3_ALPHA:
            outcome = "anti-directional"
        elif c1:
            outcome = "syndirectional, significant, specific; failed " + ", ".join(
                k for k, v in (("bridge_off silent", c3), ("gates", c4)) if not v)
        else:
            outcome = "not significant"
        o["outcome"] = outcome
    if "R" in s_ens:
        for a in s_ens:
            if a != "R":
                dd = s_ens[a] - s_ens["R"]
                out[a]["S_minus_R"] = dict(mean=float(dd.mean()), sd=float(dd.std(ddof=1)), p_two_sided=p_t(dd))
    return out


# ── test (iv) ───────────────────────────────────────────────────────────────────────────────────────────────────
def test4(arms):
    out = {}
    for a in arms:
        p = RUNS / f"t4_{a}.json"
        if not p.exists():
            continue
        d = load(p)
        per = {}
        for sd in ("left", "right"):
            fol = [r["per_side"][sd]["following"] for r in d["runs"]]
            lat = [x for r in d["runs"] for x in r["per_side"][sd]["latency_ms"]]
            per[sd] = dict(following_mean=float(np.mean([f for f in fol if f is not None])),
                           latency_median_ms=(float(np.median(lat)) if lat else None), n_latency=len(lat),
                           gf_hz=float(np.mean([r["per_side"][sd]["gf_hz"] for r in d["runs"]])),
                           ttmn_hz=float(np.mean([r["per_side"][sd]["ttmn_hz"] for r in d["runs"]])),
                           psi_hz=float(np.mean([r["per_side"][sd]["psi_hz"] for r in d["runs"]])),
                           dlm_hz=float(np.mean([r["per_side"][sd]["dlm_hz"] for r in d["runs"]])))
        # post-hoc (labelled in the report): a kick that arrives while the TTMn is refractory is lost (Brian2 holds the
        # '(unless refractory)' variables), and the GF input node has no refractory period; for Poisson GF spikes at
        # rate r the expected following is that of a dead-time counter, 1 / (1 + r * (t_rfc + one clock step))
        t_dead = 2.2e-3 + 1e-4
        for sd in per:
            per[sd]["posthoc_dead_time_following"] = float(1.0 / (1.0 + per[sd]["gf_hz"] * t_dead))
        ok = all(per[s]["following_mean"] >= M.T4_FOLLOW_MIN and per[s]["latency_median_ms"] is not None
                 and per[s]["latency_median_ms"] < M.T4_LATENCY_MAX_MS for s in per)
        out[a] = dict(record=rel(p), sha256=sha(p), per_side=per, gap_pairs=d["gap_pairs"],
                      chemical_synapses=d["chemical_synapses"],
                      network_on_hz=float(np.mean([r["network_on_hz"] for r in d["runs"]])),
                      passed=(bool(ok) if a != "R" else None))
    return out


def build():
    arms = arms_run()
    return dict(prereg="prereg/12_neuron_model_v2.md", arms_run=arms, arms_registered=list(M.ARMS),
                t1=test1(arms), t2=test2(arms), t3=test3(arms), t4=test4(arms))


# ── markdown ────────────────────────────────────────────────────────────────────────────────────────────────────
def f(x, n=2, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def pf(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def render(o):
    arms = o["arms_run"]
    L = ["# Neuron model v2 (open loop): results of prereg/12", "",
         "Generated by `scripts/mv2_report.py` from the run records in `runs/mv2/` (not committed). Design: "
         "`docs/09_NEURON_MODEL_V2.md`. Pre-registration: `prereg/12_neuron_model_v2.md`.", "",
         f"Arms run: {', '.join(arms)}. Registered: {', '.join(o['arms_registered'])}. The run-time projection for all "
         "four arms was above the owner's 6 h limit, so only R and ACG ran (prereg/12 section 2). A and AC are "
         "registered and not run.", ""]
    # (i)
    t1 = o["t1"]
    L += ["## 1. Test (i): the FlyWire brain does not break", "",
          f"The 36 labellar sugar GRNs are driven at {M.T1_RATE_HZ:g} Hz for 1 s, followed by a 200 ms cut. Seeds "
          f"{M.T1_SEEDS[0]}-{M.T1_SEEDS[-1]}. MN9 = CB0701, the mean of the two neurons. Ignition means a network "
          "rate > 1 Hz in the last cut step.", "",
          "| arm | MN9 mean (Hz) | MN9 s.d. | MN9 / R | ignited | 95 % CI | Fisher p vs R | no-input spikes | "
          "network rate during the drive (Hz) | active neurons | max rate in the last cut step (Hz) | pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, x in t1.items():
        L.append(f"| {a} | {f(x['mn9_mean_hz'])} | {f(x['mn9_sd_hz'])} | {f(x.get('mn9_ratio_to_R'), 3)} | "
                 f"{x['n_ignited']} / {x['n']} | {f(x['ignition_ci'][0], 3)}-{f(x['ignition_ci'][1], 3)} | "
                 f"{pf(x['fisher_p_vs_R']) if 'fisher_p_vs_R' in x else '-'} | {x['no_input_spikes']} | "
                 f"{f(x['network_on_hz'], 3)} | {f(x['n_active_on'], 0)} | {f(x['last_cut_max_hz'], 3)} | "
                 f"{'reference' if a == 'R' else ('pass' if x['passed'] else 'FAIL')} |")
    for a, x in t1.items():
        if a != "R":
            L += ["", f"**{a}: {'passes' if x['passed'] else 'fails'} test (i).** MN9 is {f(100 * x['mn9_ratio_to_R'], 1)} % "
                  f"of R (criterion >= {100 * M.T1_MN9_MIN_FRACTION:g} %). Ignition: {x['n_ignited']} vs "
                  f"{t1['R']['n_ignited']} in R (criterion: no more than R). FlyWire gap pairs: {x['gap_pairs']}, so ACG "
                  "acts like AC in the brain."]
    # (ii)
    t2 = o["t2"]
    L += ["", "## 2. Test (ii): BANC VNC subgraph, prereg/08 protocol", "",
          "| arm | gate 1 spikes | gate 2 ignited | gate 3' max network (Hz) | R2 seeds decaying | criterion | "
          "lowest steering rate DNa04 / DNa10 / DNa15 (Hz) | vs R |", "|---|---|---|---|---|---|---|---|"]
    fams = ("DNa04", "DNa10", "DNa15")
    for a, x in t2.items():
        g = x["gates"]
        low = " / ".join("none" if x["lowest_rate"][fm] is None else str(x["lowest_rate"][fm]) for fm in fams)
        vs = " / ".join(x.get("lowest_vs_R", {}).get(fm, "-") for fm in fams) if a != "R" else "-"
        L.append(f"| {a} | {g['gate1_spikes']} | {g['gate2_ignited']} / {g['gate2_n']} | "
                 f"{f(g['gate3_max_network_hz'], 3)} ({g['gate3_max_condition']}) | {x['r2_n_decay']} / 10 | "
                 f"{'pass' if x['passed'] else 'FAIL'} | {low} | {vs} |")
    if "R" in t2:
        eq = t2["R"]["equals_prereg08_record"]
        L += ["", f"Engine check: the R re-run equals the stored prereg/08 record bit for bit: "
              f"{'yes' if all(eq.values()) else 'NO (' + ', '.join(k for k, v in eq.items() if not v) + ')'}."]
    L += ["", "Steering per family and rate (paired t, Holm per rate; MN types and sign):", "",
          "| arm | family | 10 Hz | 25 Hz | 50 Hz | 100 Hz |", "|---|---|---|---|---|---|"]
    for a, x in t2.items():
        for fm in fams:
            cells = []
            for r in ("10", "25", "50", "100"):
                mns = x["steer_mn"][fm][r]
                cells.append(", ".join(f"{m['mn']} ({m['sign'][:6]}.)" for m in mns) if mns else "no")
            L.append(f"| {a} | {fm} | " + " | ".join(cells) + " |")
    power = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b", "tp2", "ps1")
    if t2:
        L += ["", "R1 at G100 (DNg02 both sides 100 Hz): power MN rates, left / right (Hz):", "",
              "| arm | " + " | ".join(power) + " |", "|---|" + "---|" * len(power)]
        for a, x in t2.items():
            r = x["r1_mean_hz"]["G100"]
            L.append(f"| {a} | " + " | ".join(f"{f(r.get(p + '|left'), 1)} / {f(r.get(p + '|right'), 1)}" for p in power) + " |")
    # (iii)
    t3 = o["t3"]
    L += ["", "## 3. Test (iii): hybrid optomotor (prereg/11 protocol, fresh brain seeds 30-39, 19 random wirings)", "",
          "S is the predicted turn under the v2 `lateral` mapping (Lindsay 2017 signs). S > 0 is syndirectional. The "
          "turn in degrees is G1 x S.", "",
          "| arm | S ensemble (Hz) | s.d. | seeds + / - | turn (deg, G1) | one-sided p | Holm over arms | two-sided p | "
          "random with abs S >= ensemble | permutation p | bridge_off S (two-sided p) | gates | outcome |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, x in t3.items():
        pr = x["sets"][M.T3_PRIMARY_SET]
        e, bo = pr["arms"]["ensemble"], pr["arms"]["bridge_off"]
        L.append(f"| {a} | {f(e['mean'], 2, True)} | {f(e['sd'])} | {e['n_pos']} / {e['n_neg']} | "
                 f"{f(pr['turn_deg_G1'], 2, True)} | {pf(e['p_one_sided'])} | {pf(x['p_one_sided_holm'])} | "
                 f"{pf(e['p_two_sided'])} | "
                 f"{pr['n_random_abs_ge']} / {pr['n_random']} | {f(pr['perm_p'], 3)} | {f(bo['mean'], 2, True)} "
                 f"({pf(bo['p_two_sided'])}) | {'pass' if x['gates']['passed'] else 'FAIL'} | **{x['outcome']}** |")
    for a, x in t3.items():
        g = x["gates"]
        L += ["", f"{a} gates: gate 1 {g['gate1_spikes']} spikes; gate 2 {g['gate2_ignited']} / {g['gate2_n']} ignited "
              f"(CI {f(g['gate2_ci'][0], 3)}-{f(g['gate2_ci'][1], 3)}); gate 3' max {f(g['gate3_max_network_hz'], 3)} Hz "
              f"({g['gate3_max_condition']})."]
    for a, x in t3.items():
        if "S_minus_R" in x:
            dd = x["S_minus_R"]
            L += ["", f"S({a}) - S(R), paired by seed: {f(dd['mean'], 2, True)} Hz (s.d. {f(dd['sd'])}, two-sided p "
                  f"{pf(dd['p_two_sided'])}); descriptive."]
    L += ["", "### 3.1 Secondary mapping sets (reported, not used for the decision)", "",
          "| arm | set | S ensemble (Hz) | one-sided p | random with abs S >= ensemble | random syndirectional / anti | "
          "offset B (Hz) |", "|---|---|---|---|---|---|---|"]
    for a, x in t3.items():
        for name, st in x["sets"].items():
            e = st["arms"]["ensemble"]
            L.append(f"| {a} | {name} | {f(e['mean'], 2, True)} | {pf(e['p_one_sided'])} | {st['n_random_abs_ge']} / "
                     f"{st['n_random']} | {st['random_n_syndirectional']} / {st['random_n_anti']} | "
                     f"{f(e['offset_mean'], 2, True)} |")
    L += ["", "### 3.2 Per MN type in `ensemble` (gate-passing steering types; two-sided t, Holm over types)", "",
          "| arm | type | T (Hz) | s.d. | p | p Holm | contribution to S (lateral) |", "|---|---|---|---|---|---|---|"]
    for a, x in t3.items():
        for m, v in x["per_type"].items():
            L.append(f"| {a} | {m} | {f(v['T'], 2, True)} | {f(v['sd'])} | {pf(v['p'])} | {pf(v['p_holm'])} | "
                     f"{f(v['contribution_to_S'], 2, True)} |")
    L += ["", "### 3.3 GF and its targets in `ensemble` (mean over seeds, Hz, left / right)", "",
          "| arm | stimulus | GF realised | TTMn | PSI | DLM1-4 | DLM5 | FlyWire network rate |", "|---|---|---|---|---|---|---|---|"]
    for a, x in t3.items():
        for st in STIMS:
            r, g = x["mn_rates_ensemble"][st], x["gf_realised_hz"][st]
            L.append(f"| {a} | {st} | {f(g['left'], 1)} / {f(g['right'], 1)} | "
                     + " | ".join(f"{f(r[m + '|left'], 1)} / {f(r[m + '|right'], 1)}" for m in OTHER_MN)
                     + f" | {f(x['brain_net_rate_hz'][st], 3)} |")
    # (iv)
    t4 = o["t4"]
    L += ["", "## 4. Test (iv): GF -> TTMn (both GFs at 100 Hz for 1 s; seeds 0-9)", "",
          "| arm | side | GF (Hz) | TTMn (Hz) | following (TTMn / GF) | median latency (ms) | TTMn spikes timed | "
          "PSI (Hz) | DLM (Hz) | criterion |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, x in t4.items():
        for sd, v in x["per_side"].items():
            L.append(f"| {a} | {sd} | {f(v['gf_hz'], 1)} | {f(v['ttmn_hz'], 1)} | {f(v['following_mean'], 3)} | "
                     f"{f(v['latency_median_ms'], 2)} | {v['n_latency']} | {f(v['psi_hz'], 1)} | {f(v['dlm_hz'], 1)} | "
                     f"{'-' if x['passed'] is None else ('pass' if x['passed'] else 'FAIL')} |")
    if "ACG" in t4:
        L += ["", f"ACG criterion (following >= {M.T4_FOLLOW_MIN} and median latency < {M.T4_LATENCY_MAX_MS:g} ms on "
              f"both sides): {'pass' if t4['ACG']['passed'] else 'FAIL'}. This checks the implementation against the "
              "property it was built from (prereg/12 section 7). It is not an independent finding."]
        v = t4["ACG"]["per_side"]
        L += ["", "**Post-hoc diagnosis (not registered).** The TTMn misses GF spikes for two reasons:",
              "- A gap kick that arrives while the TTMn is refractory (t_rfc 2.2 ms) is lost. Brian2 holds the "
              "`(unless refractory)` variables during refractoriness; in this engine an addition to v in that period is not kept "
              "(checked on a toy network). The same holds for chemical input to g in every arm, R included (`tests/test_model_variants.py`, "
              "`test_input_during_refractoriness_is_lost`).",
              "- The GF is a Poisson input node without a refractory period.",
              "",
              "A dead-time counter therefore predicts a following of 1 / (1 + r_GF (t_rfc + 0.1 ms)): "
              + ", ".join(f"{sd} {f(v[sd]['posthoc_dead_time_following'], 3)} (observed {f(v[sd]['following_mean'], 3)})"
                          for sd in ("left", "right"))
              + ". The registered threshold 0.9 cannot be reached at a 100 Hz Poisson GF without changing the "
              "refractory handling; nothing was changed."]
    # summary
    L += ["", "## 5. Summary", ""]
    for a in arms:
        parts = []
        if a in t1 and a != "R":
            parts.append(f"(i) {'pass' if t1[a]['passed'] else 'FAIL'}")
        if a in t2:
            parts.append(f"(ii) {'pass' if t2[a]['passed'] else 'FAIL'}")
        if a in t3:
            parts.append(f"(iii) {t3[a]['outcome']}")
        if a in t4 and t4[a]["passed"] is not None:
            parts.append(f"(iv) {'pass' if t4[a]['passed'] else 'FAIL'}")
        L.append(f"- **{a}:** " + "; ".join(parts) + ".")
    L += ["", "Controls (rule 7): `bridge_off` and the 19 random wirings in (iii). Degree-matched random graphs are not "
          "part of this open-loop model comparison (prereg/12 section 8).", ""]
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if (OUT.with_suffix(".md").exists() or OUT.with_suffix(".json").exists()) and "--overwrite" not in argv:
        raise SystemExit("output exists; use --overwrite")
    o = build()
    OUT.with_suffix(".json").write_text(json.dumps(o, indent=1) + "\n")
    OUT.with_suffix(".md").write_text(render(o))
    print("wrote", OUT.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
