"""Data-only numbers for docs/08_CLOSED_LOOP_OPTOMOTOR.md (design analysis, not a test; no simulation).

    env -u PYTHONPATH python scripts/f4_design_numbers.py            # writes docs/results/f4_design_numbers.json
                                                                       # and the GEN blocks of docs/08
Reads the F3b-3 replay records (runs/f3b3/f3b3_v3.json), the F3a-3 report JSON, the F0 latency report JSON, the
configuration files and the F3b-3 logs. Nothing is simulated.

Convention (same as scripts/f3b3_report.py): D_m = (R - L)(rot_right) - (R - L)(rot_left) of the per-neuron MN rate.
Mapping: s_m = +1 if the MN's activity raises the ipsilateral stroke amplitude, -1 if it lowers it; the global gain G
(deg per Hz) is not applied in the prediction (G only scales). A_side = A0 + G sum_m s_m r_{m,side};
dA = A_left - A_right = G sum_m s_m (r_L - r_R). dA > 0 turns the fly to the right (quasi_steady.turn_to_amplitudes).
rot_right (grating moves to the fly's right) is followed by a right turn in a syndirectional response, so a
syndirectional response has dA(rot_right) > 0 > dA(rot_left). Syndirectional index S = (dA(rot_right) - dA(rot_left)) / 2
per seed (G = 1, Hz); S > 0 = syndirectional, S < 0 = anti-directional.
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORDS = ROOT / "runs" / "f3b3" / "f3b3_v3.json"
LOGS = {"brain": ROOT / "runs" / "f3b3" / "brain.log", "vnc": ROOT / "runs" / "f3b3" / "vnc.log"}
F3A3 = ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json"
F0LAT = ROOT / "docs" / "results" / "f0_latency_hover_yaw.json"
OUT_JSON = ROOT / "docs" / "results" / "f4_design_numbers.json"
DOC = ROOT / "docs" / "08_CLOSED_LOOP_OPTOMOTOR.md"

STIMS = ("rot_left", "rot_right", "static", "grey")
N_RANDOM = 39
RANDOM_ARMS = [f"random_{i}" for i in range(1, N_RANDOM + 1)]
ALPHA = 0.05
FLIGHT_LEAD_S, FLIGHT_STIM_S, INTERVAL_MS = 0.5, 2.0, 5.0     # design of docs/08 (grey lead-in, drum rotation)
N_ARMS = 9      # ensemble, bridge_off, sign_flip, random_1..5, v1_hand

# Mapping signs: effect of the MN's activity on ipsilateral stroke amplitude (docs/08 section 4 gives the evidence).
# "direct": a published statement that relates the muscle's activity to wing-stroke amplitude.
# "lateral": direct set + signs inferred from Table 1 of Cheong et al. (activity on the inner / outer side of a turn),
# using the rule that the outer wing has the larger amplitude in a turn (the convention of body/quasi_steady.py).
SIGNS_DIRECT = {"b2": +1, "b3": -1}
SIGNS_LATERAL = {"b2": +1, "b3": -1, "i1": -1, "iii1": +1, "hg1": -1, "hg3": -1, "hg4": -1}
SETS = {
    "lateral": SIGNS_LATERAL,
    "direct": SIGNS_DIRECT,
    "lateral_i1_flipped": {**SIGNS_LATERAL, "i1": +1},
    "lateral_no_hg": {k: v for k, v in SIGNS_LATERAL.items() if not k.startswith("hg")},
    "lateral_no_hg4": {k: v for k, v in SIGNS_LATERAL.items() if k != "hg4"},
}
PRIMARY_SET = "lateral"
GATE_FAIL = ("b1", "iii3")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rates(rec):
    """rates[arm][stim] -> {type|side: array(seeds)} in seed order; seeds sorted."""
    seeds = sorted({r["seed"] for r in rec["replay"]})
    out = {}
    for r in rec["replay"]:
        out.setdefault(r["arm"], {}).setdefault(r["stimulus"], {})[r["seed"]] = r["rate_hz"]
    return out, seeds


def side_diff(rates, arm, stim, seeds, typ):
    """(R - L) of the type per seed."""
    a = rates[arm][stim]
    return np.array([a[s].get(f"{typ}|right", 0.0) - a[s].get(f"{typ}|left", 0.0) for s in seeds])


def d_per_type(rates, arm, seeds, types):
    return {t: side_diff(rates, arm, "rot_right", seeds, t) - side_diff(rates, arm, "rot_left", seeds, t)
            for t in types}


def dA_stim(rates, arm, stim, seeds, signs):
    """dA = A_left - A_right (G = 1) per seed = - sum s (R - L)."""
    return -sum(s * side_diff(rates, arm, stim, seeds, t) for t, s in signs.items())


def s_index(rates, arm, seeds, signs):
    return (dA_stim(rates, arm, "rot_right", seeds, signs) - dA_stim(rates, arm, "rot_left", seeds, signs)) / 2.0


def summarise(x):
    x = np.asarray(x, float)
    sd = float(x.std(ddof=1))
    p = float(stats.ttest_1samp(x, 0.0).pvalue) if sd > 0 else 1.0
    return dict(mean=float(x.mean()), sd=sd, p_t=p, n_pos=int((x > 0).sum()), n_neg=int((x < 0).sum()), n=int(x.size),
                values=[float(v) for v in x])


def quantiles(v):
    q = np.quantile(np.asarray(v, float), [0, .05, .25, .5, .75, .95, 1])
    return [float(a) for a in q]


def parse_constants():
    from configs import load
    body = load("body_flight")["wing"] if "wing" in load("body_flight") else load("body_flight")
    rd = load("readouts")
    out = {k: float(body[k]) for k in ("turn_dphi_deg", "yaw_rate_per_turn", "yaw_damping_tau", "phi0_deg", "f0_hz")}
    r = rd["steer"] if isinstance(rd.get("steer"), dict) else rd
    out["turn_bias_max"] = float(r["turn_bias_max"])
    src = (ROOT / "tasks" / "t0_stimuli.py").read_text()
    m = re.search(r"^WAVELENGTH_DEG, TF_HZ = ([0-9.]+), ([0-9.]+)", src, re.M)
    out["wavelength_deg"], out["tf_hz"] = float(m.group(1)), float(m.group(2))
    return out


def timings():
    t = {}
    b = LOGS["brain"].read_text()
    t["render_s_per_stimulus"] = [int(x) for x in re.findall(r"render \S+\s+\(t0 cond \d+\).*?\((\d+) s\)", b)]
    t["brain_s_per_run"] = [int(x) for x in re.findall(r"s\d+: DN spikes \d+, net [\d.]+ Hz \((\d+) s", b)]
    t["brain_engine_build_s"] = int(re.search(r"engine [\d,]+ neurons, [\d,]+ synapses; free [\d.]+ GB \((\d+) s\)", b).group(1))
    v = LOGS["vnc"].read_text()
    t["vnc_engine_build_s"] = int(re.search(r"VNC engine [\d,]+ neurons.*?\((\d+) s\)", v).group(1))
    free = [float(x) for x in re.findall(r"free ([\d.]+) GB", b)]
    t["brain_free_gb_start"], t["brain_free_gb_after_engine"] = free[0], free[1]
    ends = [int(x) for x in re.findall(r"net [\d.]+ Hz \((\d+) s\)", v)]
    t["vnc_s_per_arm_stimulus"] = [b2 - a for a, b2 in zip(ends[:-1], ends[1:]) if b2 > a][:200]
    return t


def g_candidates(signs, const, v1_turn):
    tests = json.load(open(F3A3))["r3"]["tests"]
    cells = {}
    for t in tests:
        if t["mn"] in signs:
            key = (t["family"], t["rate"], t["side"])
            cells.setdefault(key, 0.0)
            # d = change of (R - L) of the type under the steering drive; dA change = - s d
            cells[key] += -signs[t["mn"]] * t["mean"]
    best = max(cells.items(), key=lambda kv: abs(kv[1]))
    a_max = 2.0 * const["turn_dphi_deg"] * const["turn_bias_max"]            # deg, v1 range of A_L - A_R
    dna04_25 = [v for (f, r, s), v in cells.items() if f == "DNa04" and r == 25]
    ref = max(abs(v) for v in dna04_25)
    a_v1 = 2.0 * const["turn_dphi_deg"] * float(np.median(v1_turn))
    return dict(
        cells={f"{f}|{r}|{s}": v for (f, r, s), v in sorted(cells.items())},
        max_cell=dict(key="|".join(map(str, best[0])), dA_per_G_hz=float(best[1])),
        v1_dA_max_deg=a_max,
        G1_deg_per_hz=a_max / abs(best[1]),
        dna04_25_dA_per_G_hz=ref,
        v1_turn_dn_median=float(np.median(v1_turn)),
        v1_dA_typical_deg=a_v1,
        G2_deg_per_hz=a_v1 / ref,
    )


def build():
    rec = json.load(open(RECORDS))
    rates, seeds = load_rates(rec)
    assert all(len(rates[a]) == 4 for a in rates) and len(rates) == 2 + N_RANDOM
    const = parse_constants()
    primary_types = ["b3", "i1", "i2", "hg1", "hg4"]
    out = dict(source=dict(file=str(RECORDS.relative_to(ROOT)), sha256=sha256(RECORDS), n_seeds=len(seeds),
                           seeds=seeds, n_replays=len(rec["replay"])),
               constants=const, signs={k: v for k, v in SETS.items()}, primary_set=PRIMARY_SET)

    # self-check against the registered F3b-3 statistic
    t_ens = {t: float(d_per_type(rates, "ensemble", seeds, [t])[t].mean()) for t in primary_types}
    f3b3 = json.load(open(ROOT / "docs" / "results" / "f3b3_specificity.json"))
    ref = {t: f3b3["primary"]["tests"]["ensemble"][t]["mean"] for t in primary_types}
    out["selfcheck_T_equals_f3b3"] = bool(all(abs(t_ens[t] - ref[t]) < 1e-9 for t in primary_types))
    out["T_ensemble"] = t_ens

    sets = {}
    for name, signs in SETS.items():
        res = {}
        for arm in ["ensemble", "bridge_off"] + RANDOM_ARMS:
            res[arm] = summarise(s_index(rates, arm, seeds, signs))
        res["sign_flip"] = summarise(-np.array(res["ensemble"]["values"]))
        rnd = np.array([res[a]["mean"] for a in RANDOM_ARMS])
        off = (dA_stim(rates, "ensemble", "rot_right", seeds, signs) + dA_stim(rates, "ensemble", "rot_left", seeds, signs)) / 2.0
        e = res["ensemble"]
        zsum = float(stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8))
        sets[name] = dict(
            arms=res,
            ensemble_offset=summarise(off),
            n_seeds_80pct_power=(float(np.ceil((zsum * e["sd"] / abs(e["mean"])) ** 2)) if e["mean"] != 0 else None),
            random=dict(mean_S=[float(v) for v in rnd], quantiles=quantiles(rnd),
                        n_syndirectional=int((rnd > 0).sum()), n_anti=int((rnd < 0).sum()), n_zero=int((rnd == 0).sum()),
                        n_sig_syn=int(sum(1 for a in RANDOM_ARMS if res[a]["mean"] > 0 and res[a]["p_t"] < ALPHA)),
                        n_sig_anti=int(sum(1 for a in RANDOM_ARMS if res[a]["mean"] < 0 and res[a]["p_t"] < ALPHA)),
                        n_random_below_ensemble=int((rnd < res["ensemble"]["mean"]).sum()),
                        n_random_abs_smaller=int((np.abs(rnd) < abs(res["ensemble"]["mean"])).sum())))
    out["sets"] = sets

    # per-type contributions c_m = -s_m D_m / 2 (S = sum c_m) for the primary set, ensemble; and the null scale per type
    types = list(SIGNS_LATERAL)
    dens = d_per_type(rates, "ensemble", seeds, types)
    drnd = {a: d_per_type(rates, a, seeds, types) for a in RANDOM_ARMS}
    contrib = {}
    for t in types:
        c = -SIGNS_LATERAL[t] * dens[t] / 2.0
        nullT = np.array([drnd[a][t].mean() for a in RANDOM_ARMS])
        contrib[t] = dict(s=SIGNS_LATERAL[t], T=float(dens[t].mean()), c_mean=float(c.mean()), c_sd=float(c.std(ddof=1)),
                          c_n_syn=int((c > 0).sum()), c_n_anti=int((c < 0).sum()), null_sd_T=float(nullT.std(ddof=1)),
                          null_abs_median_T=float(np.median(np.abs(nullT))),
                          z=float(dens[t].mean() / nullT.std(ddof=1)) if nullT.std(ddof=1) > 0 else None,
                          direct=t in SIGNS_DIRECT)
    out["contributions"] = contrib
    absT = {t: abs(contrib[t]["T"]) for t in ("b3", "i1", "hg1", "hg4")}
    out["hg4_over_others"] = {t: absT["hg4"] / absT[t] for t in ("b3", "i1", "hg1")}

    # null-z variant (data-defined weights; not literature): S_z = - sum s_m z_m / 2
    zs = []
    for a in ["ensemble"] + RANDOM_ARMS:
        d = d_per_type(rates, a, seeds, types)
        zs.append(sum(-SIGNS_LATERAL[t] * (d[t].mean() / contrib[t]["null_sd_T"]) / 2.0
                      for t in types if contrib[t]["null_sd_T"] > 0))
    out["z_variant"] = dict(ensemble=float(zs[0]), random_n_syndirectional=int(sum(1 for z in zs[1:] if z > 0)),
                            random_n_anti=int(sum(1 for z in zs[1:] if z < 0)),
                            random_n_below_ensemble=int(sum(1 for z in zs[1:] if z < zs[0])))

    # dA per stimulus for ensemble and the baseline (static) asymmetry
    per_stim = {}
    for arm in ("ensemble", "bridge_off"):
        per_stim[arm] = {s: summarise(dA_stim(rates, arm, s, seeds, SIGNS_LATERAL)) for s in STIMS}
    out["dA_per_stimulus"] = per_stim
    rnd_static = np.array([np.mean(dA_stim(rates, a, "static", seeds, SIGNS_LATERAL)) for a in RANDOM_ARMS])
    out["static_bias"] = dict(ensemble=per_stim["ensemble"]["static"]["mean"],
                              random_quantiles=quantiles(rnd_static))

    # time resolution (ensemble, 25 ms bins; five primary types only, all with the same sign convention)
    ens_bins = {}
    for r in rec["replay"]:
        if r["arm"] == "ensemble" and "primary_bin25_hz" in r:
            ens_bins.setdefault(r["stimulus"], {})[r["seed"]] = r["primary_bin25_hz"]
    prim_signs = {t: SIGNS_LATERAL[t] for t in ("b3", "i1", "hg1", "hg4")}
    def series(stim, seed):
        b = ens_bins[stim][seed]
        return -sum(s * (np.array(b[f"{t}|right"], float) - np.array(b[f"{t}|left"], float)) for t, s in prim_signs.items())
    ser = np.array([(series("rot_right", s) - series("rot_left", s)) / 2.0 for s in seeds])   # seeds x bins
    snr = {}
    for nb in (1, 4):
        nbins = ser.shape[1] // nb
        agg = ser[:, :nbins * nb].reshape(len(seeds), nbins, nb).mean(2).ravel()
        snr[str(nb)] = dict(bin_ms=25 * nb, mean=float(agg.mean()), sd=float(agg.std(ddof=1)),
                            frac_syn=float((agg > 0).mean()), frac_anti=float((agg < 0).mean()),
                            snr=float(agg.mean() / agg.std(ddof=1)))
    out["bins"] = dict(n_bins=int(ser.shape[1]), types=sorted(prim_signs), per=snr)

    # gain candidates and predicted amplitude / yaw rate
    v1 = json.load(open(F0LAT))["turn_dn_absmax"]
    gc = g_candidates(SIGNS_LATERAL, const, list(v1.values()))
    gc["turn_dn_values"] = v1
    out["gain"] = gc
    deg_per_s_per_deg = const["yaw_rate_per_turn"] / (2.0 * const["turn_dphi_deg"]) * 180.0 / np.pi
    out["yaw_rate_per_dA_deg_per_s"] = float(deg_per_s_per_deg)
    drum = const["wavelength_deg"] * const["tf_hz"]
    out["drum_deg_per_s"] = float(drum)
    S_ens = sets[PRIMARY_SET]["arms"]["ensemble"]["mean"]
    out["predicted"] = {k: dict(G_deg_per_hz=gc[k], dA_deg=gc[k] * S_ens, yaw_deg_per_s=gc[k] * S_ens * deg_per_s_per_deg,
                                gain_vs_drum=gc[k] * S_ens * deg_per_s_per_deg / drum)
                        for k in ("G1_deg_per_hz", "G2_deg_per_hz")}

    # compute estimate (from the F3b-3 logs; per 5 ms interval)
    tm = timings()
    n_int = 300                                    # intervals of one F3b-3 replay (100 grey + 200 stimulus)
    per_int = dict(flyvis_s=float(np.median(tm["render_s_per_stimulus"])) / n_int,
                   brain_s=float(np.median(tm["brain_s_per_run"])) / n_int,
                   vnc_s=float(np.median(tm["vnc_s_per_arm_stimulus"])) / 20.0 / n_int)
    out["timing"] = dict(raw=tm, per_interval_s=per_int, per_interval_total_s=float(sum(per_int.values())))
    # design of the flight set (docs/08 section 6): 500 intervals per flight, 4 stimuli, 9 arms
    n_flight = (FLIGHT_LEAD_S + FLIGHT_STIM_S) / (INTERVAL_MS * 1e-3)
    t_full = per_int["flyvis_s"] + per_int["brain_s"] + per_int["vnc_s"]
    t_v1 = per_int["flyvis_s"] + per_int["brain_s"]                 # reference arm: no VNC
    comp = {}
    for n_seeds in (5, 10, 20):
        flights = N_ARMS * 4 * n_seeds
        wall = 4 * n_seeds * n_flight * ((N_ARMS - 1) * t_full + t_v1)
        comp[str(n_seeds)] = dict(flights=flights, wall_h=wall / 3600.0)
    out["compute"] = dict(intervals_per_flight=int(n_flight), seconds_per_flight_full=float(n_flight * t_full),
                          lead_s=FLIGHT_LEAD_S, stim_s=FLIGHT_STIM_S, n_arms=N_ARMS, per_n_seeds=comp,
                          vnc_peak_rss_gb=float(rec["peak_rss_gb"]),
                          brain_rss_gb=float(tm["brain_free_gb_start"] - tm["brain_free_gb_after_engine"]),
                          limit_h=5.0)
    return out


def fmt(x, n=2, sign=False):
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s.replace("-", "−") if False else s


def _cell(key):
    fam, rate, side = key.split("|")
    return f"{fam} at {rate} Hz, {'left' if side == 'L' else 'right'} drive"


def blocks(o):
    """Markdown blocks for docs/08 (name -> text)."""
    B = {}
    c = o["constants"]
    B["constants"] = "\n".join([
        "| constant | value | source |", "|---|---|---|",
        f"| amplitude asymmetry per unit turn_bias (each wing) | {c['turn_dphi_deg']:g} deg | `configs/body_flight.yaml` `turn_dphi_deg` (E) |",
        f"| steady yaw rate per unit turn_bias | {c['yaw_rate_per_turn']:g} rad/s | `yaw_rate_per_turn` (E) |",
        f"| yaw damping time constant | {c['yaw_damping_tau'] * 1000:g} ms | `yaw_damping_tau` (E) |",
        f"| turn_bias limit | {c['turn_bias_max']:g} | `configs/readouts.yaml` `turn_bias_max` (E) |",
        f"| v1 range of A_L - A_R | {o['gain']['v1_dA_max_deg']:g} deg | 2 x turn_dphi_deg x turn_bias_max |",
        f"| yaw rate per degree of A_L - A_R | {o['yaw_rate_per_dA_deg_per_s']:.1f} deg/s | yaw_rate_per_turn / (2 turn_dphi_deg) |",
        f"| grating wavelength / temporal frequency | {c['wavelength_deg']:g} deg / {c['tf_hz']:g} Hz | `tasks/t0_stimuli.py` |",
        f"| drum speed in the open-loop data (F3b) | {o['drum_deg_per_s']:g} deg/s | wavelength x frequency |",
    ])
    g = o["gain"]
    B["gain"] = "\n".join([
        "| candidate | rule | G (deg per Hz) | predicted A_L - A_R for `ensemble` (deg) | predicted steady yaw rate (deg/s) | ratio to the drum speed |",
        "|---|---|---|---|---|---|",
        f"| G1 | largest F3a-3 asymmetry ({_cell(g['max_cell']['key'])}, {abs(g['max_cell']['dA_per_G_hz']):.1f} Hz summed with the signs) = v1 range {g['v1_dA_max_deg']:g} deg | {g['G1_deg_per_hz']:.3f} | {o['predicted']['G1_deg_per_hz']['dA_deg']:+.2f} | {o['predicted']['G1_deg_per_hz']['yaw_deg_per_s']:+.1f} | {o['predicted']['G1_deg_per_hz']['gain_vs_drum']:+.2f} |",
        f"| G2 | F3a-3 DNa04 at 25 Hz ({g['dna04_25_dA_per_G_hz']:.1f} Hz) = typical v1 hover turn (median peak turn_dn {g['v1_turn_dn_median']:.2f} -> {g['v1_dA_typical_deg']:.1f} deg) | {g['G2_deg_per_hz']:.3f} | {o['predicted']['G2_deg_per_hz']['dA_deg']:+.2f} | {o['predicted']['G2_deg_per_hz']['yaw_deg_per_s']:+.1f} | {o['predicted']['G2_deg_per_hz']['gain_vs_drum']:+.2f} |",
    ])
    P = o["sets"][PRIMARY_SET]
    e = P["arms"]["ensemble"]
    ps = o["dA_per_stimulus"]
    rows = ["| stimulus | A_L - A_R (Hz units, G = 1), mean +- s.d. over seeds | bridge_off |", "|---|---|---|"]
    for s in STIMS:
        a, b = ps["ensemble"][s], ps["bridge_off"][s]
        rows.append(f"| {s} | {a['mean']:+.2f} +- {a['sd']:.2f} | {b['mean']:+.2f} +- {b['sd']:.2f} |")
    B["per_stimulus"] = "\n".join(rows)
    B["ensemble"] = (
        f"S(`ensemble`) = {e['mean']:+.2f} Hz (s.d. over seeds {e['sd']:.2f}; positive in {e['n_pos']} of {e['n']} seeds, "
        f"negative in {e['n_neg']}; one-sample t-test p = {e['p_t']:.3g}, descriptive). "
        f"S(`sign_flip`) = {P['arms']['sign_flip']['mean']:+.2f} Hz by construction; S(`bridge_off`) = {P['arms']['bridge_off']['mean']:+.2f} Hz.")
    rows = ["| MN type | s_m | basis | T (Hz) | contribution to S (Hz) | seeds syn / anti | T relative to the null s.d. (z) |",
            "|---|---|---|---|---|---|---|"]
    for t, v in o["contributions"].items():
        rows.append(f"| {t} | {v['s']:+d} | {'direct' if v['direct'] else 'lateralisation'} | {v['T']:+.2f} | {v['c_mean']:+.2f} | "
                    f"{v['c_n_syn']} / {v['c_n_anti']} | {v['z']:+.2f} |" if v["z"] is not None else
                    f"| {t} | {v['s']:+d} | {'direct' if v['direct'] else 'lateralisation'} | {v['T']:+.2f} | {v['c_mean']:+.2f} | {v['c_n_syn']} / {v['c_n_anti']} | - |")
    B["contrib"] = "\n".join(rows)
    h = o["hg4_over_others"]
    B["hg4"] = (f"Ratio of the absolute T of hg4 to that of b3, i1 and hg1: {h['b3']:.1f}, {h['i1']:.1f}, {h['hg1']:.1f} (F3b-3 values).")
    rows = ["| mapping set | types (sign) | S(`ensemble`) Hz | offset B (Hz) | seeds syn / anti | seeds for 80 % power (alpha 0.05) | random wirings syn / anti (of 39) | random significantly syn / anti (p < 0.05) | random wirings below `ensemble` |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, signs in SETS.items():
        r = o["sets"][name]
        a = r["arms"]["ensemble"]
        t = ", ".join(f"{k}({'+' if v > 0 else '-'})" for k, v in signs.items())
        rr = r["random"]
        rows.append(f"| {name} | {t} | {a['mean']:+.2f} | {r['ensemble_offset']['mean']:+.2f} | {a['n_pos']} / {a['n_neg']} | {r['n_seeds_80pct_power']:.0f} | {rr['n_syndirectional']} / {rr['n_anti']} | "
                    f"{rr['n_sig_syn']} / {rr['n_sig_anti']} | {rr['n_random_below_ensemble']} |")
    z = o["z_variant"]
    rows.append(f"| lateral, null-z weights (data-defined, not literature) | as lateral | {z['ensemble']:+.2f} (z units) | - | - | - | "
                f"{z['random_n_syndirectional']} / {z['random_n_anti']} | - | {z['random_n_below_ensemble']} |")
    B["variants"] = "\n".join(rows)
    rr = P["random"]
    q = rr["quantiles"]
    B["random"] = (f"Over the 39 random wirings (set `{PRIMARY_SET}`): S has quantiles min {q[0]:+.2f}, 5 % {q[1]:+.2f}, 25 % {q[2]:+.2f}, "
                   f"median {q[3]:+.2f}, 75 % {q[4]:+.2f}, 95 % {q[5]:+.2f}, max {q[6]:+.2f} Hz. "
                   f"{rr['n_syndirectional']} of 39 are syndirectional ({rr['n_sig_syn']} significantly, p < 0.05, descriptive), "
                   f"{rr['n_anti']} anti-directional ({rr['n_sig_anti']} significantly) and {rr['n_zero']} exactly zero. "
                   f"{rr['n_random_below_ensemble']} of 39 random wirings have S below `ensemble`.")
    sb = o["static_bias"]["random_quantiles"]
    B["bias"] = (f"Under the static grating `ensemble` has A_L - A_R = {o['static_bias']['ensemble']:+.2f} Hz units (a constant turn bias "
                 f"before any drum rotation); random wirings: median {sb[3]:+.2f}, 5-95 % {sb[1]:+.2f} to {sb[5]:+.2f}.")
    bn = o["bins"]["per"]
    B["snr"] = "\n".join([
        "| averaging bin | mean S (Hz) | s.d. over seeds x bins (Hz) | mean / s.d. | bins with S > 0 | bins with S < 0 |", "|---|---|---|---|---|---|"] + [
        f"| {v['bin_ms']} ms | {v['mean']:+.2f} | {v['sd']:.2f} | {v['snr']:+.2f} | {v['frac_syn']:.2f} | {v['frac_anti']:.2f} |"
        for v in bn.values()])
    t = o["timing"]
    pi = t["per_interval_s"]
    B["timing"] = "\n".join([
        "| stage | measured in F3b-3 | per 5 ms interval |", "|---|---|---|",
        f"| FlyVis + boundary transduction | {np.median(t['raw']['render_s_per_stimulus']):.0f} s per 300 intervals | {pi['flyvis_s'] * 1000:.1f} ms |",
        f"| FlyWire brain | {np.median(t['raw']['brain_s_per_run']):.0f} s per 300 intervals | {pi['brain_s'] * 1000:.1f} ms |",
        f"| BANC VNC subgraph | {np.median(t['raw']['vnc_s_per_arm_stimulus']):.0f} s per 20 replays of 300 intervals | {pi['vnc_s'] * 1000:.1f} ms |",
        f"| total | | {t['per_interval_total_s'] * 1000:.1f} ms |",
        f"| one-off engine build | brain {t['raw']['brain_engine_build_s']} s, VNC {t['raw']['vnc_engine_build_s']} s | |",
    ])
    c = o["compute"]
    rows = ["| seeds per cell | flights (9 arms x 4 stimuli) | wall time (h, serial) | over the 5 h limit |", "|---|---|---|---|"]
    for n, v in c["per_n_seeds"].items():
        rows.append(f"| {n} | {v['flights']} | {v['wall_h']:.1f} | {'yes' if v['wall_h'] > c['limit_h'] else 'no'} |")
    B["compute"] = "\n".join(rows) + (
        f"\n\nOne flight = {c['lead_s']:g} s grey + {c['stim_s']:g} s drum = {c['intervals_per_flight']} intervals = "
        f"{c['seconds_per_flight_full']:.0f} s of wall time with all three engines. RAM: FlyVis + brain engine about "
        f"{c['brain_rss_gb']:.1f} GB (free memory before / after the build in the F3b-3 log), VNC engine peak "
        f"{c['vnc_peak_rss_gb']:.1f} GB (F3b-3 record), so one flight needs about {c['brain_rss_gb'] + c['vnc_peak_rss_gb']:.1f} GB "
        "and flights run serially.")
    return B


def inject(doc, B):
    text = doc.read_text()
    for k, v in B.items():
        pat = re.compile(rf"(<!-- GEN:{k} -->).*?(<!-- /GEN:{k} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"marker GEN:{k} missing in {doc}")
        text = pat.sub(lambda m: m.group(1) + "\n" + v + "\n" + m.group(2), text)
    doc.write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if OUT_JSON.exists() and not a.overwrite:
        # the JSON is regenerated deterministically; refuse silent overwrite of a different result
        old = json.loads(OUT_JSON.read_text())
        new = build()
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            raise SystemExit(f"{OUT_JSON} exists and differs; use --overwrite")
        out = new
    else:
        out = build()
        OUT_JSON.write_text(json.dumps(out, indent=1))
    if DOC.exists():
        inject(DOC, blocks(out))
    print("selfcheck T == F3b-3:", out["selfcheck_T_equals_f3b3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
