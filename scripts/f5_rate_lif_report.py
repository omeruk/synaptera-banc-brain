"""Report of F5 step 2a (prereg/15): the untrained rate VNC against the LIF VNC on the F3a-3 protocol.

    env -u PYTHONPATH python scripts/f5_rate_lif_report.py            # writes docs/results/f5_rate_lif.{json,md}

Inputs: `runs/f5/rate_lif/record.{npz,json}` (rate model, `scripts/f5_rate_lif_run.py`) and the committed F3a-3 LIF
results `docs/results/f3a3_vnc_subgraph.json`. Refuses to overwrite without --overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "runs" / "f5" / "rate_lif"
LIF = ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json"
OUT_JSON = ROOT / "docs" / "results" / "f5_rate_lif.json"
OUT_MD = ROOT / "docs" / "results" / "f5_rate_lif.md"

C1_TYPES = ("DLM1-4", "DVM1a-c", "tp2", "ps1")
LATERAL = ("b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4")
FAMILIES = ("DNa04", "DNa10", "DNa15")
RATES = (10, 25, 50, 100)
NONZERO = 1e-6
R_OUT = 100.0


def load():
    z = np.load(REC / "record.npz")
    rec = json.loads((REC / "record.json").read_text())
    return z, rec


def build():
    z, rec = load()
    lif = json.loads(LIF.read_text())
    keys = [str(k) for k in z["keys"]]
    conds = [str(c) for c in z["conditions"]]
    nd = rec["n_intervals"]["drive"]
    mn = z["mn_state"].astype(np.float64)
    resp = {c: {k: float(mn[i, :nd, j].mean()) for j, k in enumerate(keys)} for i, c in enumerate(conds)}

    def s(cond, t, side):
        return resp[cond].get(f"{t}|{side}")

    out = dict(prereg=rec["prereg"], arm=rec["arm"], record_sha256={
        f: hashlib.sha256((REC / f).read_bytes()).hexdigest() for f in ("record.npz", "record.json")},
        lif_source=str(LIF.relative_to(ROOT)), subgraph=rec["subgraph"], engine=rec["engine"],
        n_dn_inputs=rec["n_dn_inputs"], n_intervals=rec["n_intervals"], dn_groups=rec["dn_groups"],
        engine_checks=rec["engine_checks"], run_wall_s=rec["run_wall_s"], total_wall_s=rec["total_wall_s"],
        peak_rss_gb=rec["peak_rss_gb"], threads=rec["threads"],
        symmetry={t: rec["symmetry"][t] for t in LATERAL + C1_TYPES if t in rec["symmetry"]})
    # C1
    c1 = []
    for t in C1_TYPES:
        for side in ("left", "right"):
            v = s("R1:G100", t, side)
            c1.append(dict(type=t, side=side, state=v, lif_hz=lif["r1"]["sub"][f"G100|{t}|{side}"], agrees=v > 0))
    out["c1"] = dict(items=c1, n_agree=sum(x["agrees"] for x in c1), n=len(c1), agrees=all(x["agrees"] for x in c1))
    # steering effects d for every family / rate / side / gate-passing type
    gate_types = lif["r3"]["gate_passing"]

    def d(fam, side, rate, t):
        c = f"R3:{fam}_{side}_{rate}"
        return (s(c, t, "right") - s(c, t, "left")) - (s("R3:B", t, "right") - s("R3:B", t, "left"))

    lif_tests = {(x["family"], x["rate"], x["mn"], x["side"]): x for x in lif["r3"]["tests"]}
    steer = []
    for fam in FAMILIES:
        for rate in RATES:
            for t in gate_types:
                for side in ("L", "R"):
                    lt = lif_tests[(fam, rate, t, side)]
                    v = d(fam, side, rate, t)
                    steer.append(dict(family=fam, rate=rate, mn=t, side=side, d=v, lif_mean=lt["mean"],
                                      lif_steers=lt["p_t_holm"] < 0.05,
                                      ipsilateral=(v < -NONZERO) if side == "L" else (v > NONZERO)))
    # C2
    c2 = [dict(family=x["family"], rate=x["rate"], mn=x["mn"], side=x["side"], d=x["d"], lif_mean=x["lif_mean"],
               agrees=bool(abs(x["d"]) >= NONZERO and np.sign(x["d"]) == np.sign(x["lif_mean"]))) for x in steer if x["lif_steers"]]
    out["c2"] = dict(items=c2, n_agree=sum(x["agrees"] for x in c2), n=len(c2), agrees=all(x["agrees"] for x in c2),
                     per_family={f: dict(n=sum(1 for x in c2 if x["family"] == f),
                                         n_agree=sum(1 for x in c2 if x["family"] == f and x["agrees"])) for f in FAMILIES})
    out["agreement"] = bool(out["c1"]["agrees"] and out["c2"]["agrees"])
    # descriptive
    out["dose"] = {f"{t}|{side}": {g: s(f"R1:{g}", t, side) for g in ("G20", "G50", "G100")}
                   for t in C1_TYPES for side in ("left", "right")}
    out["dose_monotone"] = {k: bool(v["G20"] <= v["G50"] <= v["G100"]) for k, v in out["dose"].items()}
    r1 = {}
    for k in keys:
        r1[k] = {g: dict(state=s(f"R1:{g}", *k.split("|")), lif_hz=lif["r1"]["sub"].get(f"{g}|{k}"))
                 for g in ("G20", "G50", "G100", "GL100", "GR100")}
    out["r1"] = r1
    pairs = [(v["G100"]["state"], v["G100"]["lif_hz"]) for v in r1.values() if v["G100"]["lif_hz"] is not None]
    rho = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    out["spearman_g100"] = dict(n=len(pairs), rho=float(rho.statistic), p=float(rho.pvalue))
    out["steering"] = steer
    nz = [x for x in steer if x["lif_mean"] != 0 and abs(x["d"]) >= NONZERO]
    out["steering_sign_vs_lif_nonzero"] = dict(n=len(nz), n_same=sum(1 for x in nz if np.sign(x["d"]) == np.sign(x["lif_mean"])))
    lif_steer_types = {f: sorted({x["mn"] for x in steer if x["family"] == f and x["lif_steers"]}) for f in FAMILIES}
    out["lif_steer_types"] = lif_steer_types
    low = {}
    for f in FAMILIES:
        lo = None
        for rate in RATES:
            if any(x["ipsilateral"] for x in steer if x["family"] == f and x["rate"] == rate and x["mn"] in lif_steer_types[f]):
                lo = rate
                break
        low[f] = dict(rate_model=lo, lif=min((x["rate"] for x in steer if x["family"] == f and x["lif_steers"]), default=None))
    out["lowest_rate"] = low
    out["off_target_100"] = {f: sorted({x["mn"] for x in steer if x["family"] == f and x["rate"] == 100
                                        and abs(x["d"]) >= NONZERO and x["mn"] not in lif_steer_types[f]}) for f in FAMILIES}
    ib = conds.index("R3:B")
    na = z["net_abs"].astype(np.float64)[ib]
    nn = z["net_abs_nondriven"].astype(np.float64)[ib]
    last_drive, last_cut = slice(nd - 20, nd), slice(len(na) - 20, len(na))
    out["decay"] = dict(drive_last_100ms=float(na[last_drive].mean()), cut_last_100ms=float(na[last_cut].mean()),
                        ratio=float(na[last_cut].mean() / na[last_drive].mean()),
                        nondriven_ratio=float(nn[last_cut].mean() / nn[last_drive].mean()),
                        lif_ratio_mean=float(np.mean([a / b for a, b in zip(lif["r2"]["A"], lif["r2"]["B"])])))
    return out


def f(x, nd=4):
    return f"{x:+.{nd}f}"


def render(o):
    L = ["# F5 step 2a: the untrained rate VNC against the LIF VNC on the F3a-3 protocol (prereg/15)", ""]
    L.append("Generated by `scripts/f5_rate_lif_run.py` and `scripts/f5_rate_lif_report.py`; rate-model record "
             "`runs/f5/rate_lif/` (not committed); LIF reference `docs/results/f3a3_vnc_subgraph.json` (F3a-3, "
             "reference arm R, seed means). Criteria and protocol are those of the committed prereg/15. The rate model "
             "(`engine/rate_torch.py`, prereg/14 section 2) is at initialisation (the `untrained` arm: gain 1, leak "
             f"{o['engine']['lam_0']}), float32, deterministic; no constant was fitted or changed.")
    L.append("")
    e = o["engine"]
    L.append("## 1. Model and run")
    L.append("")
    L.append(f"- Graph: the prereg/08 VNC subgraph, {e['n_neurons']:,} neurons, {e['n_edges']:,} edges, "
             f"{o['n_dn_inputs']:,} DNs as clamped inputs; trainable parameters {e['n_trainable']:,} "
             f"(graph SHA-256 `{e['graph_sha256'][:16]}...`).")
    L.append(f"- Protocol: {o['n_intervals']['drive']} drive + {o['n_intervals']['cut']} cut intervals of 5 ms; a "
             f"driven DN is clamped to rate / {e['r_in_hz']:.0f} Hz; 30 conditions (5 R1 + 25 R3) in one batch.")
    L.append(f"- Wall time of the 30 conditions {o['run_wall_s']:.1f} s ({o['threads']} threads); whole script "
             f"{o['total_wall_s']:.0f} s; peak RAM {o['peak_rss_gb']:.2f} GB.")
    L.append("- Response = mean state over the 200 drive intervals and the neurons of the type and side; "
             "Hz-equivalent = 100 Hz x state. The state is signed (negative = below rest).")
    L.append("")
    L.append("## 2. Engine checks on a 2,000-neuron subgraph of the real VNC (prereg/14 gate 3)")
    L.append("")
    c = o["engine_checks"]
    L.append(f"Subgraph: {c['check_subgraph_rule']}; {c['n_neurons']:,} neurons, {c['n_edges']:,} edges, "
             f"{c['n_inputs']} inputs; gains and leaks drawn from a normal (seed {c['seed']}).")
    L.append("")
    L.append("| check | value | threshold | pass |")
    L.append("|---|---|---|---|")
    fw, gr = c["forward"], c["gradient"]
    L.append(f"| forward vs NumPy, float32, {fw['intervals']} intervals: max abs diff | {fw['max_abs_diff']:.2e} | "
             f"{fw['threshold']:.0e} | {'pass' if fw['passed'] else 'FAIL'} |")
    L.append(f"| sparse backward vs dense float64, {gr['intervals']} intervals: rel. error theta / phi | "
             f"{gr['rel_err_theta']:.2e} / {gr['rel_err_phi']:.2e} | {gr['threshold']:.0e} | "
             f"{'pass' if gr['passed'] else 'FAIL'} |")
    L.append("")
    L.append("Causality and resume are unit tests on synthetic graphs (`tests/test_rate_torch.py`).")
    L.append("")
    L.append("## 3. Symmetry gate of the readout and power types (input synapses, |log2(R/L)| < 0.5)")
    L.append("")
    L.append("| type | log2 total | pass |")
    L.append("|---|---|---|")
    for t, v in o["symmetry"].items():
        L.append(f"| {t} | {v['log2_total']:+.2f} | {'pass' if v['passed'] else 'FAIL'} |")
    L.append("")
    L.append("## 4. C1: power recruitment under G100 (DNg02 both sides, 100 Hz)")
    L.append("")
    L.append("| type | side | rate-model state | Hz-equivalent | LIF (Hz) | agrees |")
    L.append("|---|---|---|---|---|---|")
    for x in o["c1"]["items"]:
        L.append(f"| {x['type']} | {x['side']} | {f(x['state'])} | {R_OUT * x['state']:+.2f} | {x['lif_hz']:.1f} | "
                 f"{'yes' if x['agrees'] else 'NO'} |")
    L.append("")
    L.append(f"**C1: {o['c1']['n_agree']} / {o['c1']['n']} agree; {'agreement' if o['c1']['agrees'] else 'MISMATCH'}.**")
    L.append("")
    L.append("## 5. C2: steering sign on every LIF-steering test (Holm p < 0.05)")
    L.append("")
    L.append("d = (s_R - s_L) under steering minus under baseline B (state units; LIF in Hz). Ipsilateral: d < 0 for "
             "a left drive, d > 0 for a right drive.")
    L.append("")
    L.append("| family | rate (Hz) | MN type | drive side | rate-model d | LIF mean d (Hz) | agrees |")
    L.append("|---|---|---|---|---|---|---|")
    for x in o["c2"]["items"]:
        L.append(f"| {x['family']} | {x['rate']} | {x['mn']} | {x['side']} | {x['d']:+.2e} | {x['lif_mean']:+.2f} | "
                 f"{'yes' if x['agrees'] else 'NO'} |")
    L.append("")
    pf = o["c2"]["per_family"]
    L.append(f"**C2: {o['c2']['n_agree']} / {o['c2']['n']} agree (" +
             ", ".join(f"{k} {v['n_agree']} / {v['n']}" for k, v in pf.items()) +
             f"); {'agreement' if o['c2']['agrees'] else 'MISMATCH'}.**")
    L.append("")
    L.append(f"**Overall (prereg/15 section 3): {'qualitative agreement' if o['agreement'] else 'mismatch'}.**")
    L.append("")
    L.append("## 6. Descriptive (prereg/15 section 4; no criterion)")
    L.append("")
    L.append("### 6.1 Dose order of the C1 types (state)")
    L.append("")
    L.append("| type and side | G20 | G50 | G100 | monotone |")
    L.append("|---|---|---|---|---|")
    for k, v in o["dose"].items():
        L.append(f"| {k} | {f(v['G20'])} | {f(v['G50'])} | {f(v['G100'])} | {'yes' if o['dose_monotone'][k] else 'no'} |")
    L.append("")
    L.append("### 6.2 R1: every readout type (rate-model state; LIF Hz in brackets)")
    L.append("")
    L.append("| type and side | G20 | G50 | G100 | GL100 | GR100 |")
    L.append("|---|---|---|---|---|---|")
    for k, v in o["r1"].items():
        cells = []
        for g in ("G20", "G50", "G100", "GL100", "GR100"):
            lh = v[g]["lif_hz"]
            cells.append(f"{f(v[g]['state'])} ({'-' if lh is None else f'{lh:.1f}'})")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    sp = o["spearman_g100"]
    L.append(f"Spearman correlation across type-sides, rate-model G100 state against LIF G100 rate: rho = "
             f"{sp['rho']:.3f} (n = {sp['n']}, p = {sp['p']:.2g}).")
    L.append("")
    L.append("### 6.3 Steering beyond C2")
    L.append("")
    sv = o["steering_sign_vs_lif_nonzero"]
    L.append(f"- Sign agreement with the LIF mean d over every family / rate / side / gate-passing type where the LIF "
             f"mean d is non-zero and |rate-model d| >= 1e-6: {sv['n_same']} / {sv['n']}.")
    for fam, v in o["lowest_rate"].items():
        L.append(f"- {fam}: lowest rate with an ipsilateral rate-model d on a LIF-steering type "
                 f"({', '.join(o['lif_steer_types'][fam])}): {v['rate_model']} Hz; LIF lowest steering rate "
                 f"{v['lif']} Hz. Off-target types with |d| >= 1e-6 at 100 Hz: "
                 f"{', '.join(o['off_target_100'][fam]) or '-'}.")
    L.append("")
    L.append("Rate-model d at 100 Hz, every gate-passing type (state units, x 1e3; L drive / R drive; LIF Hz in "
             "brackets):")
    L.append("")
    L.append("| MN type | " + " | ".join(FAMILIES) + " |")
    L.append("|---|" + "---|" * len(FAMILIES))
    by = {(x["family"], x["rate"], x["mn"], x["side"]): x for x in o["steering"]}
    for t in sorted({x["mn"] for x in o["steering"]}, key=lambda m: [x["mn"] for x in o["steering"]].index(m)):
        cells = []
        for fam in FAMILIES:
            a, b = by[(fam, 100, t, "L")], by[(fam, 100, t, "R")]
            cells.append(f"{1e3 * a['d']:+.3f} / {1e3 * b['d']:+.3f} ({a['lif_mean']:+.1f} / {b['lif_mean']:+.1f})")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    dc = o["decay"]
    L.append("### 6.4 Decay after the cut (baseline B)")
    L.append("")
    L.append(f"Network mean |s|: last 100 ms of the drive {dc['drive_last_100ms']:.3e}, last 100 ms of the 200 ms cut "
             f"{dc['cut_last_100ms']:.3e}, ratio {dc['ratio']:.3f} (non-driven neurons {dc['nondriven_ratio']:.3f}). "
             f"LIF R2 (1 s cut), mean A / B over seeds: {dc['lif_ratio_mean']:.3f}.")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("report exists (use --overwrite)")
    o = build()
    OUT_JSON.write_text(json.dumps(o, indent=1))
    OUT_MD.write_text(render(json.loads(OUT_JSON.read_text())))
    print(f"C1 {o['c1']['n_agree']}/{o['c1']['n']}, C2 {o['c2']['n_agree']}/{o['c2']['n']}, agreement {o['agreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
