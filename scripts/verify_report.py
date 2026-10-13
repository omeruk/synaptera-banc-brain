"""Recompute the numbers of the committed result reports from their raw records and check them against the text
(CLAUDE.md rule 11). Independent of the report scripts: the F3a tables are recomputed here from `runs/f3a/f3a_v3.json`
and each formatted row must appear verbatim in `docs/results/f3a_banc_vnc.md`; the crosswalk summary numbers are
checked against `docs/results/f3_dn_crosswalk.json` and its text; F3a-2 (`docs/results/f3a2_checks.md`): C1 rates
recomputed from the F0 spike records, gates / C2 / C3 statistics from `runs/f3a2/f3a2_v3.json`; F3a-3 (`docs/results/f3a3_vnc_subgraph.md`): subgraph counts against the
committed JSON, the prereg and the report, gates / R1 / R2 / R3 recomputed from `runs/f3a3/f3a3_v3.json`; F3b-1 (`docs/results/f3b1_optomotor.md`): gains against
the prereg and the registry, FlyWire DN rates and brain-side tests from the brain spike records, gates and primary tests
per arm from `runs/f3b1/f3b1_v3.json`; F3b-2 and F3b-3 (`docs/results/f3b2_dn_ensemble.md`, `f3b3_specificity.md`): see
`check_f3b2` and `check_f3b3`.

    env -u PYTHONPATH python scripts/verify_report.py            # exit 1 on a mismatch
    env -u PYTHONPATH python scripts/verify_report.py --print    # only print the recomputed strings
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F3A_REC = ROOT / "runs" / "f3a" / "f3a_v3.json"
F3A_MD = ROOT / "docs" / "results" / "f3a_banc_vnc.md"
XW_JSON = ROOT / "docs" / "results" / "f3_dn_crosswalk.json"
XW_MD = ROOT / "docs" / "results" / "f3_dn_crosswalk.md"
F3A2_REC = ROOT / "runs" / "f3a2" / "f3a2_v3.json"
F3A2_MD = ROOT / "docs" / "results" / "f3a2_checks.md"


def f3a_expected():
    d = json.loads(F3A_REC.read_text())
    runs = d["runs"]
    out = {}
    out["graph"] = f"{d['n_neurons']:,} neurons, {d['pairs']:,} pairs, {d['synapses']:,} synapses"
    out["ram_wall"] = f"peak RAM {d['peak_rss_gb']:.2f} GB, wall {d['total_wall_s']:.0f} s"
    # drive table rows
    for drive in d["drives"]:
        r = [x for x in runs if x["drive"] == drive]
        drv = np.mean([x["driven_dn_rate_hz"] for x in r if x.get("driven_dn_rate_hz") is not None]) if drive != "D0" else None
        cells = [f"{drv:.1f}" if drv is not None else "-"] + [
            f"{np.mean([x[k] for x in r]):.4f}" for k in ("network_rate_hz", "nondriven_rate_hz", "cut_mean_hz", "cut_last_step_hz")]
        out[f"drive:{drive}"] = f"| {drive} | " + " | ".join(cells) + " |"
    # MN per-type L/R cells, per drive (mean over seeds)
    for t in d["types"]:
        for drive in d["drives"]:
            if drive == "D0":
                continue
            r = [x for x in runs if x["drive"] == drive]
            L = np.mean([x["rate_hz"][f"{t}|left"] for x in r]) if f"{t}|left" in r[0]["rate_hz"] else None
            R = np.mean([x["rate_hz"][f"{t}|right"] for x in r]) if f"{t}|right" in r[0]["rate_hz"] else None
            out[f"mn:{t}:{drive}"] = (f"{L:.1f}" if L is not None else "-") + " / " + (f"{R:.1f}" if R is not None else "-")
    # gate 2
    g2 = d["gate2_runs"]
    out["gate2"] = f"{sum(x['ignited'] for x in g2)} / {len(g2)}"
    out["gate2_last"] = f"{np.mean([x['last_cut_step_hz'] for x in g2]):.3f} Hz"
    out["gate1"] = None
    return out, d


def _row(md, *cells):
    """The table row of md that starts with the given leading cells, split into cells (None if absent)."""
    head = "| " + " | ".join(cells) + " |"
    ln = next((x for x in md.splitlines() if x.startswith(head)), None)
    return None if ln is None else [c.strip() for c in ln.strip("|").split("|")]


def check_f3a2():
    """F3a-2 (prereg/07). C1 rates recomputed from the F0 spike records themselves (not from c1_rates.json); gates,
    C2 and C3 statistics recomputed from runs/f3a2/f3a2_v3.json."""
    import h5py
    from scipy.stats import ttest_1samp
    md = F3A2_MD.read_text()
    bad, n = [], 0

    def chk(key, ok, v):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    # C1: per-neuron F0 rates per family and window
    c1 = json.loads((ROOT / "runs" / "f3a2" / "c1_rates.json").read_text())
    neu = c1["neurons"]
    idx = np.array([x["index"] for x in neu])
    acc = {w: [] for w in ("perch", "flight", "landing")}
    for s in range(5):
        with h5py.File(ROOT / "runs" / "f0" / f"f0_5ms_seed{s}_data.h5", "r") as f:
            i, t, ph = f["spikes/i"][:], f["spikes/t"][:].astype(float), f["behavior/phase"][:]
        win = {"perch": (0.0, 0.25)}
        for w, codes in (("flight", (1, 2, 3, 9)), ("landing", (4, 8))):
            ks = np.flatnonzero(np.isin(ph, codes))
            assert np.all(np.diff(ks) == 1), w          # one contiguous span in these records
            win[w] = (0.45 + ks[0] * 0.005, 0.45 + (ks[-1] + 1) * 0.005)
        for w, (a, b) in win.items():
            m = (t >= a) & (t < b)
            cnt = np.array([np.count_nonzero(i[m] == k) for k in idx])
            acc[w].append(cnt / (b - a))
    for fam in c1["families"]:
        kl = [k for k, x in enumerate(neu) if x["family"] == fam and x["side"] == "left"]
        kr = [k for k, x in enumerate(neu) if x["family"] == fam and x["side"] == "right"]
        for w in acc:
            r = np.mean(acc[w], axis=0)
            row = _row(md, fam, w)
            exp = [f"{len(kl)}/{len(kr)}", ", ".join(f"{v:.1f}" for v in r[kl]), ", ".join(f"{v:.1f}" for v in r[kr]),
                   f"{np.median(r[kl + kr]):.1f}"]
            chk(f"c1:{fam}:{w}", row is not None and row[2:6] == exp, exp)
    # gates
    d = json.loads(F3A2_REC.read_text())
    g2 = d["gate2_runs"]
    chk("gate1", f"| 1: no input, seeds 0-4 (spikes) | {sum(r['spikes'] for r in d['gate1'])} |" in md, "gate1")
    chk("gate2", f"| {sum(r['ignited'] for r in g2)} / {len(g2)} (95 % CI" in md, "gate2")
    nets = [np.mean([r["network_rate_hz"] for r in d["c2"]["runs"]]), np.mean([r["network_rate_hz"] for r in g2])]
    nets += [np.mean([x["network_rate_hz"] for x in d["c3"] if x["condition"] == c]) for c in d["c3_conditions"]]
    chk("gate3", f"max {max(nets):.3f} Hz" in md, f"max {max(nets):.3f} Hz")
    # C2: last cut step per seed
    v = ", ".join(f"{r['network_step_hz'][-1]:.4f}" for r in d["c2"]["runs"])
    chk("c2:last", v in md, v)
    chk("c2:carriers", f"during the cut in any seed): {len(d['c2']['carriers'])} neurons" in md, len(d["c2"]["carriers"]))
    # C3: mean d and t-test p per reported row
    seeds = d["seeds"]["run"]
    by = {(x["condition"], x["seed"]): x["rate_hz"] for x in d["c3"]}

    def D(c, m):
        return np.array([by[(c, s)][f"{m}|right"] - by[(c, s)][f"{m}|left"] for s in seeds])

    passed = [m for m in d["steer_mn"] if d["symmetry"].get(m, {}).get("passed")]
    fams = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in d["c3_conditions"] if c != "B"))
    for fam in fams:
        for m in passed:
            dl, dr = D(f"{fam}_L", m) - D("B", m), D(f"{fam}_R", m) - D("B", m)
            row = _row(md, fam, m)
            if not dl.any() and not dr.any():
                chk(f"c3:{fam}:{m}:omitted", row is None, "row should be omitted")
                continue
            p = []
            for x in (dl, dr):
                p.append(1.0 if not x.any() else 0.0 if np.std(x) == 0 else float(ttest_1samp(x, 0.0).pvalue))
            exp = [f"{dl.mean():+.2f}", f"{p[0]:.2g}", f"{dr.mean():+.2f}", f"{p[1]:.2g}"]
            got = None if row is None or len(row) < 10 else [row[2], row[3], row[6], row[7]]
            chk(f"c3:{fam}:{m}", got == exp, exp)
    return n, bad


F3A3_REC = ROOT / "runs" / "f3a3" / "f3a3_v3.json"
F3A3_MD = ROOT / "docs" / "results" / "f3a3_vnc_subgraph.md"
F3A3_SUB = ROOT / "docs" / "results" / "f3a3_subgraph.json"
PREREG08 = ROOT / "prereg" / "08_f3a3_vnc_subgraph.md"


def _holm(p):
    """Holm step-down adjusted p values (independent of the report script)."""
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, k in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[k]))
        adj[k] = run
    return adj


def check_f3a3():
    """F3a-3 (prereg/08). Subgraph counts: committed JSON = run record, and the figures of the prereg and the report;
    gates, R1 cells, R2 per-seed decay, R3 tests and decisions recomputed from runs/f3a3/f3a3_v3.json."""
    from scipy.stats import ttest_1samp
    md, pre = F3A3_MD.read_text(), " ".join(PREREG08.read_text().split())   # prereg: line wraps ignored
    d = json.loads(F3A3_REC.read_text())
    sg = json.loads(F3A3_SUB.read_text())
    bad, n = [], 0

    def chk(key, ok, v):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    chk("sub:record", all(sg[k] == d["subgraph"][k] for k in d["subgraph"]), "run subgraph != committed json")
    chk("sub:check", sg["check_sub"] == {k: sg["kept"][k] for k in ("neurons", "pairs", "synapses")}, sg["check_sub"])
    k, w, a, r = sg["kept"], sg["whole"], sg["ascending"], sg["removed"]
    frags = [f"kept {k['neurons']:,} of {w['neurons']:,} neurons, {k['pairs']:,} of {w['pairs']:,} pairs, "
             f"{k['synapses']:,} of {w['synapses']:,} synapses",
             "By reason: region {region:,}, motor {motor:,}, descending {descending:,}, ascending {ascending:,}"
             .format(**sg["kept_by_reason"]),
             f"({r['edges_onto_dn_from_kept']['pairs']:,} pairs, {r['edges_onto_dn_from_kept']['synapses']:,} synapses)",
             f"Ascending: {a['neurons']:,} neurons ({a['by_region']['VNC']:,} VNC, {a['by_region']['SEZ']:,} SEZ, "
             f"{a['by_region']['brain']:,} brain, {a['by_region']['NA']:,} without region)",
             f"inputs kept {a['inputs_kept']['synapses']:,} synapses, inputs removed {a['inputs_removed']['synapses']:,}",
             f"outputs kept {a['outputs_kept']['synapses']:,}, outputs removed (to the BANC brain/SEZ) "
             f"{a['outputs_removed']['synapses']:,}"]
    chk("prereg08:neck0", sg["kept_by_region"]["neck"] == 0 and "(0)" in pre, "neck count")
    for i, fr in enumerate(frags):
        chk(f"prereg08:{i}", fr in pre, fr)
    chk("report:kept", f"Kept: {k['neurons']:,} neurons, {k['pairs']:,} pairs, {k['synapses']:,} synapses." in md, k)
    # gates
    g2 = d["gate2_runs"]
    chk("gate1", f"(spikes) | {sum(x['spikes'] for x in d['gate1'])} |" in md, "gate1")
    chk("gate2", f"| {sum(x['ignited'] for x in g2)} / {len(g2)} (95 % CI" in md, "gate2")
    nets = [np.mean([x["network_rate_hz"] for x in d["r1"] if x["condition"] == c]) for c in d["r1_drives"]]
    nets += [np.mean([x["network_rate_hz"] for x in d["r2"]["runs"]]), np.mean([x["network_rate_hz"] for x in g2])]
    nets += [np.mean([x["network_rate_hz"] for x in d["r3"] if x["condition"] == c]) for c in d["r3_conditions"]]
    chk("gate3", f"max {max(nets):.4f} Hz" in md, f"max {max(nets):.4f} Hz")
    # R1: every reported cell, subgraph and whole graph
    wr = json.loads(F3A_REC.read_text())["runs"]
    drives = list(d["r1_drives"])

    def m_(runs, field, c, key):
        v = [x["rate_hz"][key] for x in runs if x[field] == c and key in x["rate_hz"]]
        return f"{np.mean(v):.1f}" if v else "-"
    for t, info in d["types"].items():
        row = _row(md, info["category"], t)
        if row is None:
            continue
        exp = [f"{m_(d['r1'], 'condition', c, t + '|left')} / {m_(d['r1'], 'condition', c, t + '|right')} "
               f"({m_(wr, 'drive', c, t + '|left')} / {m_(wr, 'drive', c, t + '|right')})" for c in drives]
        chk(f"r1:{t}", row[2:2 + len(drives)] == exp, exp)
    # R2: per-seed A, B, A/B, decision
    n_on, ndec = d["n_on"], 0
    for x in d["r2"]["runs"]:
        net = x["network_step_hz"]
        A, B = float(np.mean(net[-4:])), float(np.mean(net[n_on - 4:n_on]))
        ndec += A < 0.1 * B
        row = _row(md, str(x["seed"]), f"{A:.4f}", f"{B:.4f}", f"{A / B:.3f}", "yes" if A < 0.1 * B else "no")
        chk(f"r2:{x['seed']}", row is not None, (A, B))
    chk("r2:pass", f"**R2: {ndec} / {len(d['r2']['runs'])} seeds decay; {'pass' if ndec >= 9 else 'FAIL'}.**" in md, ndec)
    # R3: tests per rate, Holm per rate, decisions
    seeds = d["seeds"]["run"]
    by = {(x["condition"], x["seed"]): x["rate_hz"] for x in d["r3"]}

    def D(c, m):
        return np.array([by[(c, s)][f"{m}|right"] - by[(c, s)][f"{m}|left"] for s in seeds])
    passed = [m for m in d["steer_mn"] if d["symmetry"].get(m, {}).get("passed")]
    for rate in d["steer_rates"]:
        keys, ds = [], []
        for fam in d["steer_families"]:
            for m in passed:
                for s in ("L", "R"):
                    keys.append((fam, m, s))
                    ds.append(D(f"{fam}_{s}_{rate}", m) - D("B", m))
        p = np.array([1.0 if not x.any() else 0.0 if np.std(x) == 0 else float(ttest_1samp(x, 0.0).pvalue) for x in ds])
        ph = dict(zip(keys, _holm(p)))
        pr, mean = dict(zip(keys, p)), {kk: x.mean() for kk, x in zip(keys, ds)}
        for fam in d["steer_families"]:
            hits = [m for m in passed if ph[(fam, m, "L")] < 0.05 and ph[(fam, m, "R")] < 0.05
                    and mean[(fam, m, "L")] * mean[(fam, m, "R")] < 0]
            row = _row(md, fam, str(rate))
            chk(f"r3:{fam}:{rate}:decision", row is not None and row[2] == ("yes" if hits else "no")
                and all(h in row[3] for h in hits), hits)
            for m in passed:
                kl, kr = (fam, m, "L"), (fam, m, "R")
                row = _row(md, fam, str(rate), m)
                if not any(dd.any() for kk, dd in zip(keys, ds) if kk in (kl, kr)):
                    chk(f"r3:{fam}:{rate}:{m}:omitted", row is None, "row should be omitted")
                    continue
                exp = [f"{mean[kl]:+.2f}", f"{pr[kl]:.2g}", f"{ph[kl]:.2g}", f"{mean[kr]:+.2f}", f"{pr[kr]:.2g}", f"{ph[kr]:.2g}"]
                got = None if row is None or len(row) < 11 else [row[3], row[4], row[5], row[7], row[8], row[9]]
                chk(f"r3:{fam}:{rate}:{m}", got == exp, exp)
    return n, bad


F3B1_REC = ROOT / "runs" / "f3b1" / "f3b1_v3.json"
F3B1_CHK = ROOT / "runs" / "f3b1" / "input_check.json"
F3B1_BRAIN = ROOT / "runs" / "f3b1" / "brain"
F3B1_MD = ROOT / "docs" / "results" / "f3b1_optomotor.md"
PREREG09 = ROOT / "prereg" / "09_f3b1_optomotor.md"


def check_f3b1():
    """F3b-1 (prereg/09). Gains recomputed from the F3a-3 and C1 JSON files (prereg table, registered constants, report);
    FlyWire DN rates and the brain-side tests recomputed from the brain spike records themselves; gates, primary tests
    per arm and specificity recomputed from runs/f3b1/f3b1_v3.json; input check from runs/f3b1/input_check.json."""
    import re

    from scipy.stats import ttest_1samp
    sys.path.insert(0, str(ROOT / "scripts"))
    import f3b1_common as C
    md, pre = F3B1_MD.read_text(), PREREG09.read_text()
    d = json.loads(F3B1_REC.read_text())
    bad, n = [], 0

    def chk(key, ok, v):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    def pt(x):
        x = np.asarray(x, float)
        return 1.0 if not x.any() else 0.0 if np.std(x) == 0 else float(ttest_1samp(x, 0.0).pvalue)

    # gains
    lo = json.loads((ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json").read_text())["r3"]["lowest_rate"]
    c1 = json.loads((ROOT / "docs" / "results" / "f3a2_checks.json").read_text())["c1"]["f0"]
    reg = json.loads((ROOT / "configs" / "assumptions.json").read_text())["constants"]["f3b1.rate_gain"]["value"]
    for f in C.STEER_FAMILIES:
        med = c1[f]["flight"]["median"]
        g = lo[f] / med
        chk(f"gain:{f}:prereg", f"| {f} | {lo[f]} | {med!r} | {g!r} |" in pre, g)
        chk(f"gain:{f}:code", C.RATE_GAIN[f] == g and reg[f] == g, g)
        chk(f"gain:{f}:report", _row(md, f, str(lo[f]), f"{med:.2f}", f"{g:.3f}") is not None, g)
    # input check
    ic = json.loads(F3B1_CHK.read_text())
    for k, v in ic["per_stimulus"].items():
        chk(f"input:{k}", _row(md, k, f"{v['A_in_T4aT5a']:.4f}", f"{v['A_in_all']:.4f}") is not None, v)
    y = (ic["per_stimulus"]["rot_left"]["A_in_T4aT5a"], ic["per_stimulus"]["rot_right"]["A_in_T4aT5a"])
    ok_in = bool(np.sign(y[0]) != np.sign(y[1]) and min(map(abs, y)) >= 0.05)
    chk("input:pass", f"Input check: **{'passed' if ok_in else 'FAILED'}**" in md, ok_in)
    # gates
    g1 = sum(x["spikes"] for x in d["gate1"])
    ig = sum(x["ignited"] for x in d["gate2_runs"])
    chk("gate1", f"seeds 0-9): {g1} spikes" in md, g1)
    chk("gate2", f"{ig} / {len(d['gate2_runs'])} ignited" in md, ig)
    nets = [np.mean([r["network_rate_hz"] for r in d["replay"] if r["arm"] == a and r["stimulus"] == s])
            for a in C.ARMS for s in C.STIMULI] + [np.mean([r["network_rate_hz"] for r in d["gate2_runs"]])]
    chk("gate3", f"maximum {max(nets):.4f} Hz" in md, max(nets))
    # brain side from the spike records
    rates = {}
    for st in C.STIMULI:
        for s in C.SEEDS:
            z = np.load(F3B1_BRAIN / f"{st}_s{s}.npz", allow_pickle=True)
            cnt = z["dn_counts"][C.WINDOW[0]:C.WINDOW[1]].sum(0)
            fam = np.array([re.sub(r"_[a-z]$", "", t) for t in z["dn_cell_type"]])
            for f in C.BRAIN_FAMILIES:
                for sd in ("left", "right"):
                    m = (fam == f) & (z["dn_side"] == sd)
                    rates[(st, f, sd, s)] = cnt[m].sum() / m.sum() / 1.0
    for f in C.BRAIN_FAMILIES:
        for sd in ("left", "right"):
            cells = [f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
                     for v in ([rates[(st, f, sd, s)] for s in C.SEEDS] for st in C.STIMULI)]
            chk(f"brain:{f}:{sd}", _row(md, f, sd, *cells) is not None, cells)
    dbr = {f: np.array([(rates[("rot_right", f, "right", s)] - rates[("rot_right", f, "left", s)])
                        - (rates[("rot_left", f, "right", s)] - rates[("rot_left", f, "left", s)]) for s in C.SEEDS])
           for f in C.BRAIN_FAMILIES}
    p = [pt(dbr[f]) for f in C.STEER_FAMILIES]
    for f, pp, ph in zip(C.STEER_FAMILIES, p, _holm(p)):
        chk(f"brain_test:{f}", _row(md, f, f"{dbr[f].mean():.2f}", f"{np.std(dbr[f], ddof=1):.2f}", f"{pp:.4g}",
                                    f"{ph:.4g}", "yes" if ph < 0.05 else "no") is not None, (pp, ph))
    chk("brain_test:DNp31", _row(md, "DNp31", f"{dbr['DNp31'].mean():.2f}") is not None, dbr["DNp31"].mean())
    # VNC primary tests from the replay record
    by = {(r["arm"], r["stimulus"], r["seed"]): r["rate_hz"] for r in d["replay"]}
    sel = {}
    for a in C.ARMS:
        D = {m: np.array([(by[(a, "rot_right", s)][f"{m}|right"] - by[(a, "rot_right", s)][f"{m}|left"])
                          - (by[(a, "rot_left", s)][f"{m}|right"] - by[(a, "rot_left", s)][f"{m}|left"]) for s in C.SEEDS])
             for m in C.PRIMARY_MN}
        p = [pt(D[m]) for m in C.PRIMARY_MN]
        ph = _holm(p)
        for m, pp, hh in zip(C.PRIMARY_MN, p, ph):
            chk(f"vnc:{a}:{m}", _row(md, a, m, f"{D[m].mean():.2f}", f"{np.std(D[m], ddof=1):.2f}", f"{pp:.4g}",
                                     f"{hh:.4g}", "yes" if hh < 0.05 else "no") is not None, (D[m].mean(), pp, hh))
        sel[a] = bool((ph < 0.05).any())
        chk(f"vnc:{a}:verdict", _row(md, a, "**yes**" if sel[a] else "no") is not None, sel[a])
    for a, ctl in C.CONTROLS.items():
        spec = sel[a] and not any(sel[c] for c in ctl)
        txt = ("direction-selective and **specific**" if spec else "direction-selective but **not specific**"
               if sel[a] else "**not direction-selective**")
        chk(f"spec:{a}", f"- `{a}`: {txt}" in md, txt)
    return n, bad


F3B2_REC = ROOT / "runs" / "f3b2" / "f3b2_v3.json"
F3B2_MD = ROOT / "docs" / "results" / "f3b2_dn_ensemble.md"
PREREG10 = ROOT / "prereg" / "10_f3b2_dn_ensemble.md"


def check_f3b2():
    """F3b-2 (prereg/10). Ensemble and permutation against the prereg text, the crosswalk JSON and the code; SHA-256 of
    the reused brain records against the files; FlyWire family rates, the 112-family brain tests and the applied DN
    rates recomputed from the brain spike records and the recorded unit table (not through coupling/dn_bridge.py);
    gates, primary and secondary tests and specificity recomputed from runs/f3b2/f3b2_v3.json."""
    import hashlib
    import re

    from scipy.stats import ttest_1samp
    sys.path.insert(0, str(ROOT / "scripts"))
    import f3b2_common as C
    md, pre = F3B2_MD.read_text(), PREREG10.read_text()
    d = json.loads(F3B2_REC.read_text())
    bad, n = [], 0

    def chk(key, ok, v):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    def pt(x):
        x = np.asarray(x, float)
        return 1.0 if not x.any() else 0.0 if np.std(x) == 0 else float(ttest_1samp(x, 0.0).pvalue)

    def section(k):
        parts = re.split(r"(?m)^## ", md)
        return next(("## " + p for p in parts if p.startswith(f"{k}.")), "")

    # ensemble and permutation: crosswalk, prereg, code, record
    cw = json.loads(XW_JSON.read_text())["summary"]["shared_gate_both"]
    chk("ens:crosswalk", list(C.ENSEMBLE) == cw and "DNg02" not in cw, len(cw))
    lst = pre.split("The list, in the order used\n  below:")[1].split("- **DNg02")[0]
    chk("ens:prereg", re.findall(r"\b(?:DN\w+|MDN|aSP22)\b", lst) == list(C.ENSEMBLE), "list")
    pairs = dict(re.findall(r"(\w+) <- (\w+)[;.]", pre.split('"BANC target <- FlyWire source":**')[1]
                            .split("- **Everything")[0]))
    chk("perm:prereg", pairs == C.PERMUTATION, len(pairs))
    chk("perm:sattolo", C.sattolo(list(C.ENSEMBLE), C.PERMUTATION_SEED) == C.PERMUTATION, C.PERMUTATION_SEED)
    chk("perm:record", d["constants"]["permutation"] == C.PERMUTATION and d["constants"]["ensemble"] == list(C.ENSEMBLE), "")
    # reused brain records
    bdir = ROOT / d["brain_records"]["dir"]
    for name, h in d["brain_records"]["sha256"].items():
        chk(f"brain_sha:{name}", hashlib.sha256((bdir / name).read_bytes()).hexdigest() == h, name)
    chk("brain_files", len(d["brain_records"]["sha256"]) == len(C.STIMULI) * len(C.SEEDS)
        and f"{len(d['brain_records']['sha256'])} files, {d['brain_records']['n_dn']:,} FlyWire DNs" in md, "")
    # gates
    g1 = sum(x["spikes"] for x in d["gate1"])
    ig = sum(x["ignited"] for x in d["gate2_runs"])
    chk("gate1", f"seeds 0-9): {g1} spikes" in md, g1)
    chk("gate2", f"{ig} / {len(d['gate2_runs'])} ignited" in md, ig)
    nets = [np.mean([r["network_rate_hz"] for r in d["replay"] if r["arm"] == a and r["stimulus"] == s])
            for a in C.ARMS for s in C.STIMULI] + [np.mean([r["network_rate_hz"] for r in d["gate2_runs"]])]
    chk("gate3", f"maximum {max(nets):.4f} Hz" in md, max(nets))
    # brain side from the spike records
    lo, hi = C.WINDOW
    dt = C.INTERVAL_MS * 1e-3
    rates, counts = {}, {}
    for st in C.STIMULI:
        for s in C.SEEDS:
            z = np.load(bdir / f"{st}_s{s}.npz", allow_pickle=True)
            counts[(st, s)] = z["dn_counts"].astype(np.int64)
            cnt = counts[(st, s)][lo:hi].sum(0)
            fam = np.array([re.sub(r"_[a-z]$", "", t) for t in z["dn_cell_type"]])
            for f in C.ENSEMBLE:
                for sd in ("left", "right"):
                    m = (fam == f) & (z["dn_side"] == sd)
                    rates[(st, f, sd, s)] = cnt[m].sum() / m.sum() / ((hi - lo) * dt)
    dbr = {f: np.array([(rates[("rot_right", f, "right", s)] - rates[("rot_right", f, "left", s)])
                        - (rates[("rot_left", f, "right", s)] - rates[("rot_left", f, "left", s)]) for s in C.SEEDS])
           for f in C.ENSEMBLE}
    p = [pt(dbr[f]) for f in C.ENSEMBLE]
    ph = _holm(p)
    sel = [f for f, h in zip(C.ENSEMBLE, ph) if h < 0.05]
    b3 = section(3)
    chk("brain:n_selective", f"Direction-selective after Holm: **{len(sel)}** of {len(C.ENSEMBLE)}" in b3, sel)
    for f, pp, hh in zip(C.ENSEMBLE, p, ph):
        if pp < 0.05:
            rl = "/".join(f"{np.mean([rates[('rot_left', f, sd, s)] for s in C.SEEDS]):.2f}" for sd in ("left", "right"))
            rr = "/".join(f"{np.mean([rates[('rot_right', f, sd, s)] for s in C.SEEDS]):.2f}" for sd in ("left", "right"))
            chk(f"brain_test:{f}", _row(b3, f, rl, rr, f"{dbr[f].mean():.2f}", f"{np.std(dbr[f], ddof=1):.2f}",
                                        f"{pp:.4g}", f"{hh:.4g}", "yes" if hh < 0.05 else "no") is not None, (pp, hh))
    # consistency with F3b-1 (same records, same statistic)
    f3b1 = json.loads((ROOT / "docs" / "results" / "f3b1_optomotor.json").read_text())["brain"]
    for f in ("DNa04", "DNa10", "DNa15"):
        chk(f"brain_f3b1:{f}", abs(pt(dbr[f]) - f3b1["tests"][f]["p_t"]) <= 1e-12, f)
    chk("brain_f3b1:DNp31", abs(pt(dbr["DNp31"]) - f3b1["dnp31"]["p_t"]) <= 1e-12, "DNp31")
    # VNC tests from the replay record
    by = {(r["arm"], r["stimulus"], r["seed"]): r["rate_hz"] for r in d["replay"]}

    def vnc(types, sec, tag):
        sel_ = {}
        for a in C.ARMS:
            D = {m: np.array([(by[(a, "rot_right", s)][f"{m}|right"] - by[(a, "rot_right", s)][f"{m}|left"])
                              - (by[(a, "rot_left", s)][f"{m}|right"] - by[(a, "rot_left", s)][f"{m}|left"])
                              for s in C.SEEDS]) for m in types}
            pp = [pt(D[m]) for m in types]
            hh = _holm(pp)
            for m, p1, h1 in zip(types, pp, hh):
                if tag == "primary" or p1 < 0.05:
                    chk(f"{tag}:{a}:{m}", _row(sec, a, m, f"{D[m].mean():.2f}", f"{np.std(D[m], ddof=1):.2f}",
                                               f"{p1:.4g}", f"{h1:.4g}", "yes" if h1 < 0.05 else "no") is not None,
                        (D[m].mean(), p1, h1))
            sel_[a] = bool((hh < 0.05).any())
            chk(f"{tag}:{a}:verdict", _row(sec, a, "**yes**" if sel_[a] else "no") is not None, sel_[a])
        for a, ctl in C.CONTROLS.items():
            spec = sel_[a] and not any(sel_[c] for c in ctl)
            txt = ("direction-selective and **specific**" if spec else "direction-selective but **not specific**"
                   if sel_[a] else "**not direction-selective**")
            chk(f"{tag}:spec:{a}", f"- `{a}`: {txt}" in sec, txt)
        return sel_
    sel_p = vnc(C.PRIMARY_MN, section(4), "primary")
    gate = [m for m in C.STEER_MN if d["symmetry"].get(m, {}).get("passed")]
    f3a3 = json.loads((ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json").read_text())["r3"]["gate_passing"]
    chk("secondary:types", sorted(gate) == sorted(f3a3) and f"({len(gate)}): {', '.join(gate)};" in section(5), gate)
    vnc(gate, section(5), "secondary")
    spec = sel_p["ensemble"] and not any(sel_p[c] for c in C.CONTROLS["ensemble"])
    outcome = "A" if spec else "B" if sel_p["ensemble"] else "C"
    chk("outcome", f"Primary test, outcome **{outcome}**" in md, outcome)
    # post-hoc: FlyWire DN spikes under grey, from the records
    g_spk = int(sum(counts[("grey", s)].sum() for s in C.SEEDS))
    chk("posthoc:grey_spikes", f"grey 0.0000-0.0000 ({g_spk} DN spikes" in md if g_spk == 0 else
        "fire no spikes under `grey`" not in md, g_spk)
    # applied rates of the ensemble arm, from the spike records and the recorded unit table
    units = d["bridges"]["ensemble"]["units"]
    W, L = C.BRIDGE_WINDOW_INTERVALS, C.BRIDGE_LATENCY_INTERVALS
    nb = {}
    for u in units:
        nb.setdefault(f"{u['target_family']}|{u['side']}", []).append(u)
    s6 = section(6)
    for key, us in sorted(nb.items()):
        cells = []
        for st in C.STIMULI:
            vals = []
            for s in C.SEEDS:
                c = counts[(st, s)]
                tot, nn = 0.0, 0
                for u in us:
                    x = c[:, u["fw_pos"]].sum(1)
                    r = np.mean([x[max(k - L - W + 1, 0):k - L + 1].sum() for k in range(lo, hi)]) / (len(u["fw_pos"]) * W * dt)
                    tot += r * u["n_banc"]
                    nn += u["n_banc"]
                vals.append(tot / nn)
            cells.append(f"{np.mean(vals):.1f} / ")
        f, sd = key.split("|")
        row = next((ln for ln in s6.splitlines() if re.match(rf"\| {re.escape(f)}( \*)? \| {sd} \|", ln)), None)
        ok = row is not None and all(cl.strip().startswith(c) for c, cl in zip(cells, row.strip("|").split("|")[3:7]))
        chk(f"applied:{key}", ok, cells)
        chk(f"applied_star:{key}", row is not None and ((f in sel) == row.startswith(f"| {f} *")), f)
    return n, bad


F3B3_REC = ROOT / "runs" / "f3b3" / "f3b3_v3.json"
F3B3_CHK = ROOT / "runs" / "f3b3" / "input_check.json"
F3B3_MD = ROOT / "docs" / "results" / "f3b3_specificity.md"
PREREG11 = ROOT / "prereg" / "11_f3b3_specificity.md"


def check_f3b3():
    """F3b-3 (prereg/11). Permutation seeds and hash against the prereg text; the 39 derangements redrawn here and
    checked against the record and the recorded bridge units; SHA-256 of the fresh brain records against the files;
    the 112-family brain tests and the applied DN rates recomputed from the brain spike records; gates, (a), (b), (c),
    null distribution, secondary and exploratory tables recomputed from runs/f3b3/f3b3_v3.json."""
    import hashlib
    import re

    from scipy.stats import ttest_1samp
    sys.path.insert(0, str(ROOT / "scripts"))
    import f3b3_common as C
    md, pre = F3B3_MD.read_text(), PREREG11.read_text()
    d = json.loads(F3B3_REC.read_text())
    bad, n = [], 0

    def chk(key, ok, v):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    def pt(x):
        x = np.asarray(x, float)
        return 1.0 if not x.any() else 0.0 if np.std(x) == 0 else float(ttest_1samp(x, 0.0).pvalue)

    def section(k):
        parts = re.split(r"(?m)^## ", md)
        return next(("## " + p for p in parts if p.startswith(f"{k}.")), "")

    # permutations: prereg seeds and hash, independent redraw, record, bridge units
    seeds = {a: int(s) for a, s in re.findall(r"(random_\d+) (\d{8})", " ".join(pre.split()))}
    chk("perm:seeds", list(seeds) == list(C.RANDOM_ARMS) and list(seeds.values()) == list(C.PERM_SEEDS), seeds)
    chk("perm:sha_prereg", C.PERMUTATIONS_SHA256 in pre and d["constants"]["permutations_sha256"] == C.PERMUTATIONS_SHA256,
        C.PERMUTATIONS_SHA256)
    redrawn = {}
    for a, s in seeds.items():
        rng = np.random.default_rng(s)
        while True:
            p = rng.permutation(len(C.ENSEMBLE))
            if (p != np.arange(len(p))).all():
                break
        redrawn[a] = {C.ENSEMBLE[i]: C.ENSEMBLE[int(j)] for i, j in enumerate(p)}
    chk("perm:sha_redrawn", hashlib.sha256(json.dumps(redrawn, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == C.PERMUTATIONS_SHA256, "")
    chk("perm:record", d["constants"]["permutations"] == redrawn, "")
    for a in C.RANDOM_ARMS:
        us = d["bridges"][a]["units"]
        chk(f"perm:units:{a}", all(redrawn[a][u["target_family"]] == u["source_family"] for u in us)
            and {u["target_family"] for u in us} == set(C.ENSEMBLE), a)
    chk("units:ensemble", all(u["source_family"] == u["target_family"] for u in d["bridges"]["ensemble"]["units"]), "")
    # fresh brain records
    bdir = ROOT / d["brain_records"]["dir"]
    for name, h in d["brain_records"]["sha256"].items():
        chk(f"brain_sha:{name}", hashlib.sha256((bdir / name).read_bytes()).hexdigest() == h, name)
    names = {f"{st}_s{s}.npz" for st in C.STIMULI for s in range(10, 30)}
    chk("brain_files", set(d["brain_records"]["sha256"]) == names and not any(f"_s{s}.npz" in x for x in names for s in range(10))
        and f"{len(names)} records (4 stimuli x seeds 10-29), {d['brain_records']['n_dn']:,} FlyWire DNs" in md, "")
    ic = json.loads(F3B3_CHK.read_text())
    chk("input_check", ic["passed"] and "Input check (prereg/09 section 3): passed" in md
        and f"bit for bit: {'yes' if ic['equals_f3b1_render'] else 'no'}" in md, ic["passed"])
    # gates
    g1 = sum(x["spikes"] for x in d["gate1"])
    ig = sum(x["ignited"] for x in d["gate2_runs"])
    chk("gate1", f"seeds 10-19): {g1} spikes" in md and [x["seed"] for x in d["gate1"]] == list(range(10, 20)), g1)
    chk("gate2", f"{ig} / {len(d['gate2_runs'])} ignited" in md and [x["seed"] for x in d["gate2_runs"]] == list(range(250, 300)), ig)
    nets = [np.mean([r["network_rate_hz"] for r in d["replay"] if r["arm"] == a and r["stimulus"] == s])
            for a in C.ARMS for s in C.STIMULI] + [np.mean([r["network_rate_hz"] for r in d["gate2_runs"]])]
    chk("gate3", f"maximum {max(nets):.4f} Hz" in md, max(nets))
    gates_ok = g1 == 0 and ig == 0 and max(nets) < 5.0
    chk("replay_count", len(d["replay"]) == len(C.ARMS) * len(C.STIMULI) * 20, len(d["replay"]))
    # brain side from the spike records
    lo, hi = C.WINDOW
    dt = C.INTERVAL_MS * 1e-3
    seeds_b = list(range(10, 30))
    rates, counts = {}, {}
    for st in C.STIMULI:
        for s in seeds_b:
            z = np.load(bdir / f"{st}_s{s}.npz", allow_pickle=True)
            chk(f"brain_seed:{st}:{s}", int(z["seed"]) == s, s)
            counts[(st, s)] = z["dn_counts"].astype(np.int64)
            cnt = counts[(st, s)][lo:hi].sum(0)
            fam = np.array([re.sub(r"_[a-z]$", "", t) for t in z["dn_cell_type"]])
            for f in C.ENSEMBLE:
                for sd in ("left", "right"):
                    m = (fam == f) & (z["dn_side"] == sd)
                    rates[(st, f, sd, s)] = cnt[m].sum() / m.sum() / ((hi - lo) * dt)
    dbr = {f: np.array([(rates[("rot_right", f, "right", s)] - rates[("rot_right", f, "left", s)])
                        - (rates[("rot_left", f, "right", s)] - rates[("rot_left", f, "left", s)]) for s in seeds_b])
           for f in C.ENSEMBLE}
    p = [pt(dbr[f]) for f in C.ENSEMBLE]
    ph = _holm(p)
    sel = [f for f, h in zip(C.ENSEMBLE, ph) if h < 0.05]
    b3 = section(3)
    chk("brain:n_selective", f"Direction-selective after Holm: **{len(sel)}** of {len(C.ENSEMBLE)}" in b3, sel)
    for f, pp, hh in zip(C.ENSEMBLE, p, ph):
        if hh < 0.05:
            rl = "/".join(f"{np.mean([rates[('rot_left', f, sd, s)] for s in seeds_b]):.2f}" for sd in ("left", "right"))
            rr = "/".join(f"{np.mean([rates[('rot_right', f, sd, s)] for s in seeds_b]):.2f}" for sd in ("left", "right"))
            chk(f"brain_test:{f}", _row(b3, f, rl, rr, f"{dbr[f].mean():.2f}", f"{np.std(dbr[f], ddof=1):.2f}",
                                        f"{pp:.4g}", f"{hh:.4g}") is not None, (pp, hh))
    # VNC: D per arm and type from the replay record
    by = {(r["arm"], r["stimulus"], r["seed"]): r["rate_hz"] for r in d["replay"]}

    def D(a, m):
        return np.array([(by[(a, "rot_right", s)][f"{m}|right"] - by[(a, "rot_right", s)][f"{m}|left"])
                         - (by[(a, "rot_left", s)][f"{m}|right"] - by[(a, "rot_left", s)][f"{m}|left"]) for s in seeds_b])

    def perm(m):
        te = D("ensemble", m).mean()
        null = np.array([D(a, m).mean() for a in C.RANDOM_ARMS])
        ge = int((np.abs(null) >= abs(te)).sum())
        same = 0 if te == 0 else int(((np.sign(null) == np.sign(te)) & (null != 0)).sum())
        return te, null, (1 + ge) / (len(null) + 1), len(null) - ge, same / len(null)

    s4 = section(4)
    sig = {}
    for a in ("ensemble", "bridge_off"):
        Ds = {m: D(a, m) for m in C.PRIMARY_MN}
        pp = [pt(Ds[m]) for m in C.PRIMARY_MN]
        hh = _holm(pp)
        sig[a] = [m for m, h in zip(C.PRIMARY_MN, hh) if h < 0.05]
        for m, p1, h1 in zip(C.PRIMARY_MN, pp, hh):
            chk(f"primary:{a}:{m}", _row(s4, a, m, f"{Ds[m].mean():+.3f}", f"{np.std(Ds[m], ddof=1):.3f}", f"{p1:.4g}",
                                         f"{h1:.4g}", "yes" if h1 < 0.05 else "no") is not None, (Ds[m].mean(), p1, h1))
    b_ok = []
    s5 = section(5)
    for m in C.PRIMARY_MN:
        te, null, pp, nsm, same = perm(m)
        if m in sig["ensemble"] and pp <= C.PERM_ALPHA:
            b_ok.append(m)
        chk(f"perm:{m}", _row(s4, m, f"{te:+.3f}", "yes" if m in sig["ensemble"] else "no", f"{nsm} / {C.N_RANDOM}",
                              f"{pp:.3f}", "yes" if m in b_ok else "no", f"{same:.3f}") is not None, (te, pp, nsm))
        q = np.quantile(null, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        chk(f"null:{m}", _row(s5, m, *[f"{x:+.3f}" for x in q], f"{te:+.3f}", f"{nsm} / {C.N_RANDOM}") is not None, q)
    nulls = {m: np.sort(perm(m)[1]) for m in C.PRIMARY_MN}
    for i in range(C.N_RANDOM):
        chk(f"null_rank:{i + 1}", _row(s5, str(i + 1), *[f"{nulls[m][i]:+.3f}" for m in C.PRIMARY_MN]) is not None, i)
    bo_sel = bool(sig["bridge_off"])
    outcome = ("not replicated" if not sig["ensemble"] else
               "specific" if b_ok and not bo_sel and gates_ok else "replicated, not specific")
    chk("outcome", f"**Outcome (prereg/11 section 7): {outcome}**" in md and f"- **Outcome: {outcome}**" in s4, outcome)
    chk("bridge_off", f"`bridge_off` direction-selective: **{'yes' if bo_sel else 'no'}**" in s4, bo_sel)
    # secondary set and exploratory signs
    gate = [m for m in C.STEER_MN if d["symmetry"].get(m, {}).get("passed")]
    f3a3 = json.loads((ROOT / "docs" / "results" / "f3a3_vnc_subgraph.json").read_text())["r3"]["gate_passing"]
    s6, s7 = section(6), section(7)
    chk("secondary:types", sorted(gate) == sorted(f3a3) and f"({len(gate)}): {', '.join(gate)};" in s6, gate)
    Dg = {m: D("ensemble", m) for m in gate}
    hh = _holm([pt(Dg[m]) for m in gate])
    hb = _holm([pt(D("bridge_off", m)) for m in gate])
    exp_pre = {m: e for m, e in re.findall(r"^\| (\w+) \| [^|]+ \| (\+|-|not fixed[^|]*) \|$", pre, re.M)}
    for m in ("tp1", "tp2", "tpn", "ps1", "ps2"):
        exp_pre[m] = "not fixed"
    for i, m in enumerate(gate):
        te, _, pp, _, same = perm(m)
        chk(f"secondary:{m}", _row(s6, m, f"{te:+.3f}", f"{hh[i]:.4g}", "yes" if hh[i] < 0.05 else "no", f"{hb[i]:.4g}",
                                   f"{pp:.3f}", f"{same:.3f}") is not None, (te, hh[i]))
        e = exp_pre.get(m)
        es = "+" if e == "+" else "-" if e == "-" else "not fixed"
        obs = "+" if te > 0 else "-" if te < 0 else "0"
        ag = "-" if es == "not fixed" or obs == "0" else ("yes" if es == obs else "no")
        row = next((ln for ln in s7.splitlines() if ln.startswith(f"| {m} |")), "")
        cells = [c.strip() for c in row.strip("|").split("|")]
        chk(f"exploratory:{m}", len(cells) == 6 and cells[2] == es and cells[3] == f"{obs} ({te:+.3f} Hz)"
            and cells[4] == ("yes" if hh[i] < 0.05 else "no") and cells[5] == ag, (m, cells))
    # applied rates of the ensemble arm, from the spike records and the recorded unit table
    units = d["bridges"]["ensemble"]["units"]
    W, L = C.BRIDGE_WINDOW_INTERVALS, C.BRIDGE_LATENCY_INTERVALS
    nb = {}
    for u in units:
        nb.setdefault(f"{u['target_family']}|{u['side']}", []).append(u)
    s8 = section(8)
    for key, us in sorted(nb.items()):
        cells = []
        for st in C.STIMULI:
            vals = []
            for s in seeds_b:
                c = counts[(st, s)]
                tot, nn = 0.0, 0
                for u in us:
                    x = c[:, u["fw_pos"]].sum(1)
                    r = np.mean([x[max(k - L - W + 1, 0):k - L + 1].sum() for k in range(lo, hi)]) / (len(u["fw_pos"]) * W * dt)
                    tot += r * u["n_banc"]
                    nn += u["n_banc"]
                vals.append(tot / nn)
            cells.append(np.mean(vals))
        f, sd = key.split("|")
        row = next((ln for ln in s8.splitlines() if re.match(rf"\| {re.escape(f)}( \*)? \| {sd} \|", ln)), None)
        if row is None:                    # only families with a zero applied rate on both sides are left out
            chk(f"applied_zero:{key}", not any(cells), cells)
            continue
        # numeric comparison within half a display unit: the mean rates hit exact .x5 ties, which the bridge and
        # this recomputation can round to different sides
        rep = [float(cl.split("/")[0]) for cl in row.strip("|").split("|")[3:7]]
        ok = all(abs(r - c) <= 0.05 + 1e-9 for r, c in zip(rep, cells))
        chk(f"applied:{key}", ok, (cells, rep))
        chk(f"applied_star:{key}", (f in sel) == row.startswith(f"| {f} *"), f)
    return n, bad


SUMMARY_MD = ROOT / "docs" / "results" / "SUMMARY_F0_F3.md"
SUMMARY_JSON = ROOT / "docs" / "results" / "summary_f0_f3.json"
DOC08 = ROOT / "docs" / "08_CLOSED_LOOP_OPTOMOTOR.md"
F4_JSON = ROOT / "docs" / "results" / "f4_design_numbers.json"
F4_JSON_V2 = ROOT / "docs" / "results" / "f4_design_numbers_v2.json"


def check_summary():
    """docs/results/SUMMARY_F0_F3.md: (1) the template carries no number of its own; (2) every fact re-captured from the
    current report equals the stored one; (3) the template re-rendered from the current reports equals the committed
    summary; (4) a set of facts is also compared with the JSON of the report they come from."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import summary_f0_f3 as S
    res = ROOT / "docs" / "results"
    bad, n = [], 0

    def chk(key, ok, v=None):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    stray = S.stray_numbers(S.TEMPLATE)
    chk("summary:template_numbers", not stray, stray)
    stored = json.loads(SUMMARY_JSON.read_text())
    vals = S.capture()
    for k in S.FACTS:
        chk(f"summary:fact:{k}", k in stored and stored[k]["values"] == vals[k] and stored[k]["file"] == S.FACTS[k][0],
            (vals[k], stored.get(k, {}).get("values")))
    chk("summary:facts_complete", set(stored) == set(S.FACTS), sorted(set(stored) ^ set(S.FACTS)))
    chk("summary:render", S.render(S.TEMPLATE, vals) == SUMMARY_MD.read_text(), "md differs from template + reports")
    # facts against the JSON of their report (independent of the regular expressions)
    xw = json.loads((res / "f3_dn_crosswalk.json").read_text())["summary"]
    chk("summary:x_gate", vals["x_gate"][0] == str(xw["n_shared_gate_both"]), vals["x_gate"])
    chk("summary:x_shared", vals["x_shared"] == [str(xw["shared_families"]), str(xw["shared_neurons_flywire"]),
                                                  str(xw["shared_neurons_banc"])], vals["x_shared"])
    ms = json.loads((res / "f0_multiseed.json").read_text())
    chk("summary:f0_pass", vals["f0_pass"][0] == str(sum(1 for r in ms if r["S1"] and r["S2"])), vals["f0_pass"])
    chk("summary:f0_n", vals["f0_n"][0] == str(len(ms)), vals["f0_n"])
    chk("summary:f0_posdiff", abs(float(vals["f0_posdiff"][0]) - max(r["pos_diff_to_v1_max_mm"] for r in ms)) < 0.005,
        vals["f0_posdiff"])
    ig = json.loads((res / "f0_ignition_frequency.json").read_text())["arms"]
    chk("summary:f0_ign100", vals["f0_ign100"][0] == f"{ig['new']['ignited']} / {ig['new']['n']}"
        and abs(float(vals["f0_ign100"][1]) - 100 * ig["new"]["ci95"][1]) < 0.05, vals["f0_ign100"])
    c3 = json.loads((res / "f3b3_specificity.json").read_text())
    for i, t in enumerate(("b3", "i1", "hg1", "hg4")):
        chk(f"summary:c_T:{t}", vals["c_T"][i] == f"{c3['primary']['tests']['ensemble'][t]['mean']:+.3f}", vals["c_T"][i])
    chk("summary:c_brain", vals["c_brain"] == [str(len(c3["brain"]["selective"])), "112"], vals["c_brain"])
    b2 = json.loads((res / "f3b2_dn_ensemble.json").read_text())
    chk("summary:b2_brain", vals["b2_brain"] == [str(len(b2["brain"]["selective"])), "112"], vals["b2_brain"])
    a3 = json.loads((res / "f3a3_vnc_subgraph.json").read_text())["subgraph"]["kept"]
    chk("summary:a3_sub", vals["a3_sub"] == [f"{a3['neurons']:,}", f"{a3['pairs']:,}", f"{a3['synapses']:,}"], vals["a3_sub"])
    return n, bad


def check_f4_design():
    """docs/08 numbers: docs/results/f4_design_numbers.json (v1) and f4_design_numbers_v2.json (Lindsay 2017 signs,
    section 10) are rebuilt from the F3b-3 records and compared; the
    syndirectional index of `ensemble` and the random-wiring counts are recomputed here with plain loops; the GEN blocks
    of docs/08 are re-rendered and compared with the document."""
    import hashlib
    import re
    sys.path.insert(0, str(ROOT / "scripts"))
    import f4_design_numbers as F
    bad, n = [], 0

    def chk(key, ok, v=None):
        nonlocal n
        n += 1
        if not ok:
            bad.append((key, v))

    rec = json.loads(F3B3_REC.read_text())
    seeds = sorted({r["seed"] for r in rec["replay"]})
    by = {(r["arm"], r["stimulus"], r["seed"]): r["rate_hz"] for r in rec["replay"]}
    doc = DOC08.read_text()
    all_blocks = set()
    for ver, path, signs in (("v1", F4_JSON, F.SIGNS_LATERAL), ("v2", F4_JSON_V2, F.SIGNS_LATERAL_V2)):
        if not path.exists():
            chk(f"f4:{ver}:json_exists", False, str(path))
            continue
        stored = json.loads(path.read_text())
        rebuilt = F.build(ver)
        chk(f"f4:{ver}:json_equals_rebuild", json.dumps(stored, sort_keys=True) == json.dumps(rebuilt, sort_keys=True))
        chk(f"f4:{ver}:source_sha", stored["source"]["sha256"] == hashlib.sha256(F.RECORDS.read_bytes()).hexdigest())
        chk(f"f4:{ver}:selfcheck", stored["selfcheck_T_equals_f3b3"] is True)

        def S(arm, seed, signs=signs):
            def dA(stim):
                rr = by[(arm, stim, seed)]
                return sum(sg * (rr.get(f"{t}|left", 0.0) - rr.get(f"{t}|right", 0.0)) for t, sg in signs.items())
            return (dA("rot_right") - dA("rot_left")) / 2.0

        ens = [S("ensemble", s) for s in seeds]
        e = stored["sets"]["lateral"]["arms"]["ensemble"]
        chk(f"f4:{ver}:S_ensemble_mean", abs(np.mean(ens) - e["mean"]) < 1e-9, (np.mean(ens), e["mean"]))
        chk(f"f4:{ver}:S_ensemble_pos", sum(1 for v in ens if v > 0) == e["n_pos"] and sum(1 for v in ens if v < 0) == e["n_neg"])
        rnd = [np.mean([S(f"random_{k}", s) for s in seeds]) for k in range(1, 40)]
        r = stored["sets"]["lateral"]["random"]
        chk(f"f4:{ver}:random_counts", (sum(1 for v in rnd if v > 0), sum(1 for v in rnd if v < 0)) ==
            (r["n_syndirectional"], r["n_anti"]), (r["n_syndirectional"], r["n_anti"]))
        chk(f"f4:{ver}:bridge_off_zero_mean_abs_small", abs(stored["sets"]["lateral"]["arms"]["bridge_off"]["mean"]) < 0.1)
        # GEN blocks of docs/08
        B = F.blocks(stored, ver)
        all_blocks |= set(B)
        for k, v in B.items():
            m = re.search(rf"<!-- GEN:{k} -->\n(.*?)\n<!-- /GEN:{k} -->", doc, re.S)
            chk(f"f4:{ver}:block:{k}", bool(m) and m.group(1) == v, k)
    if F4_JSON_V2.exists():
        st2 = json.loads(F4_JSON_V2.read_text())
        chk("f4:v2:gain_fixed_is_v1_G1", st2["gain_fixed"]["G1_v1_deg_per_hz"] ==
            json.loads(F4_JSON.read_text())["gain"]["G1_deg_per_hz"])
        chk("f4:v2:primary_excludes_conflicts", not {"iii1", "iii4"} & set(F.SIGNS_LATERAL_V2))
    chk("f4:no_unknown_blocks", set(re.findall(r"<!-- GEN:(\w+) -->", doc)) == all_blocks)
    return n, bad



def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bad = []
    exp, d = f3a_expected()
    md = F3A_MD.read_text()
    rows = {ln.split("|")[1].strip() + "|" + ln.split("|")[2].strip(): ln for ln in md.splitlines()
            if ln.startswith("| ") and ln.count("|") > 10}
    cols = None
    for ln in md.splitlines():
        if ln.startswith("| category | type |"):
            cols = [c.strip() for c in ln.strip("|").split("|")]
    n = 0
    for k, v in exp.items():
        if v is None:
            continue
        if "--print" in argv:
            print(k, "=", v)
            continue
        if k.startswith("mn:"):
            _, t, drive = k.split(":")
            row = next((ln for key, ln in rows.items() if key.endswith("|" + t)), None)
            if row is None or f"{drive} L/R" not in cols:
                bad.append((k, "row/column missing")); continue
            cell = [c.strip() for c in row.strip("|").split("|")][cols.index(f"{drive} L/R")]
            ok = cell == v
        elif k.startswith("drive:"):
            ok = v in md
        else:
            ok = v in md
        n += 1
        if not ok:
            bad.append((k, v))
    # gate 1 (spikes in no-input runs)
    n += 1
    g1 = sum(x["total_spikes"] for x in d["runs"] if x["drive"] == "D0")
    if g1 != 0 or "0 (seeds 0-9: 0)" not in md:
        bad.append(("gate1", g1))
    # crosswalk summary
    s = json.loads(XW_JSON.read_text())["summary"]
    xmd = XW_MD.read_text()
    for frag in (f"FlyWire {s['flywire_dn_neurons']}, BANC {s['banc_dn_neurons']}",
                 f"FlyWire {s['flywire_families']}, BANC {s['banc_families']}, shared by name {s['shared_families']} "
                 f"({s['shared_neurons_flywire']} FlyWire and {s['shared_neurons_banc']} BANC neurons)",
                 f"in both datasets: {s['n_shared_gate_both']}"):
        n += 1
        if frag not in xmd:
            bad.append(("crosswalk", frag))
    n += 1
    if s["n_shared_gate_both"] != len(s["shared_gate_both"]):
        bad.append(("crosswalk", "n_shared_gate_both != len(list)"))
    if F3A2_MD.exists():
        k, b = check_f3a2()
        n += k
        bad += b
    if F3A3_MD.exists():
        k, b = check_f3a3()
        n += k
        bad += b
    if F3B1_MD.exists():
        k, b = check_f3b1()
        n += k
        bad += b
    if F3B2_MD.exists():
        k, b = check_f3b2()
        n += k
        bad += b
    if F3B3_MD.exists():
        k, b = check_f3b3()
        n += k
        bad += b
    if SUMMARY_MD.exists():
        k, b = check_summary()
        n += k
        bad += b
    if DOC08.exists() and F4_JSON.exists():
        k, b = check_f4_design()
        n += k
        bad += b
    if "--print" in argv:
        return 0
    print(f"checked {n} items, {len(bad)} mismatches")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
