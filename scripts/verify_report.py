"""Recompute the numbers of the committed result reports from their raw records and check them against the text
(CLAUDE.md rule 11). Independent of the report scripts: the F3a tables are recomputed here from `runs/f3a/f3a_v3.json`
and each formatted row must appear verbatim in `docs/results/f3a_banc_vnc.md`; the crosswalk summary numbers are
checked against `docs/results/f3_dn_crosswalk.json` and its text; F3a-2 (`docs/results/f3a2_checks.md`): C1 rates
recomputed from the F0 spike records, gates / C2 / C3 statistics from `runs/f3a2/f3a2_v3.json`; F3a-3 (`docs/results/f3a3_vnc_subgraph.md`): subgraph counts against the
committed JSON, the prereg and the report, gates / R1 / R2 / R3 recomputed from `runs/f3a3/f3a3_v3.json`; F3b-1 (`docs/results/f3b1_optomotor.md`): gains against
the prereg and the registry, FlyWire DN rates and brain-side tests from the brain spike records, gates and primary tests
per arm from `runs/f3b1/f3b1_v3.json`.

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
    if "--print" in argv:
        return 0
    print(f"checked {n} items, {len(bad)} mismatches")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
