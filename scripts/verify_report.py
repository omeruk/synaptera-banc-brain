"""Recompute the numbers of the committed result reports from their raw records and check them against the text
(CLAUDE.md rule 11). Independent of the report scripts: the F3a tables are recomputed here from `runs/f3a/f3a_v3.json`
and each formatted row must appear verbatim in `docs/results/f3a_banc_vnc.md`; the crosswalk summary numbers are
checked against `docs/results/f3_dn_crosswalk.json` and its text; F3a-2 (`docs/results/f3a2_checks.md`): C1 rates
recomputed from the F0 spike records, gates / C2 / C3 statistics from `runs/f3a2/f3a2_v3.json`.

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
    if "--print" in argv:
        return 0
    print(f"checked {n} items, {len(bad)} mismatches")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
