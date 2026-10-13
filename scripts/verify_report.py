"""Recompute the numbers of the committed result reports from their raw records and check them against the text
(CLAUDE.md rule 11). Independent of the report scripts: the F3a tables are recomputed here from `runs/f3a/f3a_v3.json`
and each formatted row must appear verbatim in `docs/results/f3a_banc_vnc.md`; the crosswalk summary numbers are
checked against `docs/results/f3_dn_crosswalk.json` and its text.

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
    if "--print" in argv:
        return 0
    print(f"checked {n} items, {len(bad)} mismatches")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
