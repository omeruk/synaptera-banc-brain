"""Data-only DN type crosswalk FlyWire v783 <-> BANC v3 (input to docs/07_HYBRID_INTERFACE.md). No simulation.

    env -u PYTHONPATH python scripts/f3_dn_crosswalk.py [--overwrite]

Family key = cell_type with a single trailing ``_<letter>`` removed (FlyWire splits DNg02 into DNg02_a..h; BANC has
DNg02 plus DNg02_d/_h). Per dataset and family: neurons per side, input-synapse symmetry gate (total and excitatory,
|log2(R/L)| < 0.5, CLAUDE.md rule 5), and for BANC the direct synapses onto motor-neuron groups (wing, haltere, neck,
TTMn, leg).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import qa  # noqa: E402
from connectome.adapters import flywire783 as fw  # noqa: E402
import f3a_vnc_run as F  # noqa: E402

OUT = ROOT / "docs" / "results" / "f3_dn_crosswalk"
THR = 0.5


def family(ct):
    return re.sub(r"_[a-z]$", "", str(ct))


def per_family(cx, sc_col="super_class", dn_name="descending"):
    n = cx.neurons
    tot, exc = qa.input_synapses(cx)
    m = (n[sc_col] == dn_name).to_numpy()
    df = pd.DataFrame(dict(fam=n["cell_type"].map(family)[m].to_numpy(), side=n["side"][m].to_numpy(), tot=tot[m], exc=exc[m]))
    g = df.groupby(["fam", "side"]).agg(n=("tot", "size"), tot=("tot", "sum"), exc=("exc", "sum")).unstack("side")
    rows = {}
    for fam in g.index:
        r = {s: dict(n=int(g.loc[fam, ("n", s)]) if not pd.isna(g.loc[fam, ("n", s)]) else 0,
                     tot=float(g.loc[fam, ("tot", s)]) if not pd.isna(g.loc[fam, ("tot", s)]) else 0.0,
                     exc=float(g.loc[fam, ("exc", s)]) if not pd.isna(g.loc[fam, ("exc", s)]) else 0.0)
             for s in ("left", "right", "center")
             if ("n", s) in g.columns}
        L, R = r.get("left", dict(n=0, tot=0, exc=0)), r.get("right", dict(n=0, tot=0, exc=0))
        o = dict(n_left=L["n"], n_right=R["n"], n_center=r.get("center", dict(n=0))["n"], testable=bool(L["n"] and R["n"]))
        if o["testable"] and L["tot"] > 0 and R["tot"] > 0:
            o["log2_total"] = float(np.log2(R["tot"] / L["tot"]))
            o["log2_exc"] = float(np.log2(R["exc"] / L["exc"])) if L["exc"] > 0 and R["exc"] > 0 else None
            o["gate"] = bool(abs(o["log2_total"]) < THR and o["log2_exc"] is not None and abs(o["log2_exc"]) < THR)
        else:
            o["gate"] = None
        rows[fam] = o
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT.with_suffix(".md").exists() or OUT.with_suffix(".json").exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    fcx = fw.load_connectome(hash_files=False)
    fl = per_family(fcx)
    nfl = int((fcx.neurons["super_class"] == "descending").sum())
    del fcx
    bcx = F.load()
    bn = bcx.neurons
    bf = per_family(bcx)
    nbn = int((bn["super_class"] == "descending").sum())
    # BANC direct synapses DN family -> MN groups
    pre, post = bcx.edge_endpoints()
    w = bcx.edges["n_syn"].to_numpy()
    types = F.readout_types(bn)
    cats = {"wing": ("wing", "wing_steering"), "haltere": ("haltere",), "neck": ("neck",), "TTMn": ("TTMn",)}
    mnmask = {k: np.zeros(bcx.n_neurons, bool) for k in cats}
    for t, (c, m) in types.items():
        for k, cs in cats.items():
            if c in cs:
                mnmask[k] |= m.to_numpy()
    mnmask["leg"] = ((bn["super_class"] == "motor") & (bn["cell_class"].astype(str).str.contains("leg_motor")) &
                     (bn["cell_type"] != "TTMn")).to_numpy()
    dn = (bn["super_class"] == "descending").to_numpy()
    fam_of = bn["cell_type"].map(family).to_numpy()
    syn = {}
    sel = dn[pre]
    for k, m in mnmask.items():
        e = sel & m[post]
        s = pd.Series(w[e]).groupby(fam_of[pre[e]]).sum()
        for fam, v in s.items():
            syn.setdefault(fam, {})[k] = int(v)
    both = sorted(set(fl) & set(bf))
    rows = []
    for fam in sorted(set(fl) | set(bf)):
        f, b = fl.get(fam), bf.get(fam)
        rows.append(dict(family=fam, in_flywire=f is not None, in_banc=b is not None, flywire=f, banc=b,
                         banc_to_mn=syn.get(fam, {})))
    summary = dict(flywire_dn_neurons=nfl, banc_dn_neurons=nbn, flywire_families=len(fl), banc_families=len(bf),
                   shared_families=len(both),
                   shared_neurons_flywire=int(sum(fl[f]["n_left"] + fl[f]["n_right"] + fl[f]["n_center"] for f in both)),
                   shared_neurons_banc=int(sum(bf[f]["n_left"] + bf[f]["n_right"] + bf[f]["n_center"] for f in both)),
                   shared_gate_both=[f for f in both if fl[f]["gate"] and bf[f]["gate"]],
                   shared_count_equal=[f for f in both if (fl[f]["n_left"], fl[f]["n_right"]) == (bf[f]["n_left"], bf[f]["n_right"])])
    summary["n_shared_gate_both"] = len(summary["shared_gate_both"])
    # DN families carrying direct synapses onto flight/neck MNs in BANC
    mn_dn = [r for r in rows if r["banc_to_mn"] and any(r["banc_to_mn"].get(k, 0) for k in ("wing", "haltere", "neck", "TTMn"))]
    mn_dn.sort(key=lambda r: -sum(r["banc_to_mn"].get(k, 0) for k in ("wing", "haltere", "neck", "TTMn")))
    top = mn_dn[:40]
    out = dict(summary=summary, families=rows)
    L = ["# F3 interface prep: DN type crosswalk FlyWire v783 <-> BANC v3 (data only)", "",
         "Generated by `scripts/f3_dn_crosswalk.py`. Family key = `cell_type` without a single trailing `_<letter>`. Symmetry "
         "gate = input synapses on each dataset's own graph, total and excitatory, |log2(R/L)| < 0.5.", "",
         "## 1. Summary", "",
         f"- DN neurons: FlyWire {nfl}, BANC {nbn}; type families: FlyWire {len(fl)}, BANC {len(bf)}, shared by name {len(both)} "
         f"({summary['shared_neurons_flywire']} FlyWire and {summary['shared_neurons_banc']} BANC neurons).",
         f"- Shared families passing the symmetry gate in both datasets: {summary['n_shared_gate_both']}; with identical "
         f"left/right counts in both: {len(summary['shared_count_equal'])}.", "",
         "## 2. BANC DN families with direct synapses onto wing / haltere / neck / TTMn / leg motor neurons (top 40 by "
         "wing + haltere + neck + TTMn)", "",
         "| family | FlyWire L/R | BANC L/R | gate FlyWire | gate BANC | -> wing | -> haltere | -> neck | -> TTMn | -> leg |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    gs = lambda d: "-" if d is None or d["gate"] is None else ("pass" if d["gate"] else "FAIL")  # noqa: E731
    for r in top:
        f, b, s = r["flywire"], r["banc"], r["banc_to_mn"]
        L.append(f"| {r['family']} | {'absent' if f is None else str(f['n_left']) + '/' + str(f['n_right'])} | "
                 f"{b['n_left']}/{b['n_right']} | {gs(f)} | {gs(b)} | " +
                 " | ".join(str(s.get(k, 0)) for k in ("wing", "haltere", "neck", "TTMn", "leg")) + " |")
    n_missing = sum(1 for r in mn_dn if not r["in_flywire"])
    summary["banc_mn_dn_families"] = len(mn_dn)
    summary["banc_mn_dn_families_absent_in_flywire"] = n_missing
    summary["banc_mn_dn_families_shared_gate_both"] = sum(1 for r in mn_dn if r["in_flywire"] and r["flywire"]["gate"] and r["banc"]["gate"])
    L += ["", f"{len(mn_dn)} BANC DN families have direct synapses onto these motor neurons; {n_missing} of them have no "
          f"FlyWire family of the same name; {summary['banc_mn_dn_families_shared_gate_both']} are shared and pass the "
          "gate in both datasets.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str))
    OUT.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
