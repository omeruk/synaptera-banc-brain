"""BANC VNC subgraph of prereg/08 (data only): the part of `banc888_full_v3` the hybrid keeps when the FlyWire brain
replaces the BANC brain.

    env -u PYTHONPATH python scripts/f3a3_subgraph.py --out docs/results/f3a3_subgraph.json

Kept neurons: `region` in {VNC, neck} (region of the dominant neuropil, `connectome.adapters.banc.region_of_neuropil`),
plus every `super_class == motor`, every `super_class == descending` (input nodes) and every ascending neuron
(`super_class` in ASCENDING, outputs). Kept edges: every edge between two kept neurons, except edges onto a DN (a DN is an
input node: it is driven from outside and receives nothing). Everything else (BANC brain and SEZ neurons and their
synapses) is removed. No weight, sign or threshold changes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome.schema import Connectome  # noqa: E402

KEEP_REGIONS = ("VNC", "neck")
ASCENDING = ("ascending", "ascending_visceral_circulatory", "sensory_ascending")


def _role(n):
    sc = n["super_class"].astype(object)
    role = np.full(len(n), "region", object)
    role[sc.isin(ASCENDING).to_numpy()] = "ascending"
    role[(sc == "motor").to_numpy()] = "motor"
    role[(sc == "descending").to_numpy()] = "descending"
    return role


def vnc_subgraph(cx: Connectome):
    """(sub-Connectome, info). ``sub.meta["global_index"]`` maps subgraph -> whole-graph engine index;
    ``sub.meta["subgraph"]`` holds the definition and the counts."""
    n = cx.neurons
    reg = n["region"].astype(object).where(n["region"].notna(), "NA").to_numpy()
    role = _role(n)
    keep = np.isin(reg, KEEP_REGIONS) | (role != "region")
    is_dn = role == "descending"
    pre, post = cx.edge_endpoints()
    syn = cx.edges["n_syn"].to_numpy(np.int64)
    both = keep[pre] & keep[post]
    onto_dn = both & is_dn[post]
    ek = both & ~onto_dn
    glob = np.flatnonzero(keep)

    def es(m):
        return dict(pairs=int(m.sum()), synapses=int(syn[m].sum()))

    asc = role == "ascending"
    sc = n["super_class"].astype(object).where(n["super_class"].notna(), "NA")
    info = dict(
        definition=dict(keep_regions=list(KEEP_REGIONS), plus_super_class=["motor", "descending"] + list(ASCENDING),
                        dn_inputs_removed=True),
        whole=dict(neurons=int(cx.n_neurons), **es(np.ones(len(pre), bool))),
        kept=dict(neurons=int(keep.sum()), **es(ek)),
        kept_by_reason=dict(region=int((np.isin(reg, KEEP_REGIONS) & (role == "region")).sum()),
                            **{r: int((role == r).sum()) for r in ("motor", "descending", "ascending")}),
        kept_by_region={r: int((keep & (reg == r)).sum()) for r in ("VNC", "neck", "SEZ", "brain", "NA")},
        kept_by_super_class={k: int(v) for k, v in sc[keep].value_counts().items()},
        removed=dict(neurons=int((~keep).sum()),
                     by_region={r: int((~keep & (reg == r)).sum()) for r in ("VNC", "neck", "SEZ", "brain", "NA")},
                     edges_onto_dn_from_kept=es(onto_dn),
                     edges_removed_into_kept=es(~keep[pre] & keep[post]),
                     edges_removed_from_kept=es(keep[pre] & ~keep[post]),
                     edges_among_removed=es(~keep[pre] & ~keep[post])),
        ascending=dict(neurons=int(asc.sum()),
                       by_region={r: int((asc & (reg == r)).sum()) for r in ("VNC", "neck", "SEZ", "brain", "NA")},
                       inputs_kept=es(ek & asc[post]), inputs_removed=es(~keep[pre] & asc[post]),
                       outputs_kept=es(ek & asc[pre]), outputs_removed=es(asc[pre] & ~keep[post])),
        descending=dict(neurons=int(is_dn.sum()),
                        by_region={r: int((is_dn & (reg == r)).sum()) for r in ("VNC", "neck", "SEZ", "brain", "NA")},
                        outputs_kept=es(ek & is_dn[pre]), outputs_removed=es(is_dn[pre] & ~keep[post])),
        motor=dict(neurons=int((role == "motor").sum()),
                   by_region={r: int(((role == "motor") & (reg == r)).sum()) for r in ("VNC", "neck", "SEZ", "brain", "NA")},
                   inputs_kept=es(ek & (role == "motor")[post]), inputs_removed=es(~keep[pre] & (role == "motor")[post])),
    )
    meta = dict(cx.meta, global_index=glob, subgraph=dict(name="banc_vnc_subgraph_prereg08", **info["definition"]))
    sub = Connectome(cx.name, n.iloc[glob].reset_index(drop=True), cx.edges[ek].reset_index(drop=True), meta)
    return sub, info


def main(argv=None):
    import f3a_vnc_run as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    cx = F.load()
    sub, info = vnc_subgraph(cx)
    info["graph"] = cx.meta.get("data_source")
    info["check_sub"] = dict(neurons=int(sub.n_neurons), pairs=int(len(sub.edges)), synapses=int(sub.edges["n_syn"].sum()))
    out.write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("whole", "kept", "kept_by_reason", "kept_by_region")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
