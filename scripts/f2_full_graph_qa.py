"""Data-only QA of the full BANC v888 graph (data source ``banc888_full``; no simulation).

    env -u PYTHONPATH python scripts/f2_full_graph_qa.py --out docs/results/f2_full_graph_qa

For the primary neuron set (the 158,262 Codex neurons, edges among them, no threshold) and the sensitivity set (all
188,508 rows of the v888 meta table): pairs, synapses, pairs per threshold, and the FlyWire / BANC ratio of the mean
input synapses per postsynaptic neuron per super class (``f2_wsyn_k.ratio_table``, matched cell types and all
neurons), next to the Codex export as it was used so far. Markdown and JSON are written by this script.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import qa_report as Q  # noqa: E402
from connectome.adapters import banc, flywire783 as fw  # noqa: E402
from f2_wsyn_k import BANC_SC, CLASSES, in_synapses, ratio_table  # noqa: E402

B = ROOT / "banc"
NEURONS, CONN = B / "neurons-banc.csv.gz", B / "connections_princeton-banc.csv.gz"
EDGES, META = B / "banc_888_edgelist_simple_v2.feather", B / "banc_888_meta.feather"


def view(cx):
    return dict(neurons=int(cx.n_neurons), **Q.pair_threshold_profile(cx, (1, 3, 5)),
                median_syn_per_pair=float(cx.edges["n_syn"].median()),
                mean_in_edges=float(np.bincount(cx.edge_endpoints()[1], minlength=cx.n_neurons).mean()),
                mean_in_syn=float(in_synapses(cx).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out.with_suffix(".md").exists() or out.with_suffix(".json").exists()) and not a.overwrite:
        raise SystemExit(f"{out}.* exists")
    fx = fw.load_connectome(hash_files=False)
    fsc = fx.neurons["super_class"].astype(object).to_numpy()
    ftype = fx.neurons["cell_type"].astype(object).to_numpy()
    R = dict(flywire=view(fx), banc=dict())
    graphs = {"codex_export_ge3": banc.load_connectome(NEURONS, CONN, hash_files=False),
              "banc888_full_codex": banc.load_connectome_full(NEURONS, EDGES, META, CONN, "codex", hash_files=False),
              "banc888_full_all": banc.load_connectome_full(NEURONS, EDGES, META, CONN, "all", hash_files=False)}
    for name, cx in graphs.items():
        bsc = cx.neurons["super_class"].replace(BANC_SC).astype(object).to_numpy()
        btype = cx.neurons["cell_type"].astype(object).to_numpy()
        R["banc"][name] = dict(**view(cx), k=ratio_table(fx, cx, fsc, bsc, ftype, btype, 0))
        _, post = cx.edge_endpoints()
        sp = Q.sparseness(cx, bsc)
        R["banc"][name]["in_edges_by_class"] = {c: sp[c]["in_edges_mean"] for c in CLASSES if c in sp}
    sp = Q.sparseness(fx, fsc)
    R["flywire"]["in_edges_by_class"] = {c: sp[c]["in_edges_mean"] for c in CLASSES if c in sp}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1))
    f = R["flywire"]
    L = ["# F2 prep: full BANC v888 graph, data-only QA (`scripts/f2_full_graph_qa.py`)", "",
         "Source: `banc_888_edgelist_simple_v2.feather` (Harvard Dataverse doi:10.7910/DVN/7WTH1N, synapse set v2 at "
         "`size >= 5`, pair level, no count threshold, autapses excluded) and `banc_888_meta.feather`. Codex Root IDs are "
         "the v888 ids (all 158,262 are in the meta table). Numbers are written by the script.", "",
         "## 1. Graph size", "",
         "| graph | neurons | pairs | synapses | median syn/pair | pairs >= 3 | pairs >= 5 | mean in-edges | mean in-syn |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = [("FlyWire v783 (Shiu)", f)] + [(k, v) for k, v in R["banc"].items()]
    for name, v in rows:
        L.append(f"| {name} | {v['neurons']:,} | {v['pairs']:,} | {v['synapses']:,} | {v['median_syn_per_pair']:.0f} | "
                 f"{v['ge3']['pairs']:,} | {v['ge5']['pairs']:,} | {v['mean_in_edges']:.1f} | {v['mean_in_syn']:.1f} |")
    for view_name, label in (("matched", "matched cell types"), ("all", "all neurons of the class")):
        L += ["", f"## 2{'a' if view_name == 'matched' else 'b'}. FlyWire / BANC ratio of mean input synapses per "
              f"postsynaptic neuron, {label}", "",
              "| class | " + " | ".join(R["banc"]) + " |", "|---|" + "---|" * len(R["banc"])]
        for c in ("__all__",) + CLASSES:
            cells = []
            for k, v in R["banc"].items():
                r = v["k"][view_name][c]
                cells.append("n/a" if r["k"] is None else f"{r['k']:.2f} ({r['n_banc']:,})")
            L.append(f"| {'all' if c == '__all__' else c} | " + " | ".join(cells) + " |")
    L += ["", "Values are FlyWire / BANC (number of BANC neurons in the class). The BANC super classes are mapped as "
          "optic_lobe_intrinsic -> optic, central_brain_intrinsic -> central; ascending and motor include VNC "
          "compartments that FlyWire (brain only) lacks, so their ratios are read with the brain-neuropil caveat of "
          "`f2_wsyn_k.json`.", ""]
    out.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
