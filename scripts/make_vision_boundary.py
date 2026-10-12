"""Write the boundary-layer table and its validation (v1 --vision-boundary, Stage B); fixed before any behaviour.

Replaces synaptera@c94796d:scripts/make_vision_boundary.py; parametric by dataset through the adapters (here:
FlyWire v783 with the Codex ``consolidated_cell_types`` names; a BANC run supplies its own name map in F1).

    env -u PYTHONPATH python scripts/make_vision_boundary.py --codex-types ~/Downloads/consolidated_cell_types.csv.gz \
        --columns data/column_assignment.csv.gz --geometry-dir data --out-dir runs/boundary_v2 [--overwrite]

Outputs vision_boundary_table.csv (id, fw_type, flyvis_type, hemisphere, p, q, col_source, centroid_dist_um,
u, v, extent_dist) and vision_boundary_types.json. Existing files are never overwritten without --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome.adapters import flywire783 as fw  # noqa: E402
from sense import vision_boundary as VB  # noqa: E402
from sense import vision_flyvis as V  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", default=str(fw.BRAIN_DIR))
    ap.add_argument("--codex-types", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--geometry-dir", required=True, help="directory with neuron_arbor_centroids.npz and neuron_neuropil.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir).expanduser()
    path_table, path_types = out_dir / "vision_boundary_table.csv", out_dir / "vision_boundary_types.json"
    for p in (path_table, path_types):
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p} exists; use --overwrite or a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    cx = fw.load_connectome(a.brain_dir, hash_files=False)
    ct = VB.load_codex_types(cx.index.ids, a.codex_types)
    columns = V.load_columns(a.columns)
    geo = Path(a.geometry_dir)
    cen = np.load(geo / "neuron_arbor_centroids.npz")
    assert np.array_equal(cen["root_id"], cx.index.ids)
    npz = np.load(geo / "neuron_neuropil.npz")
    dom = npz["neuropils"][npz["dominant"]]
    net = V.load_flyvis_net()
    fv_types = list(dict.fromkeys(np.array(net.connectome.nodes.type[:]).astype(str)))

    res, missing = VB.boundary_fractions(ct, fv_types, cx)
    print(res.to_string())
    driven = res[res["driven"]]
    missing = [str(m) for m in missing]
    print(f"driven: {len(driven)} types, {int(driven['n'].sum())} neurons; FlyVis types not in the dataset: {missing}")
    tab, val = VB.build_table(cx, ct, dict(zip(driven.index, driven["flyvis_type"])), columns, cen["xyz"], dom)
    tab.to_csv(path_table, index=False)
    info = dict(rule=f"driven if share of outgoing synapses to non-FlyVis-matched neurons >= {VB.BOUNDARY_FRAC}",
                cell_types="Codex consolidated_cell_types primary_type", flyvis_model=V.FLYVIS_MODEL,
                flyvis_types_not_in_dataset=missing,
                types={t: dict(flyvis_type=r["flyvis_type"], n=int(r["n"]), syn_out=float(r["syn_out"]),
                               frac_out_nonflyvis=float(r["frac_out_nonflyvis"]), driven=bool(r["driven"]))
                       for t, r in res.iterrows()},
                n_driven_types=len(driven), n_driven_neurons=len(tab),
                centroid_sources=tab["col_source"].value_counts().to_dict(),
                column_rule="column_assignment; else partner_mean; else centroid",
                centroid_reference={k: list(v) for k, v in VB.CENTROID_REF.items()}, surface_degree=VB.SURFACE_DEG,
                centroid_validation=val)
    path_types.write_text(json.dumps(info, indent=1))
    print(f"{path_table}: {len(tab)} neurons, sources {info['centroid_sources']}")


if __name__ == "__main__":
    main()
