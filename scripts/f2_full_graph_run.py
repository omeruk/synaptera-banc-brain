"""Reference arm on the full BANC v888 graph (prereg/04_banc_full_graph.md): sanity gates + MN9 per side.

    env -u PYTHONPATH python scripts/f2_full_graph_run.py --neuron-set codex --out runs/f2/banc888_full_codex.json
    env -u PYTHONPATH python scripts/f2_full_graph_run.py --neuron-set all   --out runs/f2/banc888_full_all.json

Shiu parameters unchanged (w_syn = 0.275 mV, no scale); the protocol, seeds and gate code are those of
``scripts/f2_wsyn_calibrate.py`` (``variant_report``): gate 1 (no input, seeds 0-4), gate 2 (ignition after the cut,
seeds 200-219), run list (labellar, leg, labellar-left, labellar-right; seeds 0-2).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import qa_report as Q  # noqa: E402
from connectome.adapters import banc  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb, variant_report  # noqa: E402

B = ROOT / "banc"


def setup(neuron_set):
    cx = banc.load_connectome_full(B / "neurons-banc.csv.gz", B / "banc_888_edgelist_simple_v2.feather",
                                   B / "banc_888_meta.feather", B / "connections_princeton-banc.csv.gz",
                                   neuron_set=neuron_set, hash_files=True)
    n = cx.neurons
    sugar = n["function"].astype(str).str.contains("sugar") & n["super_class"].isin(["sensory", "sensory_ascending"])
    lab = sugar & (n["body_part"] == "labellum")
    leg = sugar & n["body_part"].isin(["front_leg", "middle_leg", "hind_leg"])
    groups = {}
    for nm, m in (("labellar", lab), ("leg", leg)):
        for sd in ("left", "right"):
            groups[f"{nm}_{sd[0].upper()}"] = np.flatnonzero((m & (n["side"] == sd)).to_numpy())
    mn9 = np.flatnonzero((n["cell_type"] == "MN9").to_numpy())
    return cx, groups, mn9


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuron-set", choices=banc.NEURON_SETS, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    cx, groups, mn9 = setup(a.neuron_set)
    eng = Brian2LIFEngine(cx, input_groups=groups, groups={"mn9": mn9}, seed=0)
    res = dict(stage="banc888_full", neuron_set=a.neuron_set, n_neurons=eng.n_neurons, pairs=int(len(cx.edges)),
               synapses=int(cx.edges["n_syn"].sum()), graph_meta={k: v for k, v in cx.meta.items() if k != "sign_rules"},
               groups={k: len(v) for k, v in groups.items()}, mn9=int(len(mn9)), model_sha256=eng.model_sha256)
    t = Q.type_symmetry(cx)
    res["mn9_symmetry"] = t[t["type"] == "MN9"].drop(columns=["super_class"]).to_dict("records")
    eng.set_weight_scale(1.0)
    res["variants"] = dict(R=variant_report(eng, cx, groups, mn9, "R"))
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
