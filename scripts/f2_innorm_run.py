"""Arms R3 and N of prereg/05_banc_v3_and_innorm.md: sanity gates + MN9 per side.

    env -u PYTHONPATH python scripts/f2_innorm_run.py --arm R3   --out runs/f2/r3_v3.json
    env -u PYTHONPATH python scripts/f2_innorm_run.py --arm N_v2 --out runs/f2/n_v2.json
    env -u PYTHONPATH python scripts/f2_innorm_run.py --arm N_v3 --out runs/f2/n_v3.json

R3: the reference arm (Shiu unchanged) on the v3 full graph. N_v2 / N_v3: per-neuron input normalisation
(``connectome.innorm``) on the v2 / v3 full graph. Gate code and protocol are those of ``f2_wsyn_calibrate.variant_report``
with gate 1 seeds 0-4, gate 2 seeds 200-249, run seeds 0-9 (prereg/05 sections 3-4).
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
from connectome import innorm, qa_report as Q  # noqa: E402
from connectome.adapters import banc, flywire783 as fw  # noqa: E402
from engine.lif_brian2 import Brian2LIFEngine  # noqa: E402
from f2_wsyn_calibrate import peak_rss_gb, variant_report  # noqa: E402

B = ROOT / "banc"
EDGES = {"v2": B / "banc_888_edgelist_simple_v2.feather", "v3": B / "banc_888_edgelist_simple_v3.feather"}
ARMS = {"R3": ("v3", False), "N_v2": ("v2", True), "N_v3": ("v3", True)}
GATE1_SEEDS, GATE2_SEEDS, RUN_SEEDS = tuple(range(5)), tuple(range(200, 250)), tuple(range(10))


def setup(version):
    cx = banc.load_connectome_full(B / "neurons-banc.csv.gz", EDGES[version], B / "banc_888_meta.feather",
                                   B / "connections_princeton-banc.csv.gz", neuron_set="codex", hash_files=True,
                                   data_source="banc888_full" if version == "v2" else "banc888_full_v3")
    n = cx.neurons
    sugar = n["function"].astype(str).str.contains("sugar") & n["super_class"].isin(["sensory", "sensory_ascending"])
    lab = sugar & (n["body_part"] == "labellum")
    leg = sugar & n["body_part"].isin(["front_leg", "middle_leg", "hind_leg"])
    groups = {}
    for nm, m in (("labellar", lab), ("leg", leg)):
        for sd in ("left", "right"):
            groups[f"{nm}_{sd[0].upper()}"] = np.flatnonzero((m & (n["side"] == sd)).to_numpy())
    return cx, groups, np.flatnonzero((n["cell_type"] == "MN9").to_numpy())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    version, normalise = ARMS[a.arm]
    t0 = time.time()
    cx, groups, mn9 = setup(version)
    res = dict(stage="prereg05", arm=a.arm, edge_list=version, normalised=normalise, n_neurons=cx.n_neurons,
               pairs=int(len(cx.edges)), synapses=int(cx.edges["n_syn"].sum()),
               graph_meta={k: v for k, v in cx.meta.items() if k != "sign_rules"},
               groups={k: len(v) for k, v in groups.items()}, mn9=int(len(mn9)),
               seeds=dict(gate1=list(GATE1_SEEDS), gate2=[GATE2_SEEDS[0], GATE2_SEEDS[-1]], run=list(RUN_SEEDS)))
    s = None
    if normalise:
        fx = fw.load_connectome(hash_files=False)
        s, res["innorm"] = innorm.innorm_factors(fx, cx)
        del fx
    eng = Brian2LIFEngine(cx, input_groups=groups, groups={"mn9": mn9}, seed=0)
    res["model_sha256"] = eng.model_sha256
    t = Q.type_symmetry(cx)
    res["mn9_symmetry"] = t[t["type"] == "MN9"].drop(columns=["super_class"]).to_dict("records")
    if normalise:
        eng.set_weight_scale_per_post(s)
    else:
        eng.set_weight_scale(1.0)
    res["variants"] = {a.arm: variant_report(eng, cx, groups, mn9, a.arm, gate1_seeds=GATE1_SEEDS,
                                             gate2_seeds=GATE2_SEEDS, run_seeds=RUN_SEEDS)}
    res["peak_rss_gb"] = peak_rss_gb()
    res["total_wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done", round(res["total_wall_s"]), "s ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
