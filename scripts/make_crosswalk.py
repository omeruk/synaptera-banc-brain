"""Write the type crosswalk (FlyWire v783 <-> BANC) and, optionally, the FlyWire sugar-GRN homolog table.

Replaces synaptera@c94796d:scripts/make_sugar_grn.py (the homolog selection is
connectome.homologs.select_homologs; the crosswalk is connectome.crosswalk).

    env -u PYTHONPATH python scripts/make_crosswalk.py --out data/crosswalk_v1.csv \
        [--banc-neurons banc/neurons-banc.csv.gz] [--banc-connections banc/connections_princeton-banc.csv.gz] \
        [--sugar-out data/sugar_grn_783_v2.csv] [--overwrite]

Outputs are never overwritten without --overwrite (CLAUDE.md rule 13). Next to the crosswalk a
``<out>.meta.json`` records the input file hashes. With --banc-connections the MN9 candidate is also compared
with FlyWire CB0701 by partner-type profile (cross-dataset cosine on shared type names; evidence, not a rule).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome import crosswalk as cwk  # noqa: E402
from connectome import homologs  # noqa: E402
from connectome.adapters import banc, file_sha256, flywire783 as fw  # noqa: E402


def _refuse_overwrite(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; use --overwrite or a new name")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flywire-dir", default=str(fw.BRAIN_DIR))
    ap.add_argument("--banc-neurons", default=str(ROOT / "banc" / "neurons-banc.csv.gz"))
    ap.add_argument("--banc-connections", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sugar-out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    _refuse_overwrite(out, a.overwrite)
    if a.sugar_out:
        _refuse_overwrite(Path(a.sugar_out), a.overwrite)

    meta = dict(inputs={})
    flywire_cx = fw.load_connectome(a.flywire_dir) if (a.sugar_out or a.banc_connections) else None
    fly_neurons = flywire_cx.neurons if flywire_cx is not None else fw.load_neurons(a.flywire_dir)
    datasets = {"flywire783": fly_neurons}
    if flywire_cx is not None:
        meta["inputs"]["flywire783"] = flywire_cx.meta["sha256"]
    if Path(a.banc_neurons).exists():
        datasets["banc"] = banc.load_neurons(a.banc_neurons)
        meta["inputs"]["banc_neurons"] = file_sha256(a.banc_neurons)

    cw = cwk.crosswalk_by_name(datasets)
    cw.to_csv(out, index=False)
    print(cwk.crosswalk_summary(cw).to_string(index=False))

    if a.banc_connections and "banc" in datasets:
        bcx = banc.load_connectome_unsigned(a.banc_neurons, a.banc_connections)
        meta["inputs"]["banc_connections"] = bcx.meta["sha256"]["connections"]
        fa = cw[(cw["query"] == "MN9") & (cw["dataset"] == "flywire783")]["id"].to_numpy()
        fb = cw[(cw["query"] == "MN9") & (cw["dataset"] == "banc")]["id"].to_numpy()
        if len(fa) and len(fb):
            cos = homologs.cross_dataset_cosine(fa, flywire_cx, fb, bcx)
            meta["mn9_profile_cosine"] = dict(
                flywire_ids=fa.tolist(), banc_ids=fb.tolist(), cosine=cos.filter(like="cos_").to_dict(),
                shared_frac_flywire=cos["shared_frac_a"].tolist(), shared_frac_banc=cos.attrs["shared_frac_b"],
                n_shared_features=cos.attrs["n_shared_features"])
            print("MN9 (CB0701 <-> MN9) profile cosine:\n", cos.round(3).to_string())

    if a.sugar_out:
        df, loo = homologs.select_homologs(flywire_cx, cell_type="LB3", seed_ids=fw.SHIU_SUGAR_IDS,
                                           seed_source="shiu2023")
        df.to_csv(a.sugar_out, index=False, float_format="%.2f")
        meta["sugar_loo"] = loo
        print(df["source"].value_counts().to_dict(), "left LOO:", loo)

    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
