"""Regenerate the sign-variant tables of FlyWire v783 (silent / impute / literature) from the Codex transmitter
predictions. Replaces synaptera@c94796d:scripts/make_nt_silent.py, make_nt_impute.py, make_nt_literature.py
(the rules live in connectome/sign_rules.py).

    env -u PYTHONPATH python scripts/make_sign_tables.py --codex-neurons ~/Downloads/neurons.csv.gz \
        --out-dir data/sign_tables_v2 [--literature-types types.csv] [--overwrite]

Writes nt_modulatory_silent_783.csv, nt_impute_783.csv, nt_literature_783.csv (same columns as v1; ``root_id``
for ``id``). Existing files are never overwritten without --overwrite (CLAUDE.md rule 13).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome import sign_rules as SR  # noqa: E402
from connectome.adapters import flywire783 as fw  # noqa: E402


def build(codex_neurons, brain_dir=fw.BRAIN_DIR, literature_types=None) -> dict:
    cx = fw.load_connectome(brain_dir, codex_neurons=codex_neurons, hash_files=False)
    nt = cx.neurons
    missing = nt["nt"].isna()
    sel = nt["nt"].isin(SR.NT_MODULATORY) | missing
    silent = pd.DataFrame({"root_id": nt.loc[sel, "id"], "nt_type": nt.loc[sel, "nt"].fillna("none"),
                           "nt_type_score": nt.loc[sel, "nt_conf"]}).reset_index(drop=True)
    SR.silent_rule(cx, "broad")            # also validates (missing prediction must have score 0)
    impute = SR.impute_table(cx).rename(columns={"id": "root_id"})
    literature = SR.literature_table(cx, types=literature_types)
    return dict(nt_modulatory_silent_783=silent, nt_impute_783=impute, nt_literature_783=literature)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-neurons", default=str(Path("~/Downloads/neurons.csv.gz").expanduser()))
    ap.add_argument("--brain-dir", default=str(fw.BRAIN_DIR))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--literature-types", default=None, help="CSV with a cell_type column (default: all types)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = {n: out / f"{n}.csv" for n in ("nt_modulatory_silent_783", "nt_impute_783", "nt_literature_783")}
    for p in targets.values():
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p} exists; use --overwrite or a new --out-dir")
    types = None if a.literature_types is None else pd.read_csv(a.literature_types)["cell_type"]
    for name, df in build(a.codex_neurons, a.brain_dir, types).items():
        df.to_csv(targets[name], index=False, float_format="%.2f")
        print(f"{targets[name]}: {len(df)} rows")


if __name__ == "__main__":
    main()
