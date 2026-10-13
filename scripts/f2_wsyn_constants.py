"""Write configs/banc_wsyn.yaml (the structural k constants, origin P) and print the k tables as markdown.

Reads docs/results/f2_wsyn_k.json (made by scripts/f2_wsyn_k.py; data only, no simulation). The fitted constant of
variant B is added after the calibration run by scripts/f2_wsyn_calibrate.py, not here.

    env -u PYTHONPATH python scripts/f2_wsyn_constants.py [--write-config]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = json.loads((ROOT / "docs/results/f2_wsyn_k.json").read_text())
CLASSES = ("sensory", "central", "optic", "visual_projection", "descending", "ascending", "motor")


def constants():
    k_global = K["as_run"]["matched"]["__all__"]["k"]
    k_class = {c: K["as_run_brain_neuropils"]["matched"][c]["k"] for c in CLASSES}
    return k_global, k_class


def table():
    L = ["| super class | n FlyWire / BANC (matched) | k as run, matched | k as run, brain neuropils, matched "
         "(A2) | k as run, all neurons | k both >= 3, matched | k both >= 5, matched |", "|---|---|---|---|---|---|---|"]
    for c in ("__all__",) + CLASSES:
        m = K["as_run"]["matched"][c]
        row = [("all" if c == "__all__" else c), f"{m['n_flywire']:,} / {m['n_banc']:,}",
               f"{m['k']:.2f}", f"{K['as_run_brain_neuropils']['matched'][c]['k']:.2f}",
               f"{K['as_run']['all'][c]['k']:.2f}", f"{K['both_ge3']['matched'][c]['k']:.2f}",
               f"{K['both_ge5']['matched'][c]['k']:.2f}"]
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    kg, kc = constants()
    if "--write-config" in argv:
        out = ROOT / "configs" / "banc_wsyn.yaml"
        if out.exists():
            raise SystemExit(f"{out} exists")
        lines = ["# BANC w_syn variants (prereg/03_banc_wsyn.md). Structural k = FlyWire / BANC mean input synapses per",
                 "# neuron on matched cell types; computed from data only by scripts/f2_wsyn_k.py (no simulation).",
                 "groups:",
                 "  - origin: P",
                 "    source: \"structural prior: docs/results/f2_wsyn_k.json, FlyWire v783 (Shiu graph, all pairs) / BANC export "
                 "(pairs >= 3), mean input synapses per postsynaptic neuron, matched cell types, as the engine runs them\"",
                 "    unit: \"ratio\"",
                 "    values:",
                 f"      wsyn_struct_k_global: {kg!r}",
                 "  - origin: P",
                 "    source: \"structural prior, per postsynaptic super class: as above, BANC input counted in brain and SEZ "
                 "neuropils only (FlyWire is brain only); neurons of other classes use wsyn_struct_k_global\"",
                 "    unit: \"ratio\"",
                 "    values:"]
        lines += [f"      wsyn_struct_class_k_{c}: {kc[c]!r}" for c in CLASSES]
        out.write_text("\n".join(lines) + "\n")
        print("wrote", out)
    print(table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
