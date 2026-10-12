"""F0 multi-seed acceptance table (seeds 0-4 x latency 0 / 1) from runs/f0 -> docs/results/f0_multiseed.{json,md}.

Per flight: S1 / S2, MN9 rate and feeding fraction after landing (steps from the landing step on), mean network
rate over the closed loop, the network rate in the last cut step and ignition (> 1 Hz there, as in
prereg/02_ignition_frequency.md), the maximum position difference to the v1 n1 record. Seed 3 records predate the
REPLAY relabel (same code and inputs); they are kept as they are, labelled ignited.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks.f0_flight import criteria  # noqa: E402

RUNS = ROOT / "runs" / "f0"


def name(seed, lat, prefix=None):
    if prefix:
        return prefix.format(seed=seed, lat=lat)
    if seed == 3:
        return f"f0_lat{lat}"
    return f"f0_lat{lat}_seed{seed}"


def row(seed, lat, prefix=None):
    p = RUNS / f"{name(seed, lat, prefix)}_data.h5"
    if not p.exists():
        return None
    c = criteria(p)
    rep = json.loads((RUNS / f"{name(seed, lat, prefix)}_report.json").read_text())
    with h5py.File(p, "r") as f:
        b, m = f["behavior"], f["meta"].attrs
        k0 = c["landed_step"] or 0
        persist = float(m["persist_net_rate_hz"])
        return dict(seed=seed, latency=lat, S1=c["S1"], S2=c["S2"], landed_step=c["landed_step"],
                    first_feeding_step=c["first_feeding_step"],
                    mn9_after_landing_hz=float(np.nanmean(b["mn9_rate"][k0:])),
                    feeding_fraction=float(np.mean(b["is_feeding"][k0:])),
                    network_rate_hz=float(np.nanmean(b["net_rate"][:])), last_cut_step_hz=persist,
                    ignited=bool(persist > 1.0), pos_diff_to_v1_max_mm=rep["vs_v1"]["pos_diff_mm"]["max"],
                    wall_s=float(m["wall_s"]), interval_ms=float(m["interval_ms"]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="run name template with {seed} and {lat}")
    ap.add_argument("--latencies", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(ROOT / "docs/results/f0_multiseed"))
    a = ap.parse_args()
    rows = [r for s in range(5) for lat in a.latencies if (r := row(s, lat, a.prefix))]
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    lines = ["| seed | latency | S1 | S2 | MN9 after landing (Hz) | feeding fraction | network (Hz) | last cut step (Hz) | ignited | max pos diff to v1 (mm) | wall (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['latency']} | {'pass' if r['S1'] else 'FAIL'} | {'pass' if r['S2'] else 'FAIL'} | "
                     f"{r['mn9_after_landing_hz']:.1f} | {r['feeding_fraction']:.2f} | {r['network_rate_hz']:.2f} | "
                     f"{r['last_cut_step_hz']:.3f} | {'yes' if r['ignited'] else 'no'} | {r['pos_diff_to_v1_max_mm']:.3f} | "
                     f"{r['wall_s']:.0f} |")
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
