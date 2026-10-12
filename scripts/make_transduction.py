"""Write the FlyVis transduction constants (a0, a_ref) once; they are then fixed (CLAUDE.md rule 13).

Replaces synaptera@c94796d:scripts/make_t45_transduction.py and make_visual_transduction.py.

    env -u PYTHONPATH python scripts/make_transduction.py --kind t45|boundary --out data/<name>.json \
        [--interval-ms 25] [--subframes 2] [--compare data/<v1 file>.json] [--overwrite]

--compare checks the result against an existing file with a numeric tolerance (review M7: the bytes differ
between platforms) and exits non-zero on a mismatch. An existing --out is never overwritten without --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sense import vision_boundary as VB  # noqa: E402
from sense import vision_flyvis as V  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("t45", "boundary"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=float, default=25.0)
    ap.add_argument("--subframes", type=int, default=V.DEFAULT_SUBFRAMES)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists; use --overwrite or a new name")
    interval = a.interval_ms * 1e-3
    d = (V.calibrate_transduction(interval_s=interval, subframes=a.subframes) if a.kind == "t45"
         else VB.calibrate(interval_s=interval, subframes=a.subframes))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1))
    print(f"{out}: {a.kind}, dt {d['dt'] * 1e3:.2f} ms")
    if a.compare:
        bad = V.compare_transduction(d, json.loads(Path(a.compare).read_text()))
        print("compare:", "equal within tolerance" if not bad else f"{len(bad)} mismatches, e.g. {bad[:5]}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
