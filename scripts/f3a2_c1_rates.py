"""C1 of prereg/07: per-neuron rates of the seven DN families in the FlyWire brain, from existing records (no simulation).

    env -u PYTHONPATH python scripts/f3a2_c1_rates.py --out runs/f3a2/c1_rates.json

Sources (prereg/07 section 1): F0 records runs/f0/f0_5ms_seed{0..4}_data.h5 (all spikes; windows perch, flight,
landing) and the v1 visual DN screen records ~/Documents/NeuroFly/logs/vis_dn/main/vd_c{01..10}_s{601..605}.npz
(per-step 25 ms counts of all 1,299 DNs; window steps 28-59).
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome.adapters import flywire783 as fw  # noqa: E402
from f3_dn_crosswalk import family  # noqa: E402

FAMILIES = ("DNg02", "DNa02", "DNp31", "DNa04", "DNa10", "DNa15", "DNa16")
F0_SEEDS = tuple(range(5))
PERCH_S, CUT_S, IV_S = 0.25, 0.20, 0.005
WINDOWS = {"perch": None, "flight": (1, 2, 3, 9), "landing": (4, 8)}
VD_DIR = Path.home() / "Documents" / "NeuroFly" / "logs" / "vis_dn" / "main"
VD_SEEDS, VD_CONDS, VD_WIN = tuple(range(601, 606)), tuple(range(1, 11)), (28, 60)
VD_NAMES = {1: "yaw-CW", 2: "yaw-CCW", 3: "static", 4: "loom-L", 5: "loom-R", 6: "loom-front", 7: "recede-L",
            8: "recede-R", 9: "exp. flow", 10: "grey"}
BIN_MS = 25.0


def f0_rates(path, idx):
    """{window: (rate per neuron over the window, max 25 ms-bin rate per neuron)} for neuron indices idx."""
    with h5py.File(path, "r") as f:
        i, t = f["spikes/i"][:], f["spikes/t"][:].astype(np.float64)
        ph = f["behavior/phase"][:]
        meta = json.loads(f["meta"].attrs["coupling"])
    assert meta["interval_ms"] == 5.0 and meta["latency_intervals"] == 1, meta
    pos = {int(x): k for k, x in enumerate(idx)}
    keep = np.isin(i, idx)
    i, t = i[keep], t[keep]
    col = np.array([pos[int(x)] for x in i], dtype=np.int64)
    t0 = PERCH_S + CUT_S
    out = {}
    for w, codes in WINDOWS.items():
        if codes is None:
            spans = [(0.0, PERCH_S)]
        else:
            ks = np.flatnonzero(np.isin(ph, codes))
            # contiguous runs of intervals -> time spans
            br = np.flatnonzero(np.diff(ks) > 1)
            starts, ends = np.r_[ks[0], ks[br + 1]], np.r_[ks[br], ks[-1]]
            spans = [(t0 + s * IV_S, t0 + (e + 1) * IV_S) for s, e in zip(starts, ends)]
        dur = sum(b - a for a, b in spans)
        cnt = np.zeros(len(idx))
        peak = np.zeros(len(idx))
        for a, b in spans:
            m = (t >= a) & (t < b)
            np.add.at(cnt, col[m], 1)
            nb = int(round((b - a) * 1e3 / BIN_MS))
            if nb:
                h = np.zeros((len(idx), nb))
                bi = np.minimum(((t[m] - a) * 1e3 // BIN_MS).astype(int), nb - 1)
                np.add.at(h, (col[m], bi), 1)
                peak = np.maximum(peak, h.max(1) / (BIN_MS * 1e-3))
        out[w] = (cnt / dur, peak, dur)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    cx = fw.load_connectome(hash_files=False)
    n = cx.neurons
    ids = fw.load_root_ids()
    dn = (n["super_class"] == "descending").to_numpy()
    fam = n["cell_type"].astype(str).map(family).to_numpy()
    side = n["side"].astype(str).to_numpy()
    neurons = []
    for f in FAMILIES:
        for k in np.flatnonzero(dn & (fam == f)):
            neurons.append(dict(index=int(k), root_id=str(ids[k]), family=f, cell_type=str(n["cell_type"].iloc[k]),
                                side=side[k]))
    idx = np.array([x["index"] for x in neurons])
    res = dict(stage="prereg07_c1", families=FAMILIES, neurons=neurons, f0={}, vd={})
    for s in F0_SEEDS:
        p = ROOT / "runs" / "f0" / f"f0_5ms_seed{s}_data.h5"
        r = f0_rates(p, idx)
        res["f0"][str(s)] = {w: dict(rate_hz=v[0].tolist(), peak25_hz=v[1].tolist(), duration_s=v[2]) for w, v in r.items()}
        print("F0 seed", s, {w: round(float(np.median(v[0])), 2) for w, v in r.items()}, flush=True)
    # v1 visual DN screen
    for s in VD_SEEDS:
        for c in VD_CONDS:
            z = np.load(VD_DIR / f"vd_c{c:02d}_s{s}.npz")
            ridx = z["dn_root_idx"]
            pos = {int(x): j for j, x in enumerate(ridx)}
            cols = np.array([pos[int(k)] for k in idx])
            win = z["dn_counts"][VD_WIN[0]:VD_WIN[1]].astype(float)
            dur = (VD_WIN[1] - VD_WIN[0]) * BIN_MS * 1e-3
            res["vd"].setdefault(str(s), {})[VD_NAMES[c]] = dict(
                rate_hz=(win[:, cols].sum(0) / dur).tolist(), peak25_hz=(win[:, cols].max(0) / (BIN_MS * 1e-3)).tolist())
    res["vd_meta"] = dict(dir=str(VD_DIR), seeds=list(VD_SEEDS), window_steps=list(VD_WIN), window_s=0.8,
                          note="v1 records (Synaptera v1 engine and input path), not re-run")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
