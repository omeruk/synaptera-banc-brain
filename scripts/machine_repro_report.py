"""Machine change aarch64 M1 -> x86_64 (docs/05 section 7.1): reproduction of `runs/f3b3/brain/rot_left_s10.npz`.

    env -u PYTHONPATH python scripts/f3b3_brain.py --out runs/machine_x86/f3b3_rot_left_s10 --seeds 10-10
        (stopped after rot_left s10 was written; no other brain record generated)
    env -u PYTHONPATH python scripts/machine_repro_report.py     # writes docs/results/machine_x86_repro.{json,md}

The prereg/14 records check (section 9, gate 5) asks for a bit-for-bit reproduction. This compares the new record and the
render it was driven by (stim_rates.npz) with the committed M1 ones, array by array. Refuses to overwrite without
--overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OLD = ROOT / "runs" / "f3b3"
NEW = ROOT / "runs" / "machine_x86" / "f3b3_rot_left_s10"
RECORD = Path("brain") / "rot_left_s10.npz"
OUT_JSON = ROOT / "docs" / "results" / "machine_x86_repro.json"
OUT_MD = ROOT / "docs" / "results" / "machine_x86_repro.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(a_path, b_path):
    """Per array key: identical, and for numeric arrays that differ the size of every difference."""
    A, B = np.load(a_path, allow_pickle=True), np.load(b_path, allow_pickle=True)
    out = dict(only_committed=sorted(set(A.files) - set(B.files)), only_new=sorted(set(B.files) - set(A.files)), keys={})
    for k in sorted(set(A.files) & set(B.files)):
        x, y = A[k], B[k]
        same = x.dtype == y.dtype and x.shape == y.shape and (
            x.tolist() == y.tolist() if x.dtype == object else x.tobytes() == y.tobytes())
        r = dict(identical=bool(same), dtype=str(x.dtype), shape=list(x.shape))
        if not same:
            u, v = x.astype(np.float64), y.astype(np.float64)
            d = np.abs(v - u)
            rows = np.flatnonzero((d.reshape(len(d), -1) > 0).any(1))
            r.update(n_different=int((d > 0).sum()), n_total=int(d.size), max_abs_diff=float(d.max()),
                     sum_committed=float(u.sum()), sum_new=float(v.sum()), first_different_row=int(rows[0]))
            if x.dtype.kind == "f":
                both = (u > 0) & (v > 0)
                rel = d[both] / np.maximum(u[both], v[both])
                flip = (u == 0) != (v == 0)
                r.update(zero_nonzero_flips=int(flip.sum()),
                         max_rate_at_flip=float(np.maximum(u[flip], v[flip]).max()) if flip.any() else 0.0,
                         max_rel_diff_both_positive=float(rel.max()), median_rel_diff_differing=float(np.median(rel[rel > 0])))
        out["keys"][k] = r
    return out


def build():
    import f3b3_common as C
    a, b = np.load(OLD / RECORD, allow_pickle=True), np.load(NEW / RECORD, allow_pickle=True)
    lo, hi = C.WINDOW
    old_chk, new_chk = (json.loads((p / "input_check.json").read_text()) for p in (OLD, NEW))
    return dict(
        prereg="prereg/14_f5_training.md section 9, gate 5 (records)",
        script="scripts/f3b3_brain.py --seeds 10-10, stopped after rot_left s10",
        machine=json.loads((NEW / "machine.json").read_text()),
        run_commit=(NEW / "commit.txt").read_text().strip(),
        run_uncommitted=(NEW / "status.txt").read_text().splitlines(),
        sha256=dict(committed_record=sha(OLD / RECORD), new_record=sha(NEW / RECORD),
                    committed_render=sha(OLD / "stim_rates.npz"), new_render=sha(NEW / "stim_rates.npz")),
        bit_identical=sha(OLD / RECORD) == sha(NEW / RECORD),
        window=[lo, hi], interval_ms=float(C.INTERVAL_MS),
        window_dn_spikes=[int(a["dn_counts"][lo:hi].sum()), int(b["dn_counts"][lo:hi].sum())],
        first_different_interval=dict(
            net=int(np.flatnonzero(a["net_counts"] != b["net_counts"])[0]),
            dn=int(np.flatnonzero((a["dn_counts"] != b["dn_counts"]).any(1))[0]),
            input=int(np.flatnonzero(a["input_mean_L"] != b["input_mean_L"])[0])),
        input_check=dict(passed=[old_chk["passed"], new_chk["passed"]],
                         equals_f3b1_render=[old_chk["equals_f3b1_render"], new_chk["equals_f3b1_render"]]),
        record=compare(OLD / RECORD, NEW / RECORD),
        render=compare(OLD / "stim_rates.npz", NEW / "stim_rates.npz"))


def summary(o):
    r, k = o["record"]["keys"], o["render"]["keys"]
    dn, net = r["dn_counts"], r["net_counts"]
    rmax = max(x["max_abs_diff"] for x in k.values() if not x["identical"])
    flips = sum(x["zero_nonzero_flips"] for x in k.values() if not x["identical"])
    fmax = max(x["max_rate_at_flip"] for x in k.values() if not x["identical"])
    f = o["first_different_interval"]
    return (f"rot_left seed 10 is {'bit-identical' if o['bit_identical'] else 'not bit-identical'} on x86_64: "
            f"window DN spikes {o['window_dn_spikes'][0]:,} (M1) vs {o['window_dn_spikes'][1]:,}; "
            f"{dn['n_different']:,} of {dn['n_total']:,} DN interval counts differ (max |diff| {dn['max_abs_diff']:.0f}); "
            f"network spikes {net['sum_committed']:,.0f} vs {net['sum_new']:,.0f} "
            f"({100 * (net['sum_new'] - net['sum_committed']) / net['sum_committed']:+.3f} %); first difference at "
            f"interval {f['net']} (network) and {f['dn']} (DNs). The FlyVis render already differs from interval "
            f"{f['input']}: max |diff| {rmax:.2e} Hz, {flips} zero/non-zero flips at rates <= {fmax:.2e} Hz.")


def fmt(v):
    return f"{v:.3e}" if isinstance(v, float) else str(v)


def render(o):
    m = o["machine"]
    L = ["# Machine change: reproduction of runs/f3b3/brain/rot_left_s10.npz on x86_64", ""]
    L.append("Generated by `scripts/machine_repro_report.py` from `runs/f3b3/` (M1 record) and "
             "`runs/machine_x86/f3b3_rot_left_s10/` (new run; not committed). Check of "
             f"{o['prereg']}. Run: `{o['script']}`; descriptive, no decision depends on it.")
    L.append("")
    L.append(f"- Machine: {m['machine']}, {m['cpu']} ({m['threads']} threads), {m['ram_gib']} GiB, {m['gpu']}; "
             f"{m['os']} (WSL2 kernel {m['kernel']}). Python {m['python']}, torch {m['torch']}, numpy {m['numpy']}, "
             f"Brian2 {m['brian2']}, MuJoCo {m['mujoco']}, FlyVis {m['flyvis']} on {m['flyvis_device']}.")
    L.append(f"- Run from commit `{o['run_commit'][:12]}` plus the uncommitted changes of the machine-change branch: "
             + ", ".join(f"`{s.strip()}`" for s in o["run_uncommitted"]) + ".")
    L.append(f"- SHA-256 record: M1 `{o['sha256']['committed_record'][:16]}`, x86_64 `{o['sha256']['new_record'][:16]}`; "
             f"render: M1 `{o['sha256']['committed_render'][:16]}`, x86_64 `{o['sha256']['new_render'][:16]}`.")
    L.append(f"- Input check passed (M1, x86_64): {o['input_check']['passed']}; render equals the F3b-1 render: "
             f"{o['input_check']['equals_f3b1_render']}.")
    L.append("")
    L.append("**Summary.** " + summary(o))
    L.append("")
    for name in ("record", "render"):
        c = o[name]
        L.append(f"## {'Record brain/rot_left_s10.npz' if name == 'record' else 'Render stim_rates.npz'}")
        L.append("")
        if c["only_committed"] or c["only_new"]:
            L.append(f"Keys only in the M1 file: {c['only_committed']}; only in the new file: {c['only_new']} "
                     "(the script gained the field after the M1 run).")
            L.append("")
        cols = ["n_different", "n_total", "max_abs_diff", "sum_committed", "sum_new", "first_different_row"]
        if name == "render":
            cols += ["zero_nonzero_flips", "max_rate_at_flip", "max_rel_diff_both_positive", "median_rel_diff_differing"]
        L.append("| key | dtype | shape | identical | " + " | ".join(cols) + " |")
        L.append("|---|---|---|---|" + "---|" * len(cols))
        for k, x in c["keys"].items():
            L.append(f"| {k} | {x['dtype']} | {'x'.join(map(str, x['shape'])) or '-'} | {x['identical']} | "
                     + " | ".join(fmt(x[c2]) if c2 in x else "" for c2 in cols) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("report exists (use --overwrite)")
    o = build()
    OUT_JSON.write_text(json.dumps(o, indent=1))
    OUT_MD.write_text(render(json.loads(OUT_JSON.read_text())))
    print(summary(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
