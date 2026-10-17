"""F5 step 2c report: gate 5 (prereg/17 section 3), worker RAM, and technical checks of a few training-seed records.

    env -u PYTHONPATH python scripts/f5_stageA_report.py          # writes docs/results/f5_stageA_records.{json,md}

Inputs (not committed): runs/f5/stageA_gate5 (the extended brain script, rot_left seed 10, 60 deg/s, phase 0),
runs/machine_x86/f3b3_rot_left_s10 (the machine-2 record of `scripts/f3b3_brain.py`), runs/f5/stageA/measure,
runs/f5/stageA (plan, input checks, manifest, the records of CHECK_SEEDS). Only fixed inputs are read, so the report
does not change while the generation goes on. Technical properties only: no record statistic is computed, and no test
seed (200-239) record is opened. Refuses to overwrite without --overwrite.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_records as F  # noqa: E402

GATE = ROOT / "runs" / "f5" / "stageA_gate5"
REF = ROOT / "runs" / "machine_x86" / "f3b3_rot_left_s10"
OUT_JSON = ROOT / "docs" / "results" / "f5_stageA_records.json"
OUT_MD = ROOT / "docs" / "results" / "f5_stageA_records.md"
CHECK_SEEDS = (0, 1, 39)                     # training seeds of batch 0 (phase 0)
MEMBERS = sorted(["stimulus", "t0_condition", "seed", "dn_counts", "net_counts", "input_mean_L", "input_mean_R",
                  "interval_ms", "weights_sha256", "model_sha256", "model_arm", "dn_index", "dn_cell_type", "dn_side"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def members(p):
    with zipfile.ZipFile(p) as z:
        return sorted(m.removesuffix(".npy") for m in z.namelist())


def tech_check(entry, manifest, ref_members, ref_dn):
    """File integrity and record format of one training-seed record; no statistic of its content."""
    p = F.OUT / entry["file"]
    assert F.split_of(entry["seed"]) != "test"
    z = np.load(p, allow_pickle=True)            # the project's own records (object DN-type columns)
    return dict(file=entry["file"], sha256_in_manifest=manifest.get(entry["file"]) == sha(p),
                zip_test_ok=zipfile.ZipFile(p).testzip() is None, members_as_f3b3=members(p) == ref_members,
                stimulus_seed_match=str(z["stimulus"]) == entry["stimulus"] and int(z["seed"]) == entry["seed"],
                shapes_ok=z["dn_counts"].shape == ref_dn["dn_counts"].shape and z["dn_counts"].dtype == np.int16
                and z["net_counts"].shape == (300,),
                dn_table_as_f3b3=all(np.array_equal(z[k], ref_dn[k]) for k in ("dn_index", "dn_cell_type", "dn_side")),
                model_as_f3b3=all(str(z[k]) == str(ref_dn[k]) for k in ("weights_sha256", "model_sha256", "model_arm")))


def build():
    gate_rec, ref_rec = GATE / "brain" / "rot_left_s10.npz", REF / "brain" / "rot_left_s10.npz"
    meas = json.loads((F.OUT / "measure" / "measure.json").read_text())
    plan = json.loads((F.OUT / "plan.json").read_text())["records"]
    manifest = dict(reversed(ln.split("  ", 1)) for ln in (F.OUT / "manifest.sha256").read_text().splitlines())
    ref_dn = np.load(ref_rec, allow_pickle=True)
    checks = [tech_check(e, manifest, members(ref_rec), ref_dn) for e in plan if e["seed"] in CHECK_SEEDS]
    ic = json.loads((F.OUT / "input_checks.json").read_text())
    ic0 = {k: v for k, v in ic.items() if k.startswith("p0_")}
    return dict(
        prereg="prereg/14 section 7 and section 9 gate 5, with prereg/17 sections 2-3",
        gate5=dict(record=str(gate_rec.relative_to(ROOT)), reference=str(ref_rec.relative_to(ROOT)),
                   members_identical=members(gate_rec) == members(ref_rec),
                   bit_identical=F.npz_identical(gate_rec, ref_rec),
                   sha256=dict(record=sha(gate_rec), reference=sha(ref_rec)),
                   render_bit_identical=F.npz_identical(GATE / "stim_rates.npz", REF / "stim_rates.npz"),
                   input_check_passed=json.loads((GATE / "input_check.json").read_text())["passed"]),
        driver_record_equals_reference=meas["brain"]["equals_machine2_f3b3_record"],
        measure=meas,
        plan=dict(n_records=len(plan), n_renders=len({e["render"] for e in plan}),
                  per_split={s: sum(e["split"] == s for e in plan) for s in ("train", "val", "test")},
                  n_batches=len(F.batches()), batch_seeds=F.BATCH_SEEDS),
        input_checks_batch0={k: v for k, v in sorted(ic0.items())},
        record_checks=checks,
        record_checks_all_ok=all(all(v for k, v in c.items() if k != "file") for c in checks))


def summary(o):
    g = o["gate5"]
    return (f"Gate 5 (prereg/17): the extended brain script reproduces the machine-2 record rot_left seed 10 at 60 deg/s "
            f"and phase 0 {'bit for bit' if g['bit_identical'] else 'NOT bit for bit'} (render "
            f"{'identical' if g['render_bit_identical'] else 'different'}); the generation driver's record of the same "
            f"run is {'identical' if o['driver_record_equals_reference'] else 'different'}.")


def render(o):
    g, m = o["gate5"], o["measure"]
    L = ["# F5 step 2c: stage-A records", ""]
    L.append("Generated by `scripts/f5_stageA_report.py` from `runs/f5/` (not committed). "
             f"{o['prereg']}. Technical properties only: no record statistic is computed, and no test-seed record "
             "is opened.")
    L.append("")
    L.append(f"- {summary(o)}")
    L.append("")
    L.append("## 1. Gate 5")
    L.append("")
    L.append(f"- Record `{g['record']}` against `{g['reference']}`: same members {g['members_identical']}, "
             f"identical uncompressed member bytes {g['bit_identical']}; SHA-256 `{g['sha256']['record'][:16]}` and "
             f"`{g['sha256']['reference'][:16]}`.")
    L.append(f"- Render identical: {g['render_bit_identical']}; prereg/09 input check passed: {g['input_check_passed']}.")
    L.append(f"- The generation driver (`scripts/f5_stageA_records.py`: separate render and brain processes, one "
             f"thread per numeric pool: torch, numba, OpenBLAS, OpenMP) gives the same record: {o['driver_record_equals_reference']}.")
    L.append("")
    L.append("## 2. Worker RAM and time (each alone, CPU, CUDA hidden, one thread per numeric pool)")
    L.append("")
    L.append("| worker | peak RSS GiB | wall s | work s | output equals the machine-2 F3b-3 output |")
    L.append("|---|---|---|---|---|")
    for k, v in m.items():
        if k.startswith("render"):
            L.append(f"| render | {v['peak_rss_gb']:.2f} | "
                     f"{v['wall_s']:.0f} | {v['render_s']:.1f} (one render) | {v['equals_machine2_f3b3_render']} |")
    b = m["brain"]
    L.append(f"| brain | {b['peak_rss_gb']:.2f} | {b['wall_s']:.0f} | {b['record_s']:.1f} (one record) | "
             f"{b['equals_machine2_f3b3_record']} |")
    L.append("")
    p = o["plan"]
    L.append("## 3. Plan")
    L.append("")
    L.append(f"- {p['n_records']:,} records (train {p['per_split']['train']:,}, validation {p['per_split']['val']:,}, "
             f"test {p['per_split']['test']:,}) from {p['n_renders']:,} distinct renders, in {p['n_batches']} batches "
             f"(seeds 0-39, then {p['batch_seeds']} seeds each).")
    L.append("")
    L.append("## 4. Input checks of batch 0 (phase 0) and technical record checks")
    L.append("")
    L.append("| render group | passed | A_in T4a/T5a rot_left | rot_right |")
    L.append("|---|---|---|---|")
    for k, v in o["input_checks_batch0"].items():
        L.append(f"| {k} | {v['passed']} | {v['A_in_T4aT5a']['rot_left']:.4f} | {v['A_in_T4aT5a']['rot_right']:.4f} |")
    L.append("")
    L.append(f"Records of training seeds {', '.join(map(str, CHECK_SEEDS))} ({len(o['record_checks'])} files): "
             "SHA-256 equals the manifest, zip test, members as the F3b-3 record, stimulus and seed fields, shapes and "
             f"dtypes, DN table and model hashes as the F3b-3 record. All passed: {o['record_checks_all_ok']}.")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if (OUT_JSON.exists() or OUT_MD.exists()) and not a.overwrite:
        raise SystemExit("output exists; use --overwrite")
    o = build()
    OUT_JSON.write_text(json.dumps(o, indent=1) + "\n")
    OUT_MD.write_text(render(o))
    print(summary(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
