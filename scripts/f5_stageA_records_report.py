"""F5 step 2c report: gate 5 (prereg/17 section 3), worker RAM, and technical checks of a few training-seed records.

    env -u PYTHONPATH python scripts/f5_stageA_records_report.py          # writes docs/results/f5_stageA_records.{json,md}

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


def completion(plan, manifest):
    """Manifest completeness (every hash recomputed from the file bytes; test-seed records are hashed, never opened),
    input-check totals (test seeds: pass / fail only) and the run history from the logs."""
    import re
    files = {f"brain/{p.name}" for p in (F.OUT / "brain").iterdir()}
    pf = {e["file"] for e in plan}
    mism = [f for f, h in manifest.items() if sha(F.OUT / f) != h]
    ic = json.loads((F.OUT / "input_checks.json").read_text())
    split = {k: ("train" if k.startswith("p0_") else F.split_of(int(k[1:4]))) for k in ic}
    icn = {sp: dict(groups=sum(v == sp for v in split.values()),
                    passed=sum(ic[k]["passed"] for k in ic if split[k] == sp)) for sp in ("train", "val", "test")}
    icn["test"]["values_kept"] = sum("A_in_T4aT5a" in ic[k] for k in ic if split[k] == "test")
    prog = (F.OUT / "progress.log").read_text().splitlines()
    drv = (F.OUT / "driver.log").read_text().splitlines()
    starts = [dict(time=ln[:19], records_existing=int(re.search(r"(\d+) exist", ln).group(1)))
              for ln in prog if re.match(r"^\S+ \S+ start: ", ln)]
    kills = []
    for i, ln in enumerate(drv):
        if " KILL: " in ln:
            mon = next(x for x in reversed(drv[:i]) if " t=" in x)
            g = re.search(r"gpu_used=(\d+)MiB temp=(\d+)C power=([\d.]+)W", mon)
            wins = [float(re.search(r"windows_free=([\d.]+)GB", x).group(1)) for x in drv[max(0, i - 12):i]
                    if "windows_free=" in x]
            kills.append(dict(time=ln[:19], reason=ln.split(" KILL: ")[1], gpu_used_mib=int(g.group(1)),
                              gpu_temp_c=int(g.group(2)), gpu_power_w=float(g.group(3)), windows_free_before_gb=wins))
    notes = [dict(time=ln[:19], text=ln[20:]) for ln in prog if " stopped " in ln]
    done = json.loads((F.OUT / "DONE").read_text())
    return dict(n_plan=len(pf), n_manifest=len(manifest), n_files=len(files), plan_equals_manifest=pf == set(manifest),
                plan_equals_files=pf == files, hash_mismatches=len(mism), input_checks=icn,
                input_checks_total=len(ic), input_checks_passed=sum(v["passed"] for v in ic.values()),
                starts=starts, guard_kills=kills, stop_notes=notes, finished=done["finished"],
                tmp_files_left=len(list((F.OUT / "tmp").iterdir())))


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
        record_checks_all_ok=all(all(v for k, v in c.items() if k != "file") for c in checks),
        completion=completion(plan, manifest))


def summary(o):
    g = o["gate5"]
    return (f"Gate 5 (prereg/17): the extended brain script reproduces the machine-2 record rot_left seed 10 at 60 deg/s "
            f"and phase 0 {'bit for bit' if g['bit_identical'] else 'NOT bit for bit'} (render "
            f"{'identical' if g['render_bit_identical'] else 'different'}); the generation driver's record of the same "
            f"run is {'identical' if o['driver_record_equals_reference'] else 'different'}. Generation complete: "
            f"{o['completion']['n_manifest']:,} records in the SHA-256 manifest (all hashes recomputed, "
            f"{o['completion']['hash_mismatches']} mismatches); input checks {o['completion']['input_checks_passed']} "
            f"of {o['completion']['input_checks_total']} passed.")


def render(o):
    g, m = o["gate5"], o["measure"]
    L = ["# F5 step 2c: stage-A records", ""]
    L.append("Generated by `scripts/f5_stageA_records_report.py` from `runs/f5/` (not committed). "
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
    c = o["completion"]
    L.append("## 5. Generation result")
    L.append("")
    L.append(f"- Records: plan {c['n_plan']:,}, manifest {c['n_manifest']:,}, files {c['n_files']:,}; plan = manifest "
             f"{c['plan_equals_manifest']}, plan = files {c['plan_equals_files']}. Every SHA-256 recomputed from the "
             f"file bytes: {c['hash_mismatches']} mismatches. Temporary files left: {c['tmp_files_left']}. Finished "
             f"{c['finished'].replace('T', ' ')}.")
    ic = c["input_checks"]
    L.append(f"- Input checks (prereg/09, per render group and speed): {c['input_checks_passed']} of "
             f"{c['input_checks_total']} passed; train {ic['train']['passed']}/{ic['train']['groups']}, validation "
             f"{ic['val']['passed']}/{ic['val']['groups']}, test {ic['test']['passed']}/{ic['test']['groups']} (test: "
             f"pass / fail only, {ic['test']['values_kept']} values kept).")
    L.append("- Sealed test seeds 200-239: their records were written and hashed only, never opened, summarised or "
             "plotted; no record statistic was logged for any seed.")
    L.append("")
    L.append("Run history (starts from `progress.log`; guard stops from `driver.log`, with the guard's last monitor "
             "reading before each; every start resumed from the existing records and renders, nothing was lost):")
    L.append("")
    L.append("| time | event |")
    L.append("|---|---|")
    ev = [(x["time"], f"start, {x['records_existing']:,} records existing") for x in c["starts"]]
    ev += [(k["time"], f"guard stop: {k['reason']}; Windows free before (GB): "
            f"{', '.join(f'{w:.1f}' for w in k['windows_free_before_gb'])}; GPU at the last reading "
            f"{k['gpu_used_mib']:,} MiB, {k['gpu_temp_c']} C, {k['gpu_power_w']:.1f} W (this job: CUDA hidden)")
           for k in c["guard_kills"]]
    ev += [(x["time"], f"note: {x['text']}") for x in c["stop_notes"]]
    for t, e in sorted(ev):
        L.append(f"| {t} | {e} |")
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
