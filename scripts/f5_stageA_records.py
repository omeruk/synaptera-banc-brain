"""F5 step 2c: the 2,400 stage-A brain records (prereg/14 section 7 with prereg/17), on machine 2, CPU only.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_stageA_records.py run --render-workers N --brain-workers M
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_stageA_records.py measure     # RAM of one worker of each
    env -u PYTHONPATH python scripts/f5_stageA_records.py manifest                          # SHA-256 of every record

Each record is a run of `scripts/f3b3_brain.py` (render `f3b1_brain.render_stimulus`, brain run
`f3b3_brain.run_record`, the same functions as the extended script) for one (stimulus, drum speed, grating phase, data
seed). Per data seed: rot_left / rot_right at 30, 60, 90, 120 deg/s, static and grey (10 records). Phase 0 for seeds
0-39, else `default_rng(seed).uniform(0, 30)` deg (one wavelength). A render depends on (stimulus, speed, phase) only:
grey is shared by every seed, and seeds 0-39 share their phase-0 renders.

Disk (the WSL disk lives on C:, ~41 MB per render, ~82 GB for all): renders are temporary. The seeds go in batches;
per batch: render (parallel processes) -> input check -> brain records (parallel processes) -> delete the batch's
renders. Temporary files stay under TMP_MAX_GB; the run stops if C: has less than MIN_DISK_GB free. Resumable: existing
renders and records are kept, files are written atomically. Stage B renders again (same code, bit for bit).

Sealed test seeds 200-239: their records are written and hashed, never opened, summarised or plotted. No record
statistic is logged for any seed. The input check (prereg/14 gate 5) is a property of the render: for test seeds only
pass / fail is kept.

Output: runs/f5/stageA/ (not committed): brain/<stimulus>[_v<speed>]_s<seed>.npz, plan.json, machine.json,
input_checks.json, progress.log, manifest.sha256; tmp/ (renders, deleted per batch).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUT = ROOT / "runs" / "f5" / "stageA"
BRAIN, TMP = OUT / "brain", OUT / "tmp"
SPEEDS = (30, 60, 90, 120)
WAVELENGTH_DEG = 30.0
TRAIN, VAL, TEST = range(0, 160), range(160, 200), range(200, 240)
SEEDS = range(0, 240)
PHASE0_SEEDS = range(0, 40)
BATCH_SEEDS = 20
RENDER_MB = 41.0                     # 300 x (17,185 + 16,936) float32, uncompressed
TMP_MAX_GB, MIN_DISK_GB = 15.0, 25.0
WINDOWS_MIN_FREE_GB = 6.0            # always left free for Windows
PEAK_GB = {"render": 2.9, "brain": 2.5}          # measured peak RSS per worker (runs/f5/stageA/measure), rounded up
SPAWN_MARGIN_GB, SPAWN_GAP_S = 1.0, 60           # a worker starts only if Windows keeps 6 + 1 GB after its peak
DISK = Path("/mnt/c")
COND = {"rot_left": 2, "rot_right": 1, "static": 3, "grey": 10}      # f3b1_common.STIMULI t0 conditions


def phase_of(seed):
    return 0.0 if seed in PHASE0_SEEDS else float(np.random.default_rng(seed).uniform(0.0, WAVELENGTH_DEG))


def split_of(seed):
    return "train" if seed in TRAIN else "val" if seed in VAL else "test"


def render_key(stim, speed, seed):
    """Name of the render a record needs: grey is shared; static has no speed; phase 0 renders are shared."""
    if stim == "grey":
        return "grey"
    ph = "p0" if seed in PHASE0_SEEDS else f"s{seed:03d}"
    return f"static_{ph}" if stim == "static" else f"{stim}_v{speed:03d}_{ph}"


def key_params(key):
    """(t0 condition, tf_hz, phase_deg) of a render key."""
    parts = key.split("_")
    stim = "grey" if key == "grey" else "static" if parts[0] == "static" else f"{parts[0]}_{parts[1]}"
    if stim == "grey":
        return COND["grey"], None, 0.0
    ph = parts[-1]
    phase = 0.0 if ph == "p0" else phase_of(int(ph[1:]))
    speed = 60 if stim == "static" else int(parts[2][1:])
    return COND[stim], speed / WAVELENGTH_DEG, phase


def record_name(stim, speed, seed):
    return f"{stim}_v{speed:03d}_s{seed:03d}.npz" if stim.startswith("rot") else f"{stim}_s{seed:03d}.npz"


def plan():
    recs = []
    for seed in SEEDS:
        for stim in ("rot_left", "rot_right"):
            for v in SPEEDS:
                recs.append((stim, v, seed))
        recs += [("static", 0, seed), ("grey", 0, seed)]
    return [dict(stimulus=st, speed_deg_s=v if st.startswith("rot") else None, seed=s, phase_deg=phase_of(s),
                 split=split_of(s), file=f"brain/{record_name(st, v, s)}", render=render_key(st, v, s))
            for st, v, s in recs]


def batches():
    out = [list(PHASE0_SEEDS)]
    rest = [s for s in SEEDS if s not in PHASE0_SEEDS]
    out += [rest[i:i + BATCH_SEEDS] for i in range(0, len(rest), BATCH_SEEDS)]
    return out


# ── machine, logging, resources ─────────────────────────────────────────────────────────────────────────────────────
def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(OUT / "progress.log", "a") as f:
        f.write(line + "\n")


def free_ram_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 2**20
    raise RuntimeError


def windows_free_gb():
    ps = "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,3)"
    try:
        return float(subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                    timeout=60).stdout.strip().splitlines()[-1])
    except Exception:                                              # noqa: BLE001
        return float("nan")


def disk_free_gb():
    return shutil.disk_usage(DISK).free / 1e9


def tmp_gb():
    return sum(p.stat().st_size for p in TMP.rglob("*") if p.is_file()) / 1e9 if TMP.exists() else 0.0


def machine():
    import torch

    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:                                          # noqa: BLE001
            return None
    cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
    env = ROOT / "env" / "neurofly_x86_64.yml"
    return dict(machine=platform.machine(), kernel=platform.release(), cpu=cpu[0], threads=len(cpu),
                ram_gib=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                   text=True).stdout.strip() or None,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_cuda_available=torch.cuda.is_available(),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, brian2=ver("brian2"),
                mujoco=ver("mujoco"), env_file=str(env.relative_to(ROOT)),
                env_file_sha256=hashlib.sha256(env.read_bytes()).hexdigest(),
                commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                uncommitted=subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True,
                                           text=True).stdout.splitlines())


def peak_rss_gb(field="VmHWM:"):
    for ln in open("/proc/self/status"):
        if ln.startswith(field):
            return int(ln.split()[1]) / 2**20
    return float("nan")


def drop_cache(path, sync=False):
    """Drop the page-cache pages of a file (WSL keeps file cache as VM memory that Windows cannot use)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ── workers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_brain():
    """Connectome, input groups and DN table of f3b1_brain.build, without the FlyVis eyes."""
    import pandas as pd

    from connectome.adapters import flywire783 as fw
    from f3b3_brain import dn_table
    from sense import vision_boundary as VB
    cx = fw.load_connectome(hash_files=True)
    sugar_ids = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    table = VB.load_table(ROOT / "data" / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    vg = VB.boundary_groups(cx, table)
    groups_in = {"sugar": cx.index.group(sugar_ids), "vbnd_L": vg["vbnd_L"], "vbnd_R": vg["vbnd_R"]}
    return cx, groups_in, *dn_table(cx)


def worker_render(keys, threads):
    import torch
    torch.set_num_threads(threads)
    from f3b1_brain import build, render_setup, render_stimulus
    import gc
    cx, groups_in, eyes = build(log=lambda *a, **k: None)
    del cx, groups_in                                  # the eyes keep what they need; frees most of the RAM
    gc.collect()
    meta = OUT / "boundary_meta.npz"
    if not meta.exists():
        tmp = meta.with_suffix(".tmp.npz")
        np.savez(tmp, tcode_L=eyes.tcode["L"], tcode_R=eyes.tcode["R"], fw_types=np.array(eyes.fw_types))
        tmp.rename(meta)
    rays, retina = render_setup()
    for key in keys:
        path = TMP / f"{key}.npz"
        if path.exists():
            continue
        t1 = time.time()
        cond, tf, ph = key_params(key)
        L, R = render_stimulus(eyes, cond, rays, retina, tf_hz=tf, phase_deg=ph)
        tmp = TMP / f"{key}.tmp.npz"
        np.savez(tmp, L=L, R=R)
        drop_cache(tmp, sync=True)
        tmp.rename(path)
        print(f"render {key} ({time.time() - t1:.1f} s, rss {peak_rss_gb('VmRSS:'):.2f} GB)", flush=True)
    print(json.dumps(dict(peak_rss_gb=peak_rss_gb(), rss_gb=peak_rss_gb("VmRSS:"), threads=threads)), flush=True)


def worker_brain(tasks):
    """tasks: list of [render key, [record entries]]; the render is loaded once per key."""
    from engine.lif_brian2 import Brian2LIFEngine
    from f3b3_brain import run_record
    cx, groups_in, dn_idx, dn_tab = build_brain()
    eng = Brian2LIFEngine(cx, input_groups=groups_in, seed=0)
    for key, recs in tasks:
        z = np.load(TMP / f"{key}.npz")
        L, R = z["L"].astype(float), z["R"].astype(float)
        drop_cache(TMP / f"{key}.npz")
        for r in recs:
            path = OUT / r["file"]
            if path.exists():
                continue
            t1 = time.time()
            rec = run_record(eng, r["stimulus"], r["seed"], L, R, dn_idx, dn_tab, "R")
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **rec)
            tmp.rename(path)
            print(f"record {r['file']} ({time.time() - t1:.1f} s)", flush=True)       # no record statistic (sealed)
    print(json.dumps(dict(peak_rss_gb=peak_rss_gb(), rss_gb=peak_rss_gb("VmRSS:"))), flush=True)


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def spawn(args, tag):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    logf = open(OUT / "workers" / f"{tag}.log", "a")
    return subprocess.Popen([sys.executable, __file__] + args, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)


def run_pool(jobs, kind):
    """Run the worker jobs in parallel and wait; stop on any failure or a resource limit. Workers start one at a time,
    SPAWN_GAP_S apart (their start-up peak), and only while Windows would keep WINDOWS_MIN_FREE_GB + SPAWN_MARGIN_GB
    free after the worker's peak; the other jobs wait for a free slot."""
    peak = PEAK_GB[kind.split("_")[0]]
    pending, procs, last = list(enumerate(jobs)), [], 0.0
    while pending or any(p.poll() is None for p in procs):
        if pending and time.time() - last >= SPAWN_GAP_S:
            win = windows_free_gb()
            running = sum(p.poll() is None for p in procs)
            if running == 0 or win - peak >= WINDOWS_MIN_FREE_GB + SPAWN_MARGIN_GB:
                i, args = pending.pop(0)
                procs.append(spawn(args, f"{kind}_{i}"))
                last = time.time()
                log(f"  {kind}: worker {i} started ({running + 1} running; Windows free {win:.1f} GB before)")
        time.sleep(15)
        failed = [i for i, p in enumerate(procs) if p.poll() not in (None, 0)]
        if failed:
            for p in procs:
                p.kill()
            raise SystemExit(f"{kind} workers {failed} failed (see runs/f5/stageA/workers/)")
        why = (f"C: below {MIN_DISK_GB} GB free" if disk_free_gb() < MIN_DISK_GB else
               f"Windows below {WINDOWS_MIN_FREE_GB} GB free RAM" if windows_free_gb() < WINDOWS_MIN_FREE_GB else None)
        if why:
            for p in procs:
                p.kill()
            raise SystemExit(f"{why}: stopped (resume by running again)")
    bad = [i for i, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"{kind} workers {bad} failed (see runs/f5/stageA/workers/)")


def split_even(items, n):
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def input_checks(batch, recs):
    """prereg/09 input check per (seed group, speed) of the batch's renders; for test seeds only pass / fail."""
    import f3b1_common as C1
    from f3b1_brain import input_check
    m = np.load(OUT / "boundary_meta.npz")
    tcode, fw_types = {"L": m["tcode_L"], "R": m["tcode_R"]}, [str(t) for t in m["fw_types"]]
    path = OUT / "input_checks.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    groups = sorted({("p0" if s in PHASE0_SEEDS else f"s{s:03d}", split_of(s)) for s in batch})
    for g, split in groups:
        for v in SPEEDS:
            name = f"{g}_v{v:03d}"
            if name in done:
                continue
            seed = 0 if g == "p0" else int(g[1:])
            keys = {st: TMP / f"{render_key(st, v, seed)}.npz" for st in C1.STIMULI}
            rates = {}
            for st, p in keys.items():
                z = np.load(p)
                rates[f"{st}__L"], rates[f"{st}__R"] = z["L"], z["R"]
            chk = input_check(rates, tcode, fw_types)
            done[name] = dict(passed=chk["passed"]) if split == "test" else dict(
                passed=chk["passed"], A_in_T4aT5a={k: chk["per_stimulus"][k]["A_in_T4aT5a"] for k in ("rot_left", "rot_right")})
            if not chk["passed"]:
                log(f"INPUT CHECK FAILED {name}")
    path.write_text(json.dumps(done, indent=1, sort_keys=True))
    return done


def npz_identical(a, b):
    """prereg/17 section 3: the same members, and identical uncompressed .npy bytes per member."""
    import zipfile
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        if sorted(za.namelist()) != sorted(zb.namelist()):
            return False
        return all(za.read(m) == zb.read(m) for m in za.namelist())


def update_manifest():
    path = OUT / "manifest.sha256"
    have = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            h, f = ln.split("  ", 1)
            have[f] = h
    for p in sorted(q for q in BRAIN.glob("*.npz") if ".tmp" not in q.name):
        f = f"brain/{p.name}"
        if f not in have:
            have[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    path.write_text("".join(f"{have[f]}  {f}\n" for f in sorted(have)))
    return len(have)


def run(a):
    for d in (BRAIN, TMP, OUT / "workers"):
        d.mkdir(parents=True, exist_ok=True)
    P = plan()
    if not (OUT / "plan.json").exists():
        (OUT / "plan.json").write_text(json.dumps(dict(n_records=len(P), records=P), indent=1))
    elif json.loads((OUT / "plan.json").read_text())["records"] != P:
        raise SystemExit("plan.json differs from the plan of this script")
    (OUT / "machine.json").write_text(json.dumps(machine(), indent=1))
    log(f"start: {len(P)} records, {sum((OUT / r['file']).exists() for r in P)} exist; render workers "
        f"{a.render_workers} x {a.threads} threads, brain workers {a.brain_workers}; C: free {disk_free_gb():.1f} GB, "
        f"RAM free {free_ram_gb():.1f} GB, Windows free {windows_free_gb():.1f} GB")
    for bi, batch in enumerate(batches()):
        recs = [r for r in P if r["seed"] in batch]
        todo = [r for r in recs if not (OUT / r["file"]).exists()]
        if not todo:
            continue
        keys = sorted({r["render"] for r in todo})
        need = [k for k in keys if not (TMP / f"{k}.npz").exists()]
        if disk_free_gb() < MIN_DISK_GB:
            raise SystemExit(f"C: below {MIN_DISK_GB} GB free: stopped")
        if tmp_gb() + len(need) * RENDER_MB / 1e3 > TMP_MAX_GB:
            raise SystemExit(f"batch {bi} would exceed {TMP_MAX_GB} GB of temporary files")
        t0 = time.time()
        log(f"batch {bi} (seeds {batch[0]}-{batch[-1]}): {len(todo)} records, {len(need)} renders to make")
        if need:
            run_pool([["worker-render", "--threads", str(a.threads), "--keys", ",".join(ks)]
                      for ks in split_even(need, a.render_workers)], f"render_b{bi}")
        log(f"batch {bi}: renders done ({time.time() - t0:.0f} s); tmp {tmp_gb():.1f} GB, C: free {disk_free_gb():.1f} GB, "
            f"Windows free {windows_free_gb():.1f} GB")
        chk = input_checks(batch, recs)
        n_fail = sum(not v["passed"] for v in chk.values())
        log(f"batch {bi}: input checks {len(chk)} so far, failed {n_fail}")
        if bi == 0 and a.preview and not (OUT / "preview" / "DONE").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "f5_stageA_preview.py"), "--copy-to", a.preview],
                           cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=False)
        tasks = {}
        for r in todo:
            tasks.setdefault(r["render"], []).append(r)
        items = sorted(tasks.items())
        jobs = []
        for i, ch in enumerate(split_even(items, a.brain_workers)):
            tf = OUT / "workers" / f"tasks_b{bi}_{i}.json"
            tf.write_text(json.dumps(ch))
            jobs.append(["worker-brain", "--tasks", str(tf)])
        run_pool(jobs, f"brain_b{bi}")
        missing = [r["file"] for r in recs if not (OUT / r["file"]).exists()]
        if missing:
            raise SystemExit(f"batch {bi}: {len(missing)} records missing")
        for k in {r["render"] for r in recs} - {"grey"}:
            (TMP / f"{k}.npz").unlink(missing_ok=True)
        n = update_manifest()
        log(f"batch {bi} done ({time.time() - t0:.0f} s): {n} records in the manifest; tmp {tmp_gb():.1f} GB, "
            f"C: free {disk_free_gb():.1f} GB")
    (TMP / "grey.npz").unlink(missing_ok=True)
    n = update_manifest()
    (OUT / "DONE").write_text(json.dumps(dict(n_records=n, finished=datetime.now().isoformat(timespec="seconds"))))
    log(f"all done: {n} records")


def measure(a):
    """RAM and time of one render worker (one phase-0 render, compared with the machine-2 F3b-3 render) and one brain
    worker (one training-seed record), each alone."""
    out = OUT / "measure"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for threads in a.threads_list:
        key = "rot_left_v060_p0"
        if (TMP / f"{key}.npz").exists():
            (TMP / f"{key}.npz").unlink()
        TMP.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "worker-render", "--threads", str(threads), "--keys", key],
                           cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
        info = json.loads(r.stdout.strip().splitlines()[-1])
        z = np.load(TMP / f"{key}.npz")
        ref = np.load(ROOT / "runs" / "machine_x86" / "f3b3_rot_left_s10" / "stim_rates.npz")
        same = bool(np.array_equal(z["L"], ref["rot_left__L"]) and np.array_equal(z["R"], ref["rot_left__R"]))
        res[f"render_threads{threads}"] = dict(wall_s=time.time() - t0, peak_rss_gb=info["peak_rss_gb"],
                                               rss_after_gb=info["rss_gb"],
                                               equals_machine2_f3b3_render=same,
                                               render_s=float(r.stdout.split("(")[-1].split(" s")[0])
                                               if "render " in r.stdout else None)
        print(threads, res[f"render_threads{threads}"], flush=True)
    rec = dict(stimulus="rot_left", seed=10, file="measure/rot_left_v060_s010.npz")
    (out / "rot_left_v060_s010.npz").unlink(missing_ok=True)
    tf = out / "tasks.json"
    tf.write_text(json.dumps([["rot_left_v060_p0", [rec]]]))
    t0 = time.time()
    r = subprocess.run([sys.executable, __file__, "worker-brain", "--tasks", str(tf)],
                       cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    res["brain"] = dict(wall_s=time.time() - t0, peak_rss_gb=info["peak_rss_gb"], rss_after_gb=info["rss_gb"],
                        record_s=float(r.stdout.split("(")[-1].split(" s")[0]),
                        equals_machine2_f3b3_record=npz_identical(
                            out / "rot_left_v060_s010.npz",
                            ROOT / "runs" / "machine_x86" / "f3b3_rot_left_s10" / "brain" / "rot_left_s10.npz"))
    print(res["brain"], flush=True)
    (out / "measure.json").write_text(json.dumps(res, indent=1))
    (TMP / "rot_left_v060_p0.npz").unlink()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "measure", "manifest", "worker-render", "worker-brain"))
    ap.add_argument("--render-workers", type=int, default=2)
    ap.add_argument("--brain-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per render worker")
    ap.add_argument("--threads-list", type=lambda s: [int(x) for x in s.split(",")], default=[1])
    ap.add_argument("--keys")
    ap.add_argument("--tasks")
    ap.add_argument("--preview", default=None, help="Windows folder (as /mnt/c/...) for the preview PNGs")
    a = ap.parse_args(argv)
    if a.mode == "worker-render":
        worker_render(a.keys.split(","), a.threads)
    elif a.mode == "worker-brain":
        worker_brain(json.loads(Path(a.tasks).read_text()))
    elif a.mode == "measure":
        measure(a)
    elif a.mode == "manifest":
        print(update_manifest(), "records")
    else:
        run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
