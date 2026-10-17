"""Run one heavy benchmark job safely (F5 step 2b, after the 2026-10-17 freeze: 4-5 CUDA processes at once put the
NVIDIA driver 596.36 into an nvlddmkm "FECS Exception" loop and the machine had to be powered off).

    env -u PYTHONPATH python scripts/f5_bench_guard.py -- python scripts/f5_bench_run.py --part b_gpu ...

Before the job (refuses to start otherwise): free RAM >= 8 GB, free GPU memory >= 5 GiB, GPU temperature < 75 C, no
other CUDA process, no nvlddmkm error in the Windows System log in the last 10 minutes.
During the job: the job runs in a systemd scope with MemoryMax=14G and no swap; every 10 s the GPU temperature, memory,
power and throttle reasons and the free RAM are appended to runs/f5/bench/monitor.log. The job is killed on a new
nvlddmkm error, GPU temperature >= 87 C, free RAM < 4 GB, or the --timeout.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "f5" / "bench" / "monitor.log"
MIN_RAM_GB, MIN_GPU_GIB, MAX_TEMP_START = 8.0, 5.0, 75
KILL_RAM_GB, KILL_TEMP = 4.0, 87
MEMORY_MAX = "14G"
POLL_S, EVENT_POLL_S = 10, 30


def free_ram_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable")


def gpu():
    q = "temperature.gpu,memory.used,memory.total,power.draw,clocks_throttle_reasons.active"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, timeout=30).stdout.strip().split(", ")
    temp, used, total = int(out[0]), float(out[1]), float(out[2])
    return dict(temp=temp, used_mib=used, free_gib=(total - used) / 1024, power_w=out[3], throttle=out[4])


def cuda_processes():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=30).stdout
    return [ln for ln in out.split() if ln.strip()]


def nvlddmkm_errors_since(t: datetime) -> int:
    """Number of nvlddmkm events in the Windows System log since t (local time); -1 if the log cannot be read."""
    ps = ("$t=[datetime]::ParseExact('%s','yyyy-MM-dd HH:mm:ss',$null); "
          "@(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=$t} "
          "-ErrorAction SilentlyContinue).Count" % t.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=60).stdout.strip()
        return int(out.splitlines()[-1])
    except Exception:                                              # noqa: BLE001
        return -1


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def precheck():
    ram, g, procs = free_ram_gb(), gpu(), cuda_processes()
    ev = nvlddmkm_errors_since(datetime.fromtimestamp(time.time() - 600))
    problems = []
    if ram < MIN_RAM_GB:
        problems.append(f"free RAM {ram:.1f} GB < {MIN_RAM_GB}")
    if g["free_gib"] < MIN_GPU_GIB:
        problems.append(f"free GPU memory {g['free_gib']:.2f} GiB < {MIN_GPU_GIB}")
    if g["temp"] >= MAX_TEMP_START:
        problems.append(f"GPU temperature {g['temp']} C >= {MAX_TEMP_START}")
    if procs:
        problems.append(f"other CUDA processes running: {procs}")
    if ev != 0:
        problems.append(f"nvlddmkm events in the last 10 min: {ev}" if ev > 0 else "Windows System log not readable")
    log(f"precheck ram={ram:.1f}GB gpu_free={g['free_gib']:.2f}GiB temp={g['temp']}C cuda_procs={len(procs)} "
        f"nvlddmkm_10min={ev} -> {'OK' if not problems else 'REFUSED: ' + '; '.join(problems)}")
    return not problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=3600, help="seconds")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not precheck():
        return 2
    start = datetime.now()
    log(f"start (MemoryMax={MEMORY_MAX}, timeout={a.timeout:.0f}s): {' '.join(cmd)}")
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_MAX}",
                          "-p", "MemorySwapMax=0", "--"] + cmd, cwd=ROOT, start_new_session=True)
    t0, last_ev, reason = time.time(), 0.0, None
    while p.poll() is None:
        time.sleep(POLL_S)
        ram, g = free_ram_gb(), gpu()
        log(f"  t={time.time() - t0:6.0f}s ram_free={ram:.1f}GB gpu_used={g['used_mib']:.0f}MiB temp={g['temp']}C "
            f"power={g['power_w']}W throttle={g['throttle']}")
        if g["temp"] >= KILL_TEMP:
            reason = f"GPU temperature {g['temp']} C >= {KILL_TEMP}"
        elif ram < KILL_RAM_GB:
            reason = f"free RAM {ram:.1f} GB < {KILL_RAM_GB}"
        elif time.time() - t0 > a.timeout:
            reason = f"timeout {a.timeout:.0f}s"
        elif time.time() - last_ev > EVENT_POLL_S:
            last_ev = time.time()
            ev = nvlddmkm_errors_since(start)
            if ev > 0:
                reason = f"{ev} nvlddmkm event(s) since the start"
        if reason and p.poll() is None:
            log(f"KILL: {reason}")
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            break
    log(f"end rc={p.returncode} after {time.time() - t0:.0f}s" + (f" (killed: {reason})" if reason else ""))
    return 3 if reason else p.returncode


if __name__ == "__main__":
    sys.exit(main())
