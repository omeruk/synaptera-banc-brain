"""Run F5 training jobs one at a time on the GPU; pause on heat or low memory and resume from the last checkpoint.

    env -u PYTHONPATH python scripts/f5_train_guard.py --log runs/f5/A/guard.log -- \\
        python scripts/f5_train.py run --arm true --seed 0 --lr 1e-3
    env -u PYTHONPATH python scripts/f5_train_guard.py --queue runs/f5/A/queue.txt     # one command per line

One GPU job at a time: the guard holds an exclusive lock (runs/f5/gpu.lock) for its whole life and starts a job only
when no other CUDA process is running (on machine 2, 4-5 CUDA processes at once put the NVIDIA driver 596.36 into an
nvlddmkm fault loop). Each job runs in a systemd scope with MemoryMax (--memory-max) and no swap.

Every 10 s the GPU temperature, memory, power, throttle reasons and the free RAM are appended to the log; every 30 s
the Windows free memory and the nvlddmkm events are read.
- **Pause** (the job's process group is stopped with SIGKILL; it has written atomic checkpoints every 25 steps):
  GPU temperature >= 87 C, free RAM < --kill-ram-gb, Windows free memory < --windows-min-free-gb, C: free <
  --disk-min-gb, or the file runs/f5/PAUSE exists (a manual pause). The guard then waits until the GPU is below
  70 C, RAM and Windows memory are 2 GB above their limits, C: is above its limit, PAUSE is gone and no other CUDA
  process runs, and starts the same command again; the trainer resumes from its latest checkpoint, bit-identically.
- **Stop** (no resume): a new nvlddmkm event (a driver fault), a job that exits non-zero on its own (a bug or an
  out-of-memory kill inside the scope), more than --max-pauses pauses of one job, or the --timeout of one job.
"""
import argparse
import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import f5_bench_guard as B  # noqa: E402

LOCK = ROOT / "runs" / "f5" / "gpu.lock"
PAUSE_FILE = ROOT / "runs" / "f5" / "PAUSE"
KILL_TEMP, RESUME_TEMP, START_TEMP = 87, 70, 75
POLL_S, EVENT_POLL_S, WAIT_POLL_S = 10, 30, 30
MARGIN_GB = 2.0
SCOPE = True                                    # systemd scope with MemoryMax (tests run without it)
LOG = ROOT / "runs" / "f5" / "A" / "guard.log"


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Limits:
    def __init__(self, a):
        self.kill_ram, self.win_min, self.disk_min = a.kill_ram_gb, a.windows_min_free_gb, a.disk_min_gb

    def pause_reason(self, g, ram, win=None, disk=None):
        """A reason to pause now, or None."""
        if g["temp"] >= KILL_TEMP:
            return f"GPU temperature {g['temp']} C >= {KILL_TEMP}"
        if ram < self.kill_ram:
            return f"free RAM {ram:.1f} GB < {self.kill_ram}"
        if win is not None and win == win and win < self.win_min:
            return f"Windows free memory {win:.1f} GB < {self.win_min}"
        if disk is not None and disk < self.disk_min:
            return f"C: free {disk:.1f} GB < {self.disk_min}"
        if PAUSE_FILE.exists():
            return f"manual pause ({PAUSE_FILE.relative_to(ROOT)} exists)"
        return None

    def wait_reason(self, g, ram, win, disk, temp_limit):
        """A reason not to (re)start yet, or None (with the margins of a resume)."""
        if g["temp"] >= temp_limit:
            return f"GPU {g['temp']} C >= {temp_limit}"
        if ram < self.kill_ram + MARGIN_GB:
            return f"free RAM {ram:.1f} GB < {self.kill_ram + MARGIN_GB}"
        if win == win and win < self.win_min + MARGIN_GB:
            return f"Windows free {win:.1f} GB < {self.win_min + MARGIN_GB}"
        if disk < self.disk_min:
            return f"C: free {disk:.1f} GB < {self.disk_min}"
        if PAUSE_FILE.exists():
            return "manual pause file present"
        procs = B.cuda_processes()
        if procs:
            return f"other CUDA processes: {procs}"
        return None


def wait_until_ok(lim, since, temp_limit):
    """Block until a job may start; returns False on a driver fault."""
    last = None
    while True:
        if B.nvlddmkm_errors_since(since) > 0:
            log("STOP: nvlddmkm event(s) since the guard started")
            return False
        g, ram, win, disk = B.gpu(), B.free_ram_gb(), B.windows_free_gb(), B.disk_free_gb()
        why = lim.wait_reason(g, ram, win, disk, temp_limit)
        if why is None:
            log(f"ok to start: GPU {g['temp']} C, {g['free_gib']:.2f} GiB free; RAM {ram:.1f} GB; Windows {win:.1f} GB; "
                f"C: {disk:.1f} GB")
            return True
        if why != last:
            log(f"waiting: {why}")
            last = why
        time.sleep(WAIT_POLL_S)


def run_job(cmd, a, lim, since):
    """Run one command to completion with pauses. Returns 0, or the code to stop the queue with."""
    pauses = 0
    temp_limit = START_TEMP
    t_job = time.time()
    while True:
        if not wait_until_ok(lim, since, temp_limit):
            return 4
        log(f"start (MemoryMax={a.memory_max}, pause #{pauses}): {' '.join(cmd)}")
        scope = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={a.memory_max}",
                 "-p", "MemorySwapMax=0", "--"] if SCOPE else []
        p = subprocess.Popen(scope + cmd, cwd=ROOT, start_new_session=True)
        t0, last_ev, reason, fault = time.time(), 0.0, None, False
        while p.poll() is None:
            time.sleep(POLL_S)
            ram, g = B.free_ram_gb(), B.gpu()
            disk = B.disk_free_gb()
            log(f"  t={time.time() - t0:6.0f}s ram_free={ram:.1f}GB gpu_used={g['used_mib']:.0f}MiB temp={g['temp']}C "
                f"power={g['power_w']}W throttle={g['throttle']} C_free={disk:.1f}GB")
            win = None
            if time.time() - last_ev > EVENT_POLL_S:
                last_ev = time.time()
                if B.nvlddmkm_errors_since(since) > 0:
                    reason, fault = "nvlddmkm event", True
                win = B.windows_free_gb()
                log(f"  windows_free={win:.1f}GB")
            if reason is None:
                reason = lim.pause_reason(g, ram, win, disk)
            if reason is None and time.time() - t_job > a.timeout:
                reason, fault = f"job timeout {a.timeout:.0f}s", True
            if reason and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                break
        if reason is None:
            log(f"end rc={p.returncode} after {time.time() - t0:.0f}s")
            return 0 if p.returncode == 0 else (p.returncode or 1)
        if fault:
            log(f"STOP: {reason}; the job is killed and not resumed")
            return 4
        pauses += 1
        log(f"PAUSE #{pauses}: {reason}; the job resumes from its last checkpoint when the limits recover")
        if pauses > a.max_pauses:
            log(f"STOP: more than {a.max_pauses} pauses")
            return 5
        temp_limit = RESUME_TEMP


def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None, help="file with one command per line (# comments); run in order")
    ap.add_argument("--timeout", type=float, default=7 * 86400, help="seconds per job, pauses included")
    ap.add_argument("--memory-max", default="14G")
    ap.add_argument("--kill-ram-gb", type=float, default=4.0)
    ap.add_argument("--windows-min-free-gb", type=float, default=6.0)
    ap.add_argument("--disk-min-gb", type=float, default=25.0)
    ap.add_argument("--max-pauses", type=int, default=50)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    LOG = Path(a.log)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if a.queue:
        jobs = [shlex.split(ln) for ln in Path(a.queue).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    elif cmd:
        jobs = [cmd]
    else:
        ap.error("no command and no --queue")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another GPU guard holds runs/f5/gpu.lock: one GPU job at a time", flush=True)
        return 2
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    since = datetime.now()
    lim = Limits(a)
    log(f"guard start: {len(jobs)} job(s); pause at GPU >= {KILL_TEMP} C, RAM < {a.kill_ram_gb} GB, Windows < "
        f"{a.windows_min_free_gb} GB, C: < {a.disk_min_gb} GB or {PAUSE_FILE.relative_to(ROOT)}; resume below "
        f"{RESUME_TEMP} C with {MARGIN_GB} GB margins")
    for i, job in enumerate(jobs):
        log(f"job {i + 1}/{len(jobs)}")
        rc = run_job(job, a, lim, since)
        if rc:
            log(f"queue stopped at job {i + 1} (rc {rc})")
            return rc
    log("queue done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
