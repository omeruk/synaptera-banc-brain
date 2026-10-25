"""Visualisation scaffold: per-neuron activity of one validation trial pair, for display only (no claim, no prereg).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_activity.py [brain|vnc|figure|all] [--overwrite]

Trials: rot_right and rot_left at 60 deg/s, validation data seed 160, grating phase as in scripts/f5_stageA_records.py
(`phase_of`). CPU only, one thread per numeric pool (as the stage-A workers).
- brain: the render (`f3b1_brain.render_stimulus`) and the brain run (`f3b3_brain.run_record`, unchanged) of the
  stage-A records; a proxy engine also keeps every neuron's cumulative count at each boundary. The DN columns must equal
  runs/f5/stageA/brain/<stimulus>_v060_s160.npz exactly (`dn_counts`, `net_counts`), else the script stops.
  -> runs/viz/activity_brain_<stimulus>.npz: counts (300, 138,639) int16 spikes per 5 ms interval, FlyWire model order.
- vnc: the stored DN counts through the `true` bridge (as in training) into the `true` and `shuffled` training-seed-0
  best.pt (selected rate); all 25,818 VNC states (VNC subgraph node order, = runs/viz/positions_vnc.npz) as rates in Hz,
  dA of both models and the teacher dA*. -> runs/viz/activity_vnc.npz
- figure: quick-check PNGs in the preview viz folder (activity_means.png, activity_dA.png).
Outputs are never overwritten without --overwrite (rule 13).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "" if _k == "CUDA_VISIBLE_DEVICES" else "1")      # as f5_stageA_records.WORKER_ENV
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUT = ROOT / "runs" / "viz"
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/viz")     # machine 2: viewing in Windows
STIMULI, SPEED, SEED = ("rot_right", "rot_left"), 60, 160
ARMS = ("true", "shuffled")
TRAINING_SEED = 0
FIGS = ("activity_means.png", "activity_dA.png")

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUE, SHUF = "#2a78d6", "#eb6834"
STIM_COL = {"rot_right": "#2a78d6", "rot_left": "#eb6834"}
STYLE = {"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
         "grid.linewidth": 0.8, "axes.axisbelow": True}


def brain_path(stim):
    return OUT / f"activity_brain_{stim}.npz"


def stored_path(stim):
    from train import f5_stageA as A
    return A.RECORDS / "brain" / A.record_name(stim, SPEED, SEED)


class Recorder:
    """Engine proxy for `f3b3_brain.run_record`: every call goes to the engine; run_coupled also keeps the cumulative
    spike count of every neuron at each boundary j > 0 (read just before run_record's own callback reads it)."""

    def __init__(self, eng):
        self._eng = eng
        self.cum = []

    def __getattr__(self, k):
        return getattr(self._eng, k)

    def run_coupled(self, n, interval_ms, on_boundary):
        self.cum = []

        def cb(j, e):
            if j > 0:
                self.cum.append(e.spike_counts())
            on_boundary(j, e)

        return self._eng.run_coupled(n, interval_ms, cb)


def run_brain(overwrite):
    todo = [s for s in STIMULI if overwrite or not brain_path(s).exists()]
    if not todo:
        print("brain: outputs exist (--overwrite to redo)", flush=True)
        return
    import torch
    torch.set_num_threads(1)
    from engine.lif_brian2 import Brian2LIFEngine
    from f3b1_brain import build, render_setup, render_stimulus
    from f3b3_brain import dn_table, run_record
    from f5_stageA_records import COND, WAVELENGTH_DEG, phase_of
    from train import f5_stageA as A
    A.check_seed(SEED)
    phase = phase_of(SEED)
    t0 = time.time()
    cx, groups_in, eyes = build(log=lambda *a, **k: None)
    rays, retina = render_setup()
    rates = {}
    for s in todo:
        t1 = time.time()
        rates[s] = render_stimulus(eyes, COND[s], rays, retina, tf_hz=SPEED / WAVELENGTH_DEG, phase_deg=phase)
        print(f"render {s} ({time.time() - t1:.0f} s)", flush=True)
    del eyes, rays, retina
    dn_idx, dn_tab = dn_table(cx)
    eng = Recorder(Brian2LIFEngine(cx, input_groups=groups_in, seed=0))
    for s in todo:
        t1 = time.time()
        L, R = (x.astype(float) for x in rates[s])
        rec = run_record(eng, s, SEED, L, R, dn_idx, dn_tab, "R")
        cum = np.array(eng.cum, np.int64)
        d = np.diff(np.concatenate([np.zeros((1, cum.shape[1]), np.int64), cum]), axis=0)
        st = A.open_record(stored_path(s))
        same = {k: bool(np.array_equal(rec[k], st[k])) for k in ("dn_counts", "net_counts", "dn_index")}
        same["full_dn_columns"] = bool(np.array_equal(d[:, dn_idx], st["dn_counts"]))
        same["full_net"] = bool(np.array_equal(d.sum(1), st["net_counts"]))
        print(f"{s}: {d.shape}, max count {d.max()}, equal to the stored record {same} ({time.time() - t1:.0f} s)",
              flush=True)
        if not all(same.values()):
            raise SystemExit(f"{s}: the recorded DN columns differ from {stored_path(s)}; stop")
        if d.max() > np.iinfo(np.int16).max:
            raise SystemExit(f"{s}: a count exceeds int16")
        np.savez_compressed(brain_path(s), counts=d.astype(np.int16), dn_index=dn_idx, stimulus=s, speed_deg_s=SPEED,
                            seed=SEED, phase_deg=phase, interval_ms=A.INTERVAL_MS, n_grey=A.N_GREY,
                            stored_record=str(stored_path(s).relative_to(ROOT)), equal_to_stored=json.dumps(same),
                            note="visualisation only; FlyWire model neuron order (runs/viz/positions_brain.npz)")
    print(f"brain done ({time.time() - t0:.0f} s)", flush=True)


def run_vnc(overwrite):
    path = OUT / "activity_vnc.npz"
    if path.exists() and not overwrite:
        print("vnc: output exists (--overwrite to redo)", flush=True)
        return
    import torch
    torch.set_num_threads(1)
    from engine import rate_torch as RT
    from f5_train import GRAPHS, load_setup, selected_lr
    from train import f5_stageA as A
    setup = load_setup()
    bridge = A.make_bridge(setup, "true", TRAINING_SEED)
    man = A.read_manifest(A.RECORDS / "manifest.sha256")
    units = np.zeros((A.T_TRIAL, len(STIMULI), len(bridge.units)), np.float32)
    for b, s in enumerate(STIMULI):
        p = stored_path(s)
        if man[str(p.relative_to(A.RECORDS))] != A.sha256_file(p):
            raise SystemExit(f"{p}: SHA-256 differs from the manifest")
        z = A.open_record(p)
        if not np.array_equal(z["dn_index"], setup.fw_index):
            raise SystemExit(f"{p}: DN columns differ from the setup")
        units[:, b] = bridge.unit_rates(z["dn_counts"]) / RT.R_IN
    out = dict(stimuli=np.array(STIMULI), speed_deg_s=SPEED, seed=SEED, interval_ms=A.INTERVAL_MS, n_grey=A.N_GREY,
               readout=np.asarray(setup.readout), r_in_hz=RT.R_IN,
               target=np.stack([A.target_series(s, SPEED) for s in STIMULI], 1),
               note="visualisation only; VNC subgraph node order (runs/viz/positions_vnc.npz)")
    check = {}
    for arm in ARMS:
        model, _ = A.make_model(arm, TRAINING_SEED, setup, bridge, GRAPHS)
        run_dir = A.run_dir_of(arm, TRAINING_SEED, selected_lr(arm), A.OUT)
        ck = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(units)
            st, _ = model.net.run(model.clamp(x), record=None)
            st = st.double().numpy()                                                 # (T, trials, n)
            dA = A.SCALE * (st[:, :, setup.readout] * np.asarray(setup.sigma, np.float64)).sum(2)
            dA_model = model(x).double().numpy()
        check[arm] = dict(run_dir=str(run_dir.relative_to(ROOT)), step=int(ck["step"]),
                          max_abs_dA_vs_forward=float(np.abs(dA - dA_model).max()))
        out[f"rates_hz_{arm}"] = (st * RT.R_IN).astype(np.float32)
        out[f"dA_{arm}"] = dA_model.astype(np.float32)
        print(f"{arm}: {run_dir.name} step {ck['step']}, states {st.shape}, |dA(all states) - forward| max "
              f"{check[arm]['max_abs_dA_vs_forward']:.2e}", flush=True)
    out["check"] = json.dumps(check)
    np.savez_compressed(path, **out)
    print(f"vnc done -> {path}", flush=True)


def figure(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train import f5_stageA as A
    v = np.load(OUT / "activity_vnc.npz")
    dt = float(v["interval_ms"]) * 1e-3
    t = (np.arange(A.T_TRIAL) - A.N_GREY) * float(v["interval_ms"])                # ms from stimulus onset
    stims = [str(s) for s in v["stimuli"]]
    rows = []
    for b, s in enumerate(stims):
        c = np.load(brain_path(s))["counts"]
        dn = np.load(brain_path(s))["dn_index"]
        rows.append(dict(brain=c.mean(1) / dt, dns=c[:, dn].mean(1) / dt, vnc=v["rates_hz_true"][:, b].mean(1),
                         mns=v["rates_hz_true"][:, b][:, v["readout"]].mean(1)))
    panels = (("brain", "FlyWire brain, all 138,639 neurons"), ("dns", f"FlyWire DNs ({len(dn):,})"),
              ("vnc", "BANC VNC subgraph, 25,818 neurons (true model)"), ("mns", "16 steering MNs (true model)"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        for ax, (k, title) in zip(axes, panels):
            for b, s in enumerate(stims):
                ax.plot(t, rows[b][k], color=STIM_COL[s], lw=1.4, label=s)
            ax.axvline(0, color=INK2, lw=0.8, ls=":")
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel("mean rate (Hz)")
        axes[0].legend(frameon=False, loc="upper left", ncol=2)
        axes[-1].set_xlabel("time from stimulus onset (ms); 5 ms intervals")
        fig.suptitle(f"Per-interval mean activity, {SPEED} deg/s, validation seed {SEED}. Visualisation only.",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / FIGS[0], dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(stims), figsize=(12, 4.2), sharey=True)
        for ax, (b, s) in zip(axes, enumerate(stims)):
            ax.plot(t, v["target"][:, b], color=INK, lw=1.6, ls="--", label="teacher dA*")
            ax.plot(t, v["dA_true"][:, b], color=TRUE, lw=1.4, label="true")
            ax.plot(t, v["dA_shuffled"][:, b], color=SHUF, lw=1.4, label="shuffled")
            ax.axvline(0, color=INK2, lw=0.8, ls=":")
            ax.set_title(s, fontsize=10)
            ax.set_xlabel("time from stimulus onset (ms)")
        axes[0].set_ylabel("dA (deg)")
        axes[0].legend(frameon=False, loc="best")
        fig.suptitle(f"dA against dA*, training-seed-0 best.pt, {SPEED} deg/s, validation seed {SEED}. "
                     "Visualisation only.", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / FIGS[1], dpi=140)
        plt.close(fig)
    for f in FIGS:
        print(out_dir / f, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all", choices=("brain", "vnc", "figure", "all"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=str(PREVIEW), help="folder of the quick-check PNGs")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("brain", "all"):
        run_brain(a.overwrite)
    if a.part in ("vnc", "all"):
        run_vnc(a.overwrite)
    if a.part in ("figure", "all"):
        figure(Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
