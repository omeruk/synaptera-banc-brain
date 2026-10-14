"""Brain stage of prereg/11_f3b3_specificity.md (section 2): fresh FlyWire v783 runs, brain seeds 10-29.

    env -u PYTHONPATH python scripts/f3b3_brain.py --out runs/f3b3

Render, input check and engine are those of `scripts/f3b1_brain.py` (prereg/09 sections 3-4). The render is redone;
whether it equals runs/f3b1/stim_rates.npz bit for bit is recorded in OUT/input_check.json (descriptive).
Per (stimulus, seed) a fresh state, per-interval spike counts of all FlyWire DNs -> OUT/brain/<stimulus>_s<seed>.npz
(existing files are skipped).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import f3b3_common as C  # noqa: E402
from f3b1_brain import MIN_FREE_GB, TRANSDUCTION, build, free_gb, input_check, peak_rss_gb, render  # noqa: E402

F3B1_STIM = ROOT / "runs" / "f3b1" / "stim_rates.npz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "f3b3"))
    ap.add_argument("--arm", default="R", choices=("R", "A", "AC", "ACG"),
                    help="neuron model arm (prereg/12; R = the unmodified reference of prereg/11)")
    ap.add_argument("--seeds", default=None, help="first-last brain seeds (default: prereg/11 seeds 10-29)")
    ap.add_argument("--stim-from", default=None,
                    help="directory with stim_rates.npz + input_check.json of an earlier render to reuse (copied)")
    a = ap.parse_args(argv)
    seeds = C.SEEDS if a.seeds is None else tuple(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1))
    out = Path(a.out)
    (out / "brain").mkdir(parents=True, exist_ok=True)
    if a.stim_from and not (out / "stim_rates.npz").exists():
        import shutil
        for f in ("stim_rates.npz", "input_check.json"):
            shutil.copyfile(Path(a.stim_from) / f, out / f)
    t0 = time.time()
    print(f"free {free_gb():.1f} GB", flush=True)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb():.1f} GB available (< {MIN_FREE_GB}); stop")
    cx, groups_in, eyes = build()
    stim_path = out / "stim_rates.npz"
    if stim_path.exists():
        rates = dict(np.load(stim_path))
    else:
        rates = render(eyes, print)
        np.savez_compressed(stim_path, **rates)
        chk = input_check(rates, eyes.tcode, list(eyes.fw_types))
        same = None
        if F3B1_STIM.exists():
            old = dict(np.load(F3B1_STIM))
            same = bool(set(old) == set(rates) and all(np.array_equal(old[k], rates[k]) for k in rates))
        chk.update(transduction=str(TRANSDUCTION.relative_to(ROOT)), interval_ms=C.INTERVAL_MS, subframes=1,
                   n_grey=C.N_GREY, n_stim=C.N_STIM, stimuli={k: v[0] for k, v in C.STIMULI.items()},
                   equals_f3b1_render=same)
        (out / "input_check.json").write_text(json.dumps(chk, indent=1))
    chk = json.loads((out / "input_check.json").read_text())
    print("input check", "passed" if chk["passed"] else "FAILED",
          {k: round(v["A_in_T4aT5a"], 4) for k, v in chk["per_stimulus"].items()},
          "equals F3b-1 render:", chk["equals_f3b1_render"], flush=True)
    if not chk["passed"]:
        raise SystemExit("input check failed: stop (prereg/11 section 2)")
    del eyes

    from engine.lif_brian2 import Brian2LIFEngine
    n = cx.neurons
    dn_idx = np.flatnonzero((n["super_class"] == "descending").to_numpy())
    dn_tab = dict(index=dn_idx, cell_type=n["cell_type"].astype(str).to_numpy()[dn_idx],
                  side=n["side"].astype(str).to_numpy()[dn_idx])
    from mv2_common import model_variant
    mv = model_variant(a.arm, cx)
    eng = Brian2LIFEngine(cx, input_groups=groups_in, seed=0, model_variant=mv)
    if mv is not None:
        print(f"model arm {a.arm}: {mv.record()}", flush=True)
    print(f"engine {eng.n_neurons:,} neurons, {len(eng.syn):,} synapses; free {free_gb():.1f} GB "
          f"({time.time() - t0:.0f} s)", flush=True)
    nI = C.N_GREY + C.N_STIM
    lo, hi = C.WINDOW
    for name in C.STIMULI:
        L, R = rates[f"{name}__L"].astype(float), rates[f"{name}__R"].astype(float)
        for s in seeds:
            path = out / "brain" / f"{name}_s{s}.npz"
            if path.exists():
                continue
            t1 = time.time()
            eng.reset(s)
            cnt = np.zeros((nI, len(dn_idx)), np.int16)
            net = np.zeros(nI, np.int64)
            st = {"prev": eng.spike_counts()}

            def on_boundary(j, e):
                if j > 0:
                    cur = e.spike_counts()
                    d = cur - st["prev"]
                    cnt[j - 1] = d[dn_idx]
                    net[j - 1] = d.sum()
                    st["prev"] = cur
                if j < nI:
                    e.set_inputs({"sugar": 0.0, "vbnd_L": L[j], "vbnd_R": R[j]})

            eng.run_coupled(nI, C.INTERVAL_MS, on_boundary)
            np.savez_compressed(path, stimulus=name, t0_condition=C.STIMULI[name][0], seed=s, dn_counts=cnt,
                                net_counts=net, input_mean_L=L.mean(1), input_mean_R=R.mean(1), interval_ms=C.INTERVAL_MS,
                                weights_sha256=eng.state_dict()["weights_sha256"], model_sha256=eng.model_sha256,
                                model_arm=a.arm,
                                **{f"dn_{k}": v for k, v in dn_tab.items()})
            print(f"{name} s{s}: DN spikes {int(cnt.sum())}, net {net[lo:hi].sum() / (eng.n_neurons * 1.0):.2f} Hz "
                  f"({time.time() - t1:.0f} s, free {free_gb():.1f} GB)", flush=True)
    (out / "brain" / "DONE").write_text(json.dumps(dict(wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb())) + "\n")
    print("done", round(time.time() - t0), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
