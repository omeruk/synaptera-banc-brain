"""Viewing-only preview of the F5 stage-A renders of training data seed 0 (phase 0; never a test seed). Nothing is
decided from these images; they are not committed.

    env -u PYTHONPATH python scripts/f5_stageA_preview.py --copy-to /mnt/c/Users/<user>/Desktop/synaptera_preview

Called by `scripts/f5_stageA_records.py run --preview ...` while the batch-0 renders exist:
- the raw eye images (luminance seen by each eye) at 4 time points of rot_left at 60 deg/s (computed from
  tasks/t0_stimuli, the input of the render);
- heatmaps of the T4/T5 boundary rates over time, left and right eye, for rot_left and rot_right at 60 deg/s and
  rot_left at 120 deg/s (from the renders in runs/f5/stageA/tmp).
Output: runs/f5/stageA/preview/*.png, copied to --copy-to.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / "runs" / "f5" / "stageA"
PREVIEW = OUT / "preview"
N_GREY, DT = 100, 0.005
SEED = 0                                       # training seed, phase 0


def eye_images():
    import mujoco

    from body.flygym_body import FlightBody
    from tasks import t0_stimuli as S
    b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs="tuck", enable_vision=True)
    mujoco.mj_forward(b.m, b.d)
    rays = S.camera_rays(b.m, b.d, b.root)
    ks = [50, 120, 200, 299]                    # interval index: grey lead-in, then 0.1, 0.5, 1.0 s into the drum
    fig, ax = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 7))
    for j, k in enumerate(ks):
        t = None if k < N_GREY else (k - N_GREY + 1) * DT
        lum = S.luminance(2, t, rays, tf_hz=60 / S.WAVELENGTH_DEG)
        for e, side in enumerate(("left", "right")):
            ax[e, j].imshow(lum[e], cmap="gray", vmin=0, vmax=1)
            ax[e, j].set_title(f"{side} eye, " + ("grey lead-in" if t is None else f"t = {t:.3f} s"), fontsize=9)
            ax[e, j].axis("off")
    fig.suptitle("rot_left, 60 deg/s, phase 0 (seed 0): raw eye images (luminance)")
    fig.tight_layout()
    fig.savefig(PREVIEW / "eye_images_rot_left_v060_seed0.png", dpi=110)
    plt.close(fig)


def heatmaps():
    m = np.load(OUT / "boundary_meta.npz")
    types = [str(t) for t in m["fw_types"]]
    t45 = [i for i, t in enumerate(types) if t.startswith(("T4", "T5"))]
    for key, title in (("rot_left_v060_p0", "rot_left 60 deg/s"), ("rot_right_v060_p0", "rot_right 60 deg/s"),
                       ("rot_left_v120_p0", "rot_left 120 deg/s")):
        z = np.load(OUT / "tmp" / f"{key}.npz")
        fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
        for e, (side, tc) in enumerate((("L", m["tcode_L"]), ("R", m["tcode_R"]))):
            sel = np.flatnonzero(np.isin(tc, t45))
            order = sel[np.argsort(tc[sel], kind="stable")]
            r = z[side][:, order].T
            im = ax[e].imshow(r, aspect="auto", cmap="magma", interpolation="nearest",
                              extent=[0, r.shape[1] * DT, len(order), 0], vmin=0, vmax=np.percentile(r, 99))
            ax[e].axvline(N_GREY * DT, color="c", lw=0.8)
            bounds = np.flatnonzero(np.diff(tc[order])) + 1
            for b in bounds:
                ax[e].axhline(b, color="w", lw=0.4)
            mids = np.r_[0, bounds] + np.diff(np.r_[0, bounds, len(order)]) / 2
            ax[e].set_yticks(mids, [types[k] for k in np.unique(tc[order])], fontsize=7)
            ax[e].set_xlabel("time (s); drum starts at the cyan line")
            ax[e].set_title(f"{'left' if side == 'L' else 'right'} eye: T4/T5 boundary rates ({len(order):,} neurons)")
            fig.colorbar(im, ax=ax[e], label="rate (Hz)")
        fig.suptitle(f"{title}, phase 0 (seed 0)")
        fig.tight_layout()
        fig.savefig(PREVIEW / f"t4t5_{key.replace('_p0', '')}_seed0.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to", default=None)
    a = ap.parse_args(argv)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    eye_images()
    heatmaps()
    if a.copy_to:
        dst = Path(a.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in PREVIEW.glob("*.png"):
            shutil.copyfile(p, dst / p.name)
    (PREVIEW / "DONE").write_text("\n".join(sorted(p.name for p in PREVIEW.glob("*.png"))) + "\n")
    print("preview:", sorted(p.name for p in PREVIEW.glob("*.png")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
