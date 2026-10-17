"""F5 step 2c (prereg/14 section 7, prereg/17): the stage-A record plan, the phase rule, and the speed / phase options
of tasks/t0_stimuli.luminance (defaults unchanged). No data files."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import f5_stageA_records as F  # noqa: E402
from tasks import t0_stimuli as S  # noqa: E402


def test_plan_counts_and_splits():
    P = F.plan()
    assert len(P) == 2400 and len({r["file"] for r in P}) == 2400
    assert sum(r["split"] == "train" for r in P) == 1600 and sum(r["split"] == "test" for r in P) == 400
    old = [r for r in P if r["seed"] < 40 and r["speed_deg_s"] in (60, None)]
    assert len(old) == 160                                          # prereg/17: the 160 regenerated records
    assert len({r["render"] for r in P}) == 10 + 200 * 9            # grey shared, seeds 0-39 share phase 0
    assert sum(len(b) for b in F.batches()) == 240


def test_phase_rule():
    assert all(F.phase_of(s) == 0.0 for s in range(40))
    ph = [F.phase_of(s) for s in range(40, 240)]
    assert all(0.0 <= p < 30.0 for p in ph) and len(set(ph)) == 200
    assert F.phase_of(45) == float(np.random.default_rng(45).uniform(0.0, 30.0))


def test_render_keys():
    assert F.render_key("grey", 0, 7) == F.render_key("grey", 0, 207) == "grey"
    assert F.render_key("rot_left", 90, 3) == "rot_left_v090_p0"
    assert F.key_params("rot_right_v030_s050") == (1, 1.0, F.phase_of(50))
    assert F.key_params("static_p0") == (3, 2.0, 0.0)


def _rays():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2, 8, 9, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_luminance_defaults_unchanged_and_options():
    rays = _rays()
    az = S.azimuth(rays) / S.WAVELENGTH_DEG
    for cond, sgn in ((1, 1.0), (2, -1.0), (3, 0.0)):
        t = 0.37
        old = S.GREY + 0.5 * S.CONTRAST * np.sin(2 * np.pi * (az + sgn * S.TF_HZ * (0.0 if cond == 3 else t)))
        assert np.array_equal(S.luminance(cond, t, rays), old)
        assert np.array_equal(S.luminance(cond, t, rays, tf_hz=60 / 30, phase_deg=0.0), old)
    # a drum at speed v moves the pattern by v * t degrees; a phase of one wavelength is the same pattern
    a = S.luminance(2, 0.25, rays, tf_hz=90 / 30)
    b = S.luminance(2, 0.0, rays, phase_deg=-90 * 0.25)
    assert np.abs(a - b).max() < 1e-9
    assert np.abs(S.luminance(1, 0.1, rays, phase_deg=30.0) - S.luminance(1, 0.1, rays)).max() < 1e-9
    assert np.array_equal(S.luminance(10, 0.1, rays, phase_deg=12.0), S.luminance(10, 0.1, rays))
