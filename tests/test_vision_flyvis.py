"""FlyVis -> connectome T4/T5 visual input: lattice, alignment, eye convention, transduction, L1 / H4 fixes.
~1-2 minutes (FlyVis, one FlyGym body, parquet read). Skipped without the FlyWire files or FlyVis."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from conftest import ROOT, require_flyvis
from sense import vision_flyvis as V

TYPES = list(V.T45_TYPES)
COLUMNS = ROOT / "data" / "column_assignment.csv.gz"
TRANSDUCTION = ROOT / "data" / "t45_transduction.json"


def test_lattice_op_maps_neighbours_to_neighbours():
    p = np.array([1, -1, 0, 0, 1, -1])
    q = np.array([0, 0, 1, -1, 1, -1])
    u, v, d = V.map_columns(p, q)
    assert np.allclose(d, 0)
    assert set(zip(u, v)) == {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}
    u, v, d = V.map_columns(np.array([0]), np.array([0]))
    assert (u[0], v[0], d[0]) == (0, 0, 0)


def test_interval_and_subframes_come_from_configuration():
    assert V.flyvis_dt(0.025, 2) == pytest.approx(0.0125)       # v1
    assert V.flyvis_dt(0.005, 1) == pytest.approx(0.005)        # 5 ms coupling, one render per interval
    assert V.DEFAULT_SUBFRAMES == 2


def test_loom_is_the_subframe_mean_and_empty_frames_are_an_error():
    """Review L1: v1 took the last sub-frame and divided by zero on an empty list."""
    eyes = V.FlyVisEyes.__new__(V.FlyVisEyes)
    eyes.t5a, eyes.t5b = np.array([0, 1]), np.array([2])
    a1 = np.zeros((2, 4))
    a2 = np.zeros((2, 4))
    a1[0, :3] = [1.0, 1.0, 1.0]            # left eye loom of the first sub-frame: 1 + 1
    a2[0, :3] = [3.0, 3.0, 3.0]            # second sub-frame: 3 + 3
    a1[1, :3] = 2.0
    a2[1, :3] = 2.0
    assert eyes.loom([a1, a2])[0] == pytest.approx(4.0)          # mean(2, 6); the last sub-frame alone gives 6
    assert eyes.loom([a1, a2])[1] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="no vision frames"):
        eyes._run_frames([])


pytestmark_data = pytest.mark.skipif(not (COLUMNS.exists() and TRANSDUCTION.exists()),
                                     reason="needs data/column_assignment.csv.gz and t45_transduction.json")


@pytest.fixture(scope="module")
def cx():
    from connectome.adapters import flywire783 as fw
    return fw.load_connectome(hash_files=False)


@pytest.fixture(scope="module")
def columns():
    return V.load_columns(COLUMNS)


@pytest.fixture(scope="module")
def group_ids(cx, columns):
    c = V.t45_columns(columns)
    c = c[c["id"].isin(cx.index.ids)]
    idx = cx.index.index_of(c["id"].to_numpy())
    out = {}
    for s, sd in (("L", "left"), ("R", "right")):
        ii = np.sort(idx[(c["hemisphere"] == sd).to_numpy()])
        out[s] = cx.index.ids_of(ii)
    return out


@pytest.fixture(scope="module")
def eyes(columns, group_ids):
    require_flyvis()
    return V.FlyVisEyes(columns, group_ids, V.load_transduction(TRANSDUCTION), interval_s=0.025, subframes=2,
                        verbose=False)


@pytestmark_data
def test_group_sizes_match_v1(group_ids):
    assert (len(group_ids["L"]), len(group_ids["R"])) == (5901, 5921)


@pytestmark_data
def test_alignment_from_connectomes(eyes, cx, columns):
    res, vecs = V.fit_alignment(cx, columns, flyvis_edges=eyes.net.connectome.edges)
    for h in ("left", "right"):
        (s0, rot, refl), (s1, _, _) = res[h][0], res[h][1]
        assert (rot, refl) == (V.ALIGN_ROT, V.ALIGN_REFLECT)
        assert s0 > 1.5 * s1
        for t in ("T4a", "T4b", "T4c", "T4d"):
            for src in ("Mi4", "Mi9"):
                a, b = vecs[h][(t, src)]
                assert a @ b / np.linalg.norm(a) / np.linalg.norm(b) > 0.8, (h, t, src)
        for t in ("T5a", "T5b", "T5c", "T5d"):
            a, b = vecs[h][(t, "Tm9")]
            assert a @ b / np.linalg.norm(a) / np.linalg.norm(b) > 0.8, (h, t)


@pytestmark_data
def test_every_t45_neuron_has_a_flyvis_node(eyes, group_ids):
    for s in "LR":
        assert len(eyes.node[s]) == len(group_ids[s])
        assert eyes.n_outside[s] < 0.1 * len(eyes.node[s])


def _grating_rates(eyes, dir_L, dir_R, n_steps=40, skip=10):
    n = n_steps * eyes.subframes
    gL = list(V.grating_frames(dir_L, n, dt=eyes.dt))
    gR = list(V.grating_frames(dir_R, n, dt=eyes.dt))
    eyes.reset()
    out = []
    for k in range(n_steps):
        fr = [np.stack([gL[i][0], gR[i][1]]) for i in range(k * eyes.subframes, (k + 1) * eyes.subframes)]
        a, _ = eyes.step(fr)
        if k >= skip:
            out.append(eyes.type_means(eyes.rates(a)))
    return np.mean(out, 0)


@pytest.fixture(scope="module")
def grating_rates(eyes):
    return dict(ftb=_grating_rates(eyes, (-1, 0), (1, 0)), btf=_grating_rates(eyes, (1, 0), (-1, 0)),
                up=_grating_rates(eyes, (0, -1), (0, -1)), down=_grating_rates(eyes, (0, 1), (0, 1)))


@pytestmark_data
def test_eye_convention_synthetic(grating_rates):
    g = grating_rates
    i = {t: TYPES.index(t) for t in TYPES}
    for e in (0, 1):
        assert g["ftb"][e, i["T4a"]] > 2 * g["btf"][e, i["T4a"]]
        assert g["btf"][e, i["T4b"]] > 2 * g["ftb"][e, i["T4b"]]
        assert g["up"][e, i["T4c"]] > 2 * g["down"][e, i["T4c"]]
    for k in g:
        np.testing.assert_allclose(g[k][0], g[k][1], rtol=0.1, atol=2.0)


@pytestmark_data
def test_rates_scale(eyes, grating_rates):
    eyes.reset()
    grey = np.full((2, 721, 2), 0.5, np.float32)
    a, _ = eyes.step([grey] * eyes.subframes)
    r = eyes.rates(a)
    assert max(r["L"].mean(), r["R"].mean()) < 1.0
    for t, k in (("T4a", "ftb"), ("T4b", "btf"), ("T4c", "up")):
        assert grating_rates[k][:, TYPES.index(t)] == pytest.approx(V.R_MAX, rel=0.2)


@pytestmark_data
def test_eye_convention_rendered_yaw(eyes):
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    from body.flygym_body import FlightBody
    b = FlightBody(spawn_pos=(60, -60, 60), legs="tuck", enable_vision=True)
    adr = b.m.jnt_qposadr[b.m.body_jntadr[b.root]]
    q0 = b.d.qpos[adr + 3:adr + 7].copy()
    w = np.radians(90)

    def run(sign):
        eyes.reset()
        acc = []
        for k in range(40):
            R = Rot.from_rotvec([0, 0, sign * w * eyes.dt * k]) * Rot.from_quat(q0[[1, 2, 3, 0]])
            b.d.qpos[adr + 3:adr + 7] = R.as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(b.m, b.d)
            a, _ = eyes.step([b.update_vision().copy()])
            if k >= 10:
                acc.append([[a[s][eyes.type_code[s] == j].mean() for j in range(8)] for s in "LR"])
        return np.mean(acc, 0)

    d = run(-1) - run(+1)
    a, bb = TYPES.index("T4a"), TYPES.index("T4b")
    assert d[0, a] > 0 and d[0, bb] < 0
    assert d[1, bb] > 0 and d[1, bb] > 2 * d[1, a]


@pytestmark_data
def test_transduction_constants_reproducible(eyes):
    """Numeric tolerance (review M7: bit-level hashes differ across platforms)."""
    d = V.calibrate_transduction(eyes.net, interval_s=0.025, subframes=2)
    ref = V.load_transduction(TRANSDUCTION)
    for t in TYPES:
        np.testing.assert_allclose(d["a0"][t], ref["a0"][t], rtol=1e-4, atol=1e-6)
        assert d["a_ref"][t] == pytest.approx(ref["a_ref"][t], rel=1e-4, abs=1e-6), t
    assert ref["r_max"] == V.R_MAX and ref["dt"] == pytest.approx(eyes.dt)
