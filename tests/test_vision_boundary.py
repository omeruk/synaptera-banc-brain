"""Boundary layer: selection, columns, runtime layer (v1 tables reproduced), display writer; sensors; geometry."""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, require_flyvis
from connectome import geometry
from connectome.schema import Connectome, IdIndex
from sense import sensors
from sense import vision_boundary as VB
from sense import vision_flyvis as V
from body.arena import load_arena
from toy import edges_table, neurons_table

DATA = ROOT / "data"
NEEDS_V1 = pytest.mark.skipif(not all((DATA / n).exists() for n in (
    "vision_boundary_783.csv", "column_assignment.csv.gz", "neuron_arbor_centroids.npz", "neuron_neuropil.npz",
    "visual_transduction.json")) or not (ROOT / "brain_model" / "Connectivity_783.parquet").exists(),
    reason="needs the v1 data files and brain_model/")
CODEX_TYPES = os.path.expanduser("~/Downloads/consolidated_cell_types.csv.gz")


# ── pure functions ───────────────────────────────────────────────────────────
def test_hex_distances():
    assert VB.hex_dist_flywire(1, 1) == 1 and VB.hex_dist_flywire(1, -1) == 2 and VB.hex_dist_flywire(3, 0) == 3
    assert VB.hex_radius_flyvis(2, -1) == 2 and VB.hex_radius_flyvis(0, 0) == 0


def test_name_mapping():
    assert VB.flywire_names("R3") == ("R1-6",) and VB.flywire_names("Mi1") == ("Mi1",)
    m = VB.flywire_to_flyvis(["R1", "R2", "CT1(Lo1)", "CT1(M10)", "Mi1", "Mi3"], {"R1-6", "CT1", "Mi1"})
    assert m == {"R1-6": "R1", "CT1": "CT1(Lo1)", "Mi1": "Mi1"}
    assert VB.flywire_names("X", {"X": ("Y1", "Y2")}) == ("Y1", "Y2")


def test_boundary_fraction_rule_toy():
    """Type A sends 90 % to a non-FlyVis neuron (driven), type B 10 % (not driven)."""
    nr = [dict(id=i, cell_type=t, side="left") for i, t in enumerate(["A", "A", "B", "B", "OUT", "OUT"])]
    cx = Connectome("t", neurons_table(nr), edges_table([(0, 4, 9), (0, 2, 1), (1, 4, 9), (1, 3, 1),
                                                         (2, 4, 1), (2, 3, 9), (3, 4, 1), (3, 2, 9)]))
    ct = cx.neurons["cell_type"].to_numpy(object)
    res, missing = VB.boundary_fractions(ct, ["A", "B", "Z"], cx)
    assert res.loc["A", "driven"] and not res.loc["B", "driven"]
    assert res.loc["A", "frac_out_nonflyvis"] == pytest.approx(0.9)
    assert res.loc["B", "n"] == 2 and missing == ["Z"]


def test_column_surface_recovers_a_linear_map():
    """Reference neurons whose centroid is an affine function of (p, q): the cubic surface assigns each
    centroid to its own column."""
    ids, rows, xyz = [], [], []
    k = 0
    for hemi in ("left", "right"):
        for p in range(-3, 4):
            for q in range(-3, 4):
                for t in ("Mi1", "Tm1", "T4a"):
                    ids.append(k)
                    rows.append(dict(id=k, hemisphere=hemi, type=t, p=p, q=q))
                    xyz.append([1000.0 * p + 10 * q, 700.0 * q - 5 * p, 100.0 * (p > 0)])
                    k += 1
    col = pd.DataFrame(rows)
    surf = VB.ColumnSurfaces(col, np.array(xyz), IdIndex(np.array(ids)))
    p, q, d, npl = surf.assign("left", "ME", np.array([2000.0 + 10, 700.0 - 10, 100.0]))
    assert (p, q) == (2, 1) and d < 0.5 and npl == "ME"
    assert surf.fit[("right", "LO")]["n_ref"] == 49


def test_display_writer_streams_and_refuses_overwrite(tmp_path):
    h5py = pytest.importorskip("h5py")
    eyes = type("E", (), dict(display_nodes=np.arange(3), display_type=np.array(["Mi1"] * 3),
                              display_u=np.zeros(3), display_v=np.zeros(3),
                              display_fw=dict(idx=np.arange(2, dtype=np.int32))))()
    p = tmp_path / "d.h5"
    w = VB.DisplayWriter(p, eyes)
    w.add(-1, np.ones((2, 3), np.float16))
    w.add(0, np.zeros((2, 3), np.float16))
    w.close()
    with h5py.File(p) as f:
        assert f["flyvis/activity"].shape == (2, 2, 3) and f["flyvis/step_idx"][:].tolist() == [-1, 0]
    with pytest.raises(FileExistsError):
        VB.DisplayWriter(p, eyes)


def test_compare_transduction_numeric_tolerance():
    a = dict(a0={"T4a": [0.1, 0.2]}, a_ref={"T4a": 1.0}, model="m", r_max=50.0)
    b = json.loads(json.dumps(a))
    b["a0"]["T4a"][0] += 1e-9
    assert V.compare_transduction(a, b) == []
    b["a_ref"]["T4a"] = 1.1
    assert V.compare_transduction(a, b) == ["/a_ref/T4a"]


# ── sensors (geometry only) ──────────────────────────────────────────────────
def test_platform_azimuth():
    a = load_arena()
    cx, cy = a.food_platform[:2]
    assert sensors.platform_azimuth(a, (cx, cy - 100, 0), np.pi / 2) == pytest.approx(0.0)
    assert sensors.platform_azimuth(a, (cx, cy - 100, 0), 0.0) == pytest.approx(90.0)
    assert sensors.platform_azimuth(a, (cx - 100, cy, 0), np.pi / 2) == pytest.approx(-90.0)


def test_expansion_rate_and_gaze_frame():
    a = load_arena()
    ex = sensors.ExpansionRate(a, 0.025)
    cx, cy, _, zt = a.food_platform
    assert ex((cx - 100, cy, zt)) == 0.0                       # first call: no previous value
    e = ex((cx - 90, cy, zt))
    assert e > 0                                               # approaching: angular radius grows
    th0, th1 = np.arctan2(10, 100), np.arctan2(10, 90)
    assert e == pytest.approx((th1 - th0) / (0.025 * th1))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])         # yaw +90 deg
    np.testing.assert_allclose(sensors.gaze_frame(R), R, atol=1e-12)
    tilt = R @ np.array([[np.cos(.3), 0, np.sin(.3)], [0, 1, 0], [-np.sin(.3), 0, np.cos(.3)]])
    np.testing.assert_allclose(sensors.gaze_frame(tilt), R, atol=1e-9)    # pitch removed, heading kept


# ── geometry reductions ──────────────────────────────────────────────────────
def test_arbor_centroids_and_dominant_neuropil():
    ix = IdIndex(np.array([10, 20, 30]))
    ids = np.array([10, 10, 20, 99])
    xyz = np.array([[0, 0, 0], [2, 4, 6], [5, 5, 5], [9, 9, 9]], float)
    g = geometry.arbor_centroids(ix, ids, xyz, fallback_xyz=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3.0]]),
                                 fallback_ok=np.array([True, True, True]))
    np.testing.assert_allclose(g["xyz"], [[1, 2, 3], [5, 5, 5], [3, 3, 3]])
    assert g["source"].tolist() == [1, 1, 2] and g["n_syn"].tolist() == [2, 1, 0]
    d = geometry.dominant_neuropil(ix, np.array([10, 20, 30]), ["ME_L", "ME_R", "AL_L", "FB"],
                                   np.array([[5, 1, 0, 0], [0, 0, 0, 9], [0, 0, 0, 0]]))
    assert list(d["neuropils"]) == ["AL", "FB", "ME"]
    assert d["dominant"].tolist() == [2, 1, -1] and d["side"].tolist() == ["L", "C", ""]
    assert d["frac"][0] == pytest.approx(5 / 6)


# ── v1 data reproduced ───────────────────────────────────────────────────────
@NEEDS_V1
@pytest.mark.skipif(not os.path.exists(CODEX_TYPES), reason="needs ~/Downloads/consolidated_cell_types.csv.gz")
def test_boundary_selection_and_table_reproduce_v1():
    require_flyvis()
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    ct = VB.load_codex_types(cx.index.ids, CODEX_TYPES)
    fv_types = list(dict.fromkeys(np.array(V.load_flyvis_net().connectome.nodes.type[:]).astype(str)))
    res, missing = VB.boundary_fractions(ct, fv_types, cx)
    driven = res[res["driven"]]
    assert len(driven) == 32 and int(driven["n"].sum()) == 34_121
    assert {str(m) for m in missing} == {"Mi3", "Mi11", "Mi12", "Tm5Y", "Tm28", "Tm30", "TmY13", "TmY18"}
    cen = np.load(DATA / "neuron_arbor_centroids.npz")
    npz = np.load(DATA / "neuron_neuropil.npz")
    tab, val = VB.build_table(cx, ct, dict(zip(driven.index, driven["flyvis_type"])),
                              V.load_columns(DATA / "column_assignment.csv.gz"), cen["xyz"],
                              npz["neuropils"][npz["dominant"]])
    ref = pd.read_csv(DATA / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    m = tab.merge(ref, on="id", suffixes=("", "_v1"), validate="one_to_one")
    assert len(m) == len(ref) == 34_121
    for c in ("fw_type", "flyvis_type", "hemisphere", "p", "q", "col_source", "u", "v"):
        assert (m[c] == m[c + "_v1"]).all(), c
    np.testing.assert_allclose(m["centroid_dist_um"], m["centroid_dist_um_v1"], atol=1e-9)
    assert tab["col_source"].value_counts().to_dict() == {"column_assignment": 25096, "partner_mean": 8974,
                                                          "centroid_ME": 51}


@NEEDS_V1
def test_boundary_eyes_runtime_on_v1_table():
    require_flyvis()
    from connectome.adapters import flywire783 as fw
    cx = fw.load_connectome(hash_files=False)
    tab = pd.read_csv(DATA / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    groups = VB.boundary_groups(cx, tab)
    assert (len(groups["vbnd_L"]), len(groups["vbnd_R"])) == (17185, 16936)
    ids = {"L": cx.index.ids_of(groups["vbnd_L"]), "R": cx.index.ids_of(groups["vbnd_R"])}
    tr = VB.load_transduction(DATA / "visual_transduction.json")
    eyes = VB.BoundaryEyes(tab, ids, tr, V.load_columns(DATA / "column_assignment.csv.gz"), cx.index,
                           interval_s=0.025, subframes=2, verbose=False)
    assert len(eyes.fw_types) == 32 and set(eyes.unresponsive) == {"Tm4"}
    grey = np.full((2, 721, 2), 0.5, np.float32)
    a, loom = eyes.step([grey, grey])
    r = eyes.rates(a)
    assert max(float(r["L"].mean()), float(r["R"].mean())) < 1.0
    assert eyes.display.shape == (2, len(eyes.display_nodes)) and np.isfinite(loom).all()
    assert eyes.type_means(r).shape == (2, 32)
    with pytest.raises(ValueError, match="no vision frames"):
        eyes.step([])
