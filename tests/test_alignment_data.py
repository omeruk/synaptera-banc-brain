"""docs/figures/landmarks.csv and warp_check.csv (FlyWire -> BANC alignment data of the figures): both parse with the
expected columns, and check_alignment of scripts/verify_report.py passes on the committed files and fails on a
perturbed landmark coordinate, a perturbed warp error and a changed set label (copies in a temporary folder)."""
import csv
import shutil
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as V  # noqa: E402

FIG = ROOT / "docs" / "figures"
FILES = ("landmarks.csv", "warp_check.csv", "transforms.json", "warp_check.json")
COLUMNS = {
    "landmarks.csv": ["cell_type", "side", "n_fw", "n_banc", "fw_x", "fw_y", "fw_z", "banc_x", "banc_y", "banc_z",
                      "set"],
    "warp_check.csv": ["id", "cell_type", "side", "soma", "landmark", "n_roots_native", "nodes_native", "nodes_warped",
                       "same_tree", "nearest_warped_node_um", "root_soma_um_nm", "root_soma_um_voxel", "root_soma_um",
                       "root_ok", "used", "root_vs_root_um", "affine_vs_warp_um", "dist_from_landmark_centroid_um",
                       "diff_x_um", "diff_y_um", "diff_z_um"],
}


def _read(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("name", list(COLUMNS))
def test_parses_with_the_expected_columns(name):
    fields, rows = _read(FIG / name)
    assert fields == COLUMNS[name] and rows


def test_check_alignment_passes_on_the_committed_files():
    n, bad = V.check_alignment()
    assert n > 0 and not bad, bad[:5]


def _copy(tmp_path):
    for f in FILES:
        shutil.copy(FIG / f, tmp_path / f)
    return tmp_path


def test_fails_on_a_perturbed_landmark_coordinate(tmp_path):
    d = _copy(tmp_path)
    fields, rows = _read(d / "landmarks.csv")
    row = next(r for r in rows if r["set"] == "control")
    row["fw_x"] = repr(float(row["fw_x"]) + 1000.0)                  # 1 um
    _write(d / "landmarks.csv", fields, rows)
    assert V.check_alignment(d)[1]


def test_fails_on_a_perturbed_warp_error(tmp_path):
    d = _copy(tmp_path)
    fields, rows = _read(d / "warp_check.csv")
    row = next(r for r in rows if r["used"] == "True")
    row["affine_vs_warp_um"] = repr(float(row["affine_vs_warp_um"]) * 1.1)
    _write(d / "warp_check.csv", fields, rows)
    assert V.check_alignment(d)[1]


def test_fails_on_a_changed_set_label(tmp_path):
    d = _copy(tmp_path)
    fields, rows = _read(d / "landmarks.csv")
    row = next(r for r in rows if r["set"] == "control")
    row["set"] = "check"
    _write(d / "landmarks.csv", fields, rows)
    bad = V.check_alignment(d)[1]
    assert any(k.startswith("align:c:") for k, _ in bad)
