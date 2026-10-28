"""Figure 7 (`figure` of scripts/readme_visual_input.py): the direction colours copied from scripts/viz_activity_figure.py,
the SHA-256 refusal on a changed render, rule 13 and the preview guard (both before any input is read), the A_in axis
labels (mathtext subscript), the space-time check (a row that differs from the hashed frame stops), the refactored
eye_images() giving the committed `eye_images` entry and space-time rows equal to row H // 2 of the hashed frames
(needs MuJoCo), and a run into a temporary folder (2000 px wide, at most 1.5 MB; needs the renders in
runs/f5/stageA/readme_renders/, which a fresh clone does not have)."""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import readme_visual_input as V  # noqa: E402

SCRIPT = ROOT / "scripts" / "readme_visual_input.py"
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")
VI = json.loads((ROOT / "docs" / "figures" / "visual_input.json").read_text())
HAVE_MUJOCO = all(importlib.util.find_spec(m) is not None for m in ("mujoco", "flygym"))
HAVE_RENDERS = all((V.RENDERS / f"{k}.npz").exists() for k in VI["provenance"]["renders"]) and V.META.exists()


def _run(*args, tmp):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(tmp / "mplconfig"), MUJOCO_GL="egl")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp, env=env, capture_output=True, text=True,
                          timeout=600)


def test_dir_col_equals_viz_activity_figure():
    tree = ast.parse((ROOT / "scripts" / "viz_activity_figure.py").read_text())
    names, dir_col = {}, None
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        t = n.targets[0]
        if isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple):
            for a, b in zip(t.elts, n.value.elts):
                if isinstance(b, ast.Constant):
                    names[a.id] = b.value
        elif isinstance(t, ast.Name) and isinstance(n.value, ast.Constant):
            names[t.id] = n.value.value
        elif isinstance(t, ast.Name) and t.id == "DIR_COL":
            dir_col = n.value
    assert dir_col is not None and {"NEG", "POS"} <= set(names)
    resolved = {k.value: names[v.id] for k, v in zip(dir_col.keys, dir_col.values)}
    assert resolved == V.DIR_COL
    assert (V.NEG, V.POS) == (names["NEG"], names["POS"])


def _fake_inputs(tmp_path):
    renders = tmp_path / "renders"
    renders.mkdir()
    prov = {}
    for k in V.KEYS:
        (renders / f"{k}.npz").write_bytes(k.encode())
        prov[k] = dict(sha256=hashlib.sha256(k.encode()).hexdigest())
    meta = tmp_path / "boundary_meta.npz"
    meta.write_bytes(b"meta")
    d = dict(provenance=dict(renders=prov, boundary_meta_sha256=hashlib.sha256(b"meta").hexdigest()))
    return d, renders, meta


def test_check_inputs_passes_on_matching_files(tmp_path):
    d, renders, meta = _fake_inputs(tmp_path)
    V.check_inputs(d, renders, meta)


@pytest.mark.parametrize("change", ["render", "missing", "meta"])
def test_check_inputs_refuses_a_changed_render(tmp_path, change):
    d, renders, meta = _fake_inputs(tmp_path)
    if change == "render":
        (renders / f"{V.KEYS[0]}.npz").write_bytes(b"changed")
    elif change == "missing":
        (renders / f"{V.KEYS[-1]}.npz").unlink()
    else:
        meta.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differ from visual_input.json"):
        V.check_inputs(d, renders, meta)


@pytest.mark.skipif(not HAVE_RENDERS, reason="the renders in runs/f5/stageA/readme_renders/ are not on this machine")
def test_check_inputs_passes_on_the_renders():
    V.check_inputs(VI)


def test_existing_output_is_not_overwritten(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / V.FIG7).write_bytes(b"placeholder")
    r = _run("figure", "--out", str(out), tmp=tmp_path)
    assert r.returncode != 0 and "--overwrite" in r.stderr and "nothing written" in r.stderr
    assert sorted(p.name for p in out.iterdir()) == [V.FIG7]
    assert (out / V.FIG7).read_bytes() == b"placeholder"


@pytest.mark.skipif(not PREVIEW.exists(), reason="the local preview folder does not exist on this machine")
def test_out_inside_the_preview_folder_is_refused(tmp_path):
    target = PREVIEW / "readme_test_never_written"
    r = _run("figure", "--out", str(target), tmp=tmp_path)
    assert r.returncode != 0 and "preview folder" in r.stderr
    assert not target.exists()


def test_a_in_labels_use_a_mathtext_subscript():
    from matplotlib.mathtext import MathTextParser
    for lab in V.A_IN_LABEL.values():
        assert r"$A_\mathrm{in}$" in lab and "A_in" not in lab.replace(r"$A_\mathrm{in}$", "")
        MathTextParser("agg").parse(lab)


def _fake_space_time():
    rng = np.random.default_rng(0)
    frames = [rng.random((2, 6, 5)) for _ in range(3)]
    entry = dict(images=[dict(interval=k) for k in (1, 4, 7)])
    rows = rng.random((2, 9, 5))
    for im, f in zip(entry["images"], frames):
        rows[:, im["interval"]] = f[:, 3]
    return entry, frames, dict(rows=rows, row=3)


def test_check_space_time_passes_on_equal_rows():
    V.check_space_time(*_fake_space_time())


def test_check_space_time_refuses_a_differing_row():
    entry, frames, st = _fake_space_time()
    st["rows"][1, entry["images"][-1]["interval"], 0] += 1e-15
    with pytest.raises(SystemExit, match="differs from the hashed frame"):
        V.check_space_time(entry, frames, st)


@pytest.mark.skipif(not HAVE_MUJOCO, reason="MuJoCo / FlyGym are not installed")
def test_eye_images_give_the_committed_entry():
    entry, lums, st = V.eye_images()
    assert entry == VI["eye_images"] and st is None     # what `export` writes, unchanged by the refactor
    assert len(lums) == len(entry["images"])
    for im, lum in zip(entry["images"], lums):
        assert list(lum.shape) == im["shape"] and str(lum.dtype) == im["dtype"]
    n = VI["window"]["window"][1]                       # the trial length of the input-check window
    entry2, lums2, st = V.eye_images(n)
    assert entry2 == entry
    assert st["rows"].shape == (2, n, im["shape"][2]) and st["row"] == im["shape"][1] // 2
    V.check_space_time(entry2, lums2, st)
    for im, lum in zip(entry["images"], lums):
        assert np.array_equal(st["rows"][:, im["interval"]], lum[:, st["row"]])
    assert all(np.all(np.diff(a) < 0) or np.all(np.diff(a) > 0) for a in st["azimuth_deg"])


@pytest.mark.skipif(not (HAVE_MUJOCO and HAVE_RENDERS), reason="needs MuJoCo and the renders in runs/")
def test_figure_is_2000_px_and_refuses_to_overwrite(tmp_path):
    from PIL import Image
    out = tmp_path / "out"
    r = _run("figure", "--out", str(out), tmp=tmp_path)
    assert r.returncode == 0, r.stderr
    path = out / V.FIG7
    with Image.open(path) as im:
        assert im.size[0] == 2000
    assert path.stat().st_size <= 1_500_000
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    r = _run("figure", "--out", str(out), tmp=tmp_path)
    assert r.returncode != 0 and "--overwrite" in r.stderr
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
