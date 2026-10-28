"""Figure 7 (`figure` of scripts/readme_visual_input.py): the direction colours copied from scripts/viz_activity_figure.py,
the SHA-256 refusal on a changed render, rule 13 and the preview guard (both before any input is read), the refactored
eye_images() giving the committed `eye_images` entry (needs MuJoCo), and a run into a temporary folder (2000 px wide,
at most 1.5 MB; needs the renders in runs/f5/stageA/readme_renders/, which a fresh clone does not have)."""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.skipif(not HAVE_MUJOCO, reason="MuJoCo / FlyGym are not installed")
def test_eye_images_give_the_committed_entry():
    entry, lums = V.eye_images()
    assert entry == VI["eye_images"]                    # what `export` writes, unchanged by the refactor
    assert len(lums) == len(entry["images"])
    for im, lum in zip(entry["images"], lums):
        assert list(lum.shape) == im["shape"] and str(lum.dtype) == im["dtype"]


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
