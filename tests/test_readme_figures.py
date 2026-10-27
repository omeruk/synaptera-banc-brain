"""scripts/readme_figures.py: --help, a run into a temporary folder (2000 px wide, at most 1.5 MB), the refusal to
overwrite without --overwrite, and the refusal of an --out inside the local preview folder. Every call runs in its own
subprocess; only docs/results/f5_stageA_report.json is read."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

SCRIPT = ROOT / "scripts" / "readme_figures.py"
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")


def _run(*args, tmp):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(tmp / "mplconfig"))
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp, env=env, capture_output=True, text=True,
                          timeout=300)


def test_help_exits_zero(tmp_path):
    r = _run("--help", tmp=tmp_path)
    assert r.returncode == 0 and "usage:" in r.stdout


def test_writes_a_2000_px_png_and_refuses_to_overwrite(tmp_path):
    from PIL import Image
    out = tmp_path / "out"
    r = _run("--out", str(out), tmp=tmp_path)
    assert r.returncode == 0, r.stderr
    png = out / "fig1_stageA_result.png"
    with Image.open(png) as im:
        assert im.size[0] == 2000
    assert png.stat().st_size <= 1_500_000
    assert sorted(p.name for p in out.iterdir()) == ["fig1_stageA_result.png"]
    before = (png.read_bytes(), png.stat().st_mtime_ns)
    r = _run("--out", str(out), tmp=tmp_path)
    assert r.returncode != 0 and "--overwrite" in r.stderr
    assert (png.read_bytes(), png.stat().st_mtime_ns) == before


@pytest.mark.skipif(not PREVIEW.exists(), reason="the local preview folder does not exist on this machine")
def test_out_inside_the_preview_folder_is_refused(tmp_path):
    target = PREVIEW / "posthoc" / "readme_test_never_written"
    r = _run("--out", str(target), tmp=tmp_path)
    assert r.returncode != 0 and "preview folder" in r.stderr
    assert not target.exists()
