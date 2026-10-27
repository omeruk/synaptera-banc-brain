"""scripts/readme_figures.py: --help, a run into a temporary folder (Figures 0 and 1, each 2000 px wide, at most
1.5 MB), the refusal to overwrite without --overwrite (one existing output blocks both), and the refusal of an --out
inside the local preview folder. Every call runs in its own subprocess; only docs/results/f5_stageA_report.json is
read."""
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


def test_writes_both_figures_2000_px_and_refuses_to_overwrite(tmp_path):
    from PIL import Image
    out = tmp_path / "out"
    r = _run("--out", str(out), tmp=tmp_path)
    assert r.returncode == 0, r.stderr
    names = ["fig0_pipeline.png", "fig1_stageA_result.png"]
    assert sorted(p.name for p in out.iterdir()) == names
    for n in names:
        with Image.open(out / n) as im:
            assert im.size[0] == 2000
        assert (out / n).stat().st_size <= 1_500_000
    before = {n: ((out / n).read_bytes(), (out / n).stat().st_mtime_ns) for n in names}
    r = _run("--out", str(out), tmp=tmp_path)
    assert r.returncode != 0 and "--overwrite" in r.stderr
    assert {n: ((out / n).read_bytes(), (out / n).stat().st_mtime_ns) for n in names} == before


def test_one_existing_output_blocks_both(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "fig0_pipeline.png").write_bytes(b"placeholder")
    r = _run("--out", str(out), tmp=tmp_path)
    assert r.returncode != 0 and "nothing written" in r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["fig0_pipeline.png"]
    assert (out / "fig0_pipeline.png").read_bytes() == b"placeholder"


@pytest.mark.skipif(not PREVIEW.exists(), reason="the local preview folder does not exist on this machine")
def test_out_inside_the_preview_folder_is_refused(tmp_path):
    target = PREVIEW / "posthoc" / "readme_test_never_written"
    r = _run("--out", str(target), tmp=tmp_path)
    assert r.returncode != 0 and "preview folder" in r.stderr
    assert not target.exists()
