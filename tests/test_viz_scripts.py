"""scripts/viz_*.py (demonstration code, no claim): a smoke test that needs no data. Each script imports cleanly, its
--help exits with code 0, and neither writes anything: the repository tree (runs/ included) and the machine-2 preview
folder are compared entry by entry (size, mtime) before and after. Every call runs in its own subprocess with bytecode
writing off, so a script's module-level settings (e.g. CUDA_VISIBLE_DEVICES in viz_activity.py) never reach the test
process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

SCRIPTS = ("viz_activity", "viz_activity_figure", "viz_anatomy", "viz_positions", "viz_preview", "viz_render",
           "viz_video")
PREVIEW = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview")      # machine 2; skipped where it does not exist


def _snapshot(root: Path, skip=(".git",)) -> dict:
    """Every file and folder under root (links not followed): (is_dir, size, mtime_ns)."""
    out = {}
    for d, dirs, files in os.walk(root):
        if Path(d) == root:
            dirs[:] = [x for x in dirs if x not in skip]
        for name in dirs + files:
            p = Path(d) / name
            st = p.lstat()
            out[str(p.relative_to(root))] = (p.is_dir() and not p.is_symlink(), st.st_size, st.st_mtime_ns)
    return out


def _state() -> dict:
    s = {"repo": _snapshot(ROOT)}
    if PREVIEW.exists():
        s["preview"] = _snapshot(PREVIEW)
    return s


def _run(args, cwd, tmp):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(tmp / "mplconfig"))
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_the_seven_viz_scripts_are_the_ones_tested():
    assert sorted(p.stem for p in (ROOT / "scripts").glob("viz_*.py")) == sorted(SCRIPTS)


@pytest.mark.parametrize("name", SCRIPTS)
def test_viz_script_imports_and_help_write_nothing(name, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    before = _state()
    imp = _run(["-c", f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import {name}"], cwd, tmp_path)
    assert imp.returncode == 0, imp.stderr
    hlp = _run([str(ROOT / "scripts" / f"{name}.py"), "--help"], cwd, tmp_path)
    assert hlp.returncode == 0, hlp.stderr
    assert "usage:" in hlp.stdout
    after = _state()
    for where in before:
        changed = sorted(k for k in before[where].keys() | after[where].keys()
                         if before[where].get(k) != after[where].get(k))
        assert not changed, f"{name} wrote into {where}: {changed[:10]}"
    assert not any(cwd.iterdir()), f"{name} wrote into its working directory: {list(cwd.iterdir())}"
