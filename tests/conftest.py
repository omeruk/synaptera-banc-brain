"""pytest configuration.

Ported from synaptera@c94796d:tests/conftest.py. The stale ``collect_ignore`` list of walking scripts is not
carried over (L5). Tests that build a full connectome network are marked ``slow`` and run with ``--runslow``.
A test that fails only because an input data file is missing is reported as skipped with the file name.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # repo root: tests import connectome, engine, ... regardless of the invocation
DATA_DIRS = (ROOT / "brain_model", ROOT / "data", ROOT / "banc")
MODEL_PY = ROOT / "brain_model" / "model.py"   # Shiu et al. model, imported by engine/lif_brian2.py


def require_flyvis():
    """Skip without FlyVis, else import it pinned to the CPU. Not pytest.importorskip("flyvis"): a plain import picks
    CUDA when available and makes it torch's default device for the whole test process."""
    if importlib.util.find_spec("flyvis") is None:
        pytest.skip("needs flyvis")
    from sense import vision_flyvis
    return vision_flyvis.import_flyvis()


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests (full-brain build, long physics)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full-brain or long-running test (needs --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _missing_data_file(excinfo):
    """Path of a missing input file under brain_model/, data/ or banc/ that caused excinfo, else None."""
    err = excinfo.value
    while err is not None:
        if isinstance(err, ModuleNotFoundError) and err.name == "model" and not MODEL_PY.exists():
            return MODEL_PY
        if isinstance(err, FileNotFoundError) and err.filename:
            p = Path(str(err.filename)).resolve()
            if not p.exists() and any(d in p.parents for d in DATA_DIRS):
                return p
        err = err.__cause__ or err.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        p = _missing_data_file(call.excinfo)
        if p is not None:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), item.location[1] or 0,
                            f"Skipped: missing data file {p.relative_to(ROOT)} (run scripts/fetch_data.py)")
