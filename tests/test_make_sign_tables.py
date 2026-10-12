"""scripts/make_sign_tables.py regenerates the v1 tables from the Codex predictions (numeric tolerance)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

CODEX = Path("~/Downloads/neurons.csv.gz").expanduser()
pytestmark = pytest.mark.skipif(not CODEX.exists() or not (ROOT / "data" / "nt_impute_783.csv").exists(),
                                reason="needs ~/Downloads/neurons.csv.gz and the v1 tables in data/")


def test_regenerated_tables_equal_v1(tmp_path):
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "make_sign_tables.py"), "--codex-neurons", str(CODEX),
           "--out-dir", str(tmp_path), "--literature-types", str(ROOT / "data" / "nt_literature_783.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    a = pd.read_csv(tmp_path / "nt_modulatory_silent_783.csv").set_index("root_id")
    b = pd.read_csv(ROOT / "data" / "nt_modulatory_silent_783.csv").set_index("root_id")
    assert set(a.index) == set(b.index)
    assert (a["nt_type"] == b["nt_type"].reindex(a.index)).all()
    np.testing.assert_allclose(a["nt_type_score"], b["nt_type_score"].reindex(a.index), atol=1e-9)
    a = pd.read_csv(tmp_path / "nt_impute_783.csv").set_index("root_id")
    b = pd.read_csv(ROOT / "data" / "nt_impute_783.csv").set_index("root_id").reindex(a.index)
    assert len(a) == 19_042
    for c in ("published_sign", "imputed_sign", "n_peers", "peer_syn_pos", "peer_syn_neg"):
        assert (a[c] == b[c]).all(), c
    a = pd.read_csv(tmp_path / "nt_literature_783.csv").set_index("cell_type")
    b = pd.read_csv(ROOT / "data" / "nt_literature_783.csv").set_index("cell_type").reindex(a.index)
    for c in ("n", "sign_lit", "n_change"):
        assert (a[c] == b[c]).all(), c
    # refuses to overwrite
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode != 0 and "exists" in (r2.stdout + r2.stderr)
