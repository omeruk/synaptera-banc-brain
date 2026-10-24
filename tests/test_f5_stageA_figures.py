"""scripts/f5_stageA_figures.py on the committed stage-A report JSON, into a temporary directory."""
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import f5_stageA_figures as F  # noqa: E402


def test_writes_the_three_pngs(tmp_path):
    assert F.REPORT.exists()
    assert F.main(["--out", str(tmp_path)]) == 0
    for f in F.FILES:
        p = tmp_path / f
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(F.FILES)
