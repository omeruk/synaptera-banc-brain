"""README.md, docs/FIGURES.md, CITATION.cff and docs/readme_numbers.json: the committed files equal the generator output
(scripts/readme_text.py --check), every relative link and image resolves, every contents link matches a heading anchor,
every figure in docs/figures/ is shown, no vendor/ link and no e-mail address, CITATION.cff and the number list parse;
and the number scan of scripts/verify_report.py fails closed."""
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as V  # noqa: E402

PAGES = ("README.md", "docs/FIGURES.md", "docs/figures/README.md", "docs/provenance/README.md")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "readme_text.py"), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def _targets(text):
    """Relative link and image targets (no URL, no in-page anchor)."""
    found = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    return [t.split("#")[0] for t in found if not re.match(r"[a-z]+:", t) and not t.startswith("#")]


def _anchor(heading):
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_committed_files_equal_the_generator_output():
    r = _run("--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0 and "usage:" in r.stdout


@pytest.mark.parametrize("page", PAGES)
def test_every_relative_link_and_image_resolves(page):
    path = ROOT / page
    missing = [t for t in _targets(path.read_text()) if t and not (path.parent / t).exists()]
    assert not missing, missing


def test_contents_links_match_heading_anchors():
    text = (ROOT / "README.md").read_text()
    anchors = {_anchor(h) for h in re.findall(r"(?m)^#+ (.+)$", text)}
    links = re.findall(r"\]\(#([^)]+)\)", text)
    assert links and all(a in anchors for a in links), [a for a in links if a not in anchors]


def test_every_figure_is_shown():
    shown = set()
    for page in ("README.md", "docs/FIGURES.md"):
        path = ROOT / page
        shown |= {(path.parent / t).resolve() for t in _targets(path.read_text())}
    figs = [p for p in (ROOT / "docs" / "figures").rglob("*") if p.suffix in (".png", ".gif", ".jpg", ".svg")]
    assert figs and not [p.relative_to(ROOT) for p in figs if p.resolve() not in shown]


def test_no_vendor_link_and_no_email():
    for page in PAGES:
        assert not [t for t in _targets((ROOT / page).read_text()) if "vendor/" in t], page
    for f in ("README.md", "docs/FIGURES.md", "CITATION.cff"):
        assert not EMAIL.search((ROOT / f).read_text(encoding="utf-8")), f


def test_citation_cff_parses():
    yaml = pytest.importorskip("yaml")
    d = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    for k in ("cff-version", "message", "title", "authors", "repository-code", "license"):
        assert d.get(k), k
    assert "version" not in d and "date-released" not in d


def test_number_list_parses():
    entries = json.loads((ROOT / "docs" / "readme_numbers.json").read_text())["entries"]
    assert entries
    for e in entries:
        assert {"string", "files", "kind", "source"} <= set(e) and e["files"], e


# ── the verify_report number check fails closed ──────────────────────────────────────────────────────────────────
def test_scan_flags_a_number_that_is_not_recorded():
    text = "Mean d was 0.0268 and R2 rose in Figure S3; a typed 0.1234 stays.\n"
    assert V._readme_scan(text, ["0.0268"]) == ["0.1234"]


def test_scan_ignores_code_urls_and_categories():
    text = ("1. Step `--seed 3` at https://example.org/x9 (doi:10.1/2), prereg/14 section 8.1, Figures 1-4, "
            "CC BY 4.0, T4/T5, float32.\n")
    assert V._readme_scan(text, []) == []


def test_unknown_formula_is_a_mismatch(tmp_path, monkeypatch):
    entry = dict(string="160", files=[], kind="data",
                 source=dict(file="configs/f5_splits.json", keys=["val"], formula="not a registered formula",
                             format="{}"))
    nums = tmp_path / "readme_numbers.json"
    nums.write_text(json.dumps(dict(entries=[entry])))
    monkeypatch.setattr(V, "README_NUMBERS", nums)
    n, bad = V.check_readme()
    assert any(b[0].startswith("readme:data:160") for b in bad)
