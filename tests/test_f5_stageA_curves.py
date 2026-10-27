"""scripts/f5_stageA_curves.py and docs/results/f5_stageA_curves.json: --help, the committed JSON's keys, the refusal to
overwrite, and check_f5_stageA_curves of scripts/verify_report.py (passes on the committed JSON, fails on a perturbed
copy). Needs no run directory: the refusal comes before any log is read."""
import copy
import json
import os
import subprocess
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_report as V  # noqa: E402

SCRIPT = ROOT / "scripts" / "f5_stageA_curves.py"
CURVES = ROOT / "docs" / "results" / "f5_stageA_curves.json"


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=300)


def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0 and "usage:" in r.stdout


def test_committed_json_has_the_expected_keys():
    d = json.loads(CURVES.read_text())
    assert {"split", "batch", "steps", "eval_every", "set_accuracy", "lr_grid", "selected_lr", "n_runs", "runs"} <= set(d)
    assert d["split"] == "validation" and d["n_runs"] == len(d["runs"])
    for r in d["runs"]:
        assert {"arm", "training_seed", "lr", "replicate", "selected_rate", "final_step", "selected_step", "batch",
                "config_steps", "evaluations", "sha256"} <= set(r)
        e = r["evaluations"]
        assert e["step"][0] == 0 and len({len(v) for v in e.values()}) == 1
        assert set(e) == {"step", "trials", "loss", "r2", "sign_accuracy"}


def test_refuses_to_overwrite():
    before = (CURVES.read_bytes(), CURVES.stat().st_mtime_ns)
    r = _run()
    assert r.returncode != 0 and "--overwrite" in r.stderr
    assert (CURVES.read_bytes(), CURVES.stat().st_mtime_ns) == before


def test_check_passes_on_the_committed_json():
    n, bad = V.check_f5_stageA_curves()
    assert n > 0 and not bad, bad[:5]


def test_check_fails_on_a_perturbed_copy(tmp_path):
    d = json.loads(CURVES.read_text())
    for edit in ("final_loss", "selected_step", "accuracy_at_first_eval", "drop_run"):
        p = copy.deepcopy(d)
        r = next(x for x in p["runs"] if x["arm"] == "true" and x["training_seed"] == 3)
        e = r["evaluations"]
        if edit == "final_loss":
            e["loss"][-1] *= 1.001
        elif edit == "selected_step":
            r["selected_step"] = e["step"][-1] if r["selected_step"] != e["step"][-1] else e["step"][-2]
        elif edit == "accuracy_at_first_eval":
            e["sign_accuracy"][1] = 0.5
        else:
            p["runs"].remove(r)
        path = tmp_path / f"{edit}.json"
        path.write_text(json.dumps(p))
        n, bad = V.check_f5_stageA_curves(path)
        assert bad, edit
