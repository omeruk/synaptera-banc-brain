"""configs/assumptions.json: every constant has an origin class (CLAUDE.md rule 2) and equals the code value."""
import importlib
import json
from pathlib import Path

import pytest

REGISTER = json.loads((Path(__file__).resolve().parent.parent / "configs" / "assumptions.json").read_text())
CLASSES = set(REGISTER["origin_classes"])


@pytest.mark.parametrize("key", sorted(REGISTER["constants"]))
def test_constant_is_classified_and_matches_code(key):
    c = REGISTER["constants"][key]
    assert c["origin"] in CLASSES
    assert c.get("source"), "every constant names its source"
    if c["origin"] == "F":
        assert c.get("fitted_to"), "an F constant states what it was fitted to"
    for ref in c["used_in"]:
        module, attr = ref.split(":")
        value = getattr(importlib.import_module(module.removesuffix(".py").replace("/", ".")), attr)
        if isinstance(value, tuple):
            value = list(value)
        assert value == c["value"] or (not isinstance(value, list) and float(value) == pytest.approx(c["value"])), ref


@pytest.mark.parametrize("key", sorted(k for k, c in REGISTER["constants"].items() if "shiu_param" in c))
def test_shiu_parameters_equal_the_pinned_model(key):
    """The registered Shiu parameters are what the hashed brain_model/model.py defines."""
    from conftest import MODEL_PY
    if not MODEL_PY.exists():
        pytest.skip("brain_model/model.py missing")
    pytest.importorskip("brian2")
    from engine.lif_brian2 import load_shiu_model
    model, _ = load_shiu_model(MODEL_PY)
    c = REGISTER["constants"][key]
    assert float(model.default_params[c["shiu_param"]]) == pytest.approx(c["value"], rel=1e-9)
