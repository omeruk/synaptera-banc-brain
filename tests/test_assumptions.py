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
        assert value == c["value"] or float(value) == pytest.approx(c["value"]), ref
