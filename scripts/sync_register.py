"""Mirror every constant of configs/*.yaml into configs/assumptions.json as ``<file>.<name>`` (rule 2).

    env -u PYTHONPATH python scripts/sync_register.py [--check]

--check exits non-zero when the register is out of date (used by tests/test_assumptions.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import configs  # noqa: E402

REGISTER = ROOT / "configs" / "assumptions.json"


def synced() -> dict:
    d = json.loads(REGISTER.read_text())
    consts = {k: v for k, v in d["constants"].items() if v.get("config_file") is None}
    for yml in sorted(configs.CONFIG_DIR.glob("*.yaml")):
        if "groups" not in __import__("yaml").safe_load(yml.read_text()):
            continue                       # not a constants file (e.g. data_sources.yaml)
        for k, rec in configs.entries(yml.stem).items():
            consts[f"{yml.stem}.{k}"] = dict(rec, config_file=f"configs/{yml.name}", used_in=[])
    d["constants"] = consts
    return d


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = json.dumps(synced(), indent=2) + "\n"
    if "--check" in argv:
        return 0 if REGISTER.read_text() == text else 1
    REGISTER.write_text(text)
    print(f"{len(json.loads(text)['constants'])} constants in {REGISTER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
