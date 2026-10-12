"""Configuration files with a parameter-origin class on every constant (CLAUDE.md rule 2).

A YAML file under configs/ holds ``groups``; each group has an ``origin`` class (M / P / F / E / I), a ``source``
text, ``fitted_to`` for class F, and ``values`` (name -> value). :func:`load` returns the flat name -> value
dict, :func:`entries` the full records. ``scripts/sync_register.py`` mirrors every record into
``configs/assumptions.json`` as ``<file>.<name>`` and a test checks that the mirror is current.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
ORIGINS = ("M", "P", "F", "E", "I")


def entries(name: str, path=None) -> dict:
    p = Path(path) if path else CONFIG_DIR / f"{name}.yaml"
    doc = yaml.safe_load(p.read_text())
    out = {}
    for g in doc["groups"]:
        origin = g["origin"]
        if origin not in ORIGINS:
            raise ValueError(f"{p.name}: origin {origin!r} not in {ORIGINS}")
        if not g.get("source"):
            raise ValueError(f"{p.name}: group without source ({list(g['values'])[:2]})")
        if origin == "F" and not g.get("fitted_to"):
            raise ValueError(f"{p.name}: F group without fitted_to ({list(g['values'])[:2]})")
        for k, v in g["values"].items():
            if k in out:
                raise ValueError(f"{p.name}: duplicate constant {k}")
            rec = dict(value=v, origin=origin, source=g["source"], unit=g.get("unit", ""))
            if origin == "F":
                rec["fitted_to"] = g["fitted_to"]
            out[k] = rec
    return out


def load(name: str, path=None) -> dict:
    return {k: r["value"] for k, r in entries(name, path).items()}
