"""F1 scripts on toy data: sugar-GRN / MN9 group selection, hop distances."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from connectome.schema import Connectome
from conftest import ROOT
from toy import edges_table, neurons_table


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke_groups_select_sugar_grns_by_function_and_body_part():
    smoke = _load("f1_banc_smoke")
    n = neurons_table([dict(id=i, cell_type="x", side="left") for i in range(6)])
    n["function"] = ["sugar,Gr5a", "sugar", "bitter", "sugar", None, "sugar"]
    n["body_part"] = ["labellum", "front_leg", "labellum", "hind_leg", None, "wing_margin"]
    n["super_class"] = ["sensory", "sensory_ascending", "sensory", "sensory", "motor", "sensory"]
    n.loc[4, "cell_type"] = "MN9"
    n.loc[4, "super_class"] = "motor"
    cx = Connectome("t", n, edges_table([(0, 1, 1)]))
    g = smoke.groups(cx)
    assert g["labellar"].tolist() == [0] and g["leg"].tolist() == [1, 3] and g["mn9"].tolist() == [4]


def test_hops_breadth_first():
    diag = _load("f1_banc_smoke_diag")
    n = neurons_table([dict(id=i, cell_type="x", side="left") for i in range(5)])
    cx = Connectome("t", n, edges_table([(0, 1, 1), (1, 2, 1), (2, 3, 1)]))
    d = diag.hops(cx, np.array([0]))
    assert d.tolist() == [0, 1, 2, 3, -1]
