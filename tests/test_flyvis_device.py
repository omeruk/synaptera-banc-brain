"""FlyVis is pinned to the CPU (sense.vision_flyvis.import_flyvis), also on a machine with CUDA. Each check runs in a
fresh interpreter, because the pin acts on the first `import flyvis` of a process. ~30 s. Skipped without FlyVis."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT

pytestmark = pytest.mark.skipif(importlib.util.find_spec("flyvis") is None, reason="needs flyvis")


def run_py(code):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", r.stderr


def test_flyvis_network_and_eye_are_on_the_cpu():
    code = """
import json, torch
cuda = torch.cuda.is_available()
from sense import vision_flyvis as V
flyvis = V.import_flyvis()
net = V.load_flyvis_net()
from flyvis.datasets.rendering import BoxEye
from flygym.examples.vision.vision_network import RetinaMapper
RetinaMapper()
print(json.dumps(dict(cuda_before=cuda, cuda_after=torch.cuda.is_available(), flyvis=str(flyvis.device),
                      default=str(torch.get_default_device()), new_tensor=str(torch.zeros(1).device),
                      params=sorted({str(p.device) for p in net.parameters()}),
                      eye=str(BoxEye(extent=V.FLYVIS_EXTENT).receptor_centers.device))))
"""
    rc, out, err = run_py(code)
    assert rc == 0, err[-3000:]
    d = json.loads(out)
    assert d["cuda_after"] == d["cuda_before"]          # CUDA is hidden from the flyvis import only
    assert d["flyvis"] == d["default"] == d["new_tensor"] == d["eye"] == "cpu"
    assert d["params"] == ["cpu"]


def test_flyvis_forward_runs_on_the_cpu():
    """The forward pass that failed on CUDA (FlyVis initialisation draws from a CPU generator)."""
    code = """
import json, numpy as np, torch
from sense import vision_flyvis as V
net = V.load_flyvis_net()
with torch.no_grad():
    net.stimulus.zero(2, 1)
    net.stimulus.add_input(torch.full((2, 1, 1, 721), 0.5))
    st = net.forward(net.stimulus(), 0.005, as_states=True)[-1]
a = st.nodes.activity
print(json.dumps(dict(device=str(a.device), finite=bool(torch.isfinite(a).all()))))
"""
    rc, out, err = run_py(code)
    assert rc == 0, err[-3000:]
    assert json.loads(out) == {"device": "cpu", "finite": True}


def test_flyvis_imported_on_cuda_first_is_refused():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: without it a plain `import flyvis` is already on the CPU")
    code = """
import flyvis
from sense import vision_flyvis as V
try:
    V.import_flyvis()
    print("accepted")
except RuntimeError:
    print("refused")
"""
    rc, out, err = run_py(code)
    assert rc == 0, err[-3000:]
    assert out == "refused"
