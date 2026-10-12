"""F0 bisection step c3: fingerprint the built network (v1 FlightBrain scratch copy vs new engine)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
which, out = sys.argv[1], sys.argv[2]


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


if which == "v1":
    sys.path.insert(0, sys.argv[3])
    from flight.brain import FlightBrain
    b = FlightBrain(seed=3, olfaction=False, vision_boundary=True, verbose=False)
    neu, syn, net = b.neu, b.syn, b.net
    pois = {n: b.inputs[n] for n in b.inputs}
    objs = {o.name: o for o in net.objects}
    ps = {n: objs[f"poi_syn_{n}"] for n in b.inputs}
else:
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from connectome.adapters import flywire783 as fw
    from engine.lif_brian2 import Brian2LIFEngine
    cx = fw.load_connectome(hash_files=False)
    ids = {}
    d = np.load(sys.argv[3])
    eng = Brian2LIFEngine(cx, input_groups={"sugar": d["idx_sugar"], "vbnd_L": d["idx_L"], "vbnd_R": d["idx_R"]})
    neu, syn = eng.neu, eng.syn
    ps = eng.poisson_synapses
    pois = eng._poisson
res = dict(
    n=len(neu), n_syn=len(syn), syn_i=h(syn.i[:]), syn_j=h(syn.j[:]), syn_w=h(np.asarray(syn.w_[:])),
    delay=float(np.ravel(np.asarray(syn.delay[:]))[0]),
    rfc=h(np.asarray(neu.rfc_[:])), n_rfc0=int((np.asarray(neu.rfc_[:]) == 0).sum()),
    v=h(np.asarray(neu.v_[:])), g=h(np.asarray(neu.g_[:])),
    poisson={n: dict(j=h(np.asarray(s.j[:])), w=h(np.asarray(s.w_[:])), n=len(s),
                     delay=float(np.ravel(np.asarray(s.delay[:]))[0])) for n, s in ps.items()},
    w_stats=dict(sum=float(np.asarray(syn.w_[:]).sum()), n_zero=int((np.asarray(syn.w_[:]) == 0).sum())),
    namespace_keys=sorted(k for k in getattr(neu, "namespace", {}) if isinstance(k, str))[:30]
    if hasattr(neu, "namespace") else None)
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "namespace_keys"}, indent=1))
