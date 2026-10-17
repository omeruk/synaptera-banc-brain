"""engine/rate_hybrid.py: brain -> DN bridge -> VNC (prereg/14 stages B, C). Synthetic graphs, no data files: the
clamp equals the registered window formula, causal latency, bridge_off, gradient into the brain, checkpointing,
GPU against CPU."""
import numpy as np
import pytest
import torch

from engine import rate_hybrid as H
from engine import rate_torch as R
from test_rate_torch import random_graph

T, B, L, W = 40, 3, 1, 5


def make(seed=0, n_units=6, device="cpu", dtype=torch.float32, empty=False):
    """Brain: 300 neurons, 20 driven, no inputs; its 'DNs' are 30 non-driven neurons. VNC: 200 neurons, 12 inputs."""
    rng = np.random.default_rng(seed)
    gb = random_graph(seed=seed, n=300, n_in=0, n_drv=20, k_in=6)
    gv = random_graph(seed=seed + 1, n=200, n_in=12, n_drv=0, k_in=6)
    dn = np.arange(100, 130)
    units = [] if empty else [(rng.choice(dn, size=rng.integers(1, 5), replace=False), np.array([2 * u, 2 * u + 1]))
                              for u in range(n_units)]
    base = np.zeros(12)
    base[[0, 1]] = 1.0                     # a HAND baseline that units 0 overwrite, as in f5_reach_run
    base[[-1]] = 1.0                       # one baseline no unit touches
    brain, vnc = R.RateNet(gb, dtype=dtype), R.RateNet(gv, dtype=dtype)
    gen = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for net in (brain, vnc):
            net.theta.copy_(torch.randn(net.theta.shape, generator=gen, dtype=dtype))
            net.phi.copy_(torch.randn(net.phi.shape, generator=gen, dtype=dtype))
    hy = H.RateHybrid(brain, vnc, units, base, latency_intervals=L, window_intervals=W).to(device)
    u = torch.from_numpy(rng.normal(0, 1.0, (T, B, len(gb.drives))).astype(np.float32))
    return hy, units, base, u


def clamps(hy, u, **kw):
    st, _, _ = hy.run(u, record=hy.vnc.graph.inputs, **kw)
    return st


def test_clamp_equals_window_formula():
    hy, units, base, u = make(dtype=torch.float64)
    with torch.no_grad():
        sb, _ = hy.brain.run(u=u)                        # the brain alone: the same dynamics, (T, B, n)
        got = clamps(hy, u).numpy()
    r = np.maximum(sb.numpy(), 0)
    want = np.broadcast_to(base, (T, B, len(base))).copy()
    for k in range(T):
        for f, t in units:
            js = [j for j in range(k - L - W + 1, k - L + 1) if j >= 0]
            v = sum(r[j][:, f].mean(1) for j in js) / W if js else np.zeros(B)
            want[k][:, t] = v[:, None]
    assert np.abs(got - want).max() < 1e-12
    assert np.abs(want[:, :, : 2 * len(units)]).max() > 0.01          # the bridge carries something


def test_clamp_reacts_exactly_latency_intervals_later():
    hy, units, _, u = make(dtype=torch.float64)
    k0 = 17
    f0 = int(units[0][0][0])
    drv = int(np.flatnonzero(hy.brain.graph.drives == f0)[0]) if f0 in hy.brain.graph.drives else None
    u2 = u.clone()
    if drv is None:                                      # perturb every drive: some path reaches the unit's DNs
        u2[k0] += 3.0
    else:
        u2[k0, :, drv] += 3.0
    with torch.no_grad():
        a, b = clamps(hy, u), clamps(hy, u2)
        sa, _ = hy.brain.run(u=u)
        sb, _ = hy.brain.run(u=u2)
    fw = np.unique(np.concatenate([f for f, _ in units]))           # the brain neurons the bridge reads
    first_brain = int(np.flatnonzero((sa - sb)[:, :, fw].abs().amax((1, 2)).numpy() > 0)[0])
    d = (a - b).abs().amax((1, 2)).numpy()
    assert (d[: first_brain + L] == 0).all()
    assert d[first_brain + L] > 0


def test_bridge_off_keeps_the_base_clamp():
    hy, _, base, u = make(empty=True)
    with torch.no_grad():
        got = clamps(hy, u).numpy()
    assert (got == np.broadcast_to(base, got.shape).astype(np.float32)).all()
    assert hy.record()["bridge"]["n_units"] == 0


def test_gradient_reaches_the_brain():
    hy, _, _, u = make()
    st, _, _ = hy.run(u, record=np.arange(150, 160))
    st.square().sum().backward()
    assert float(hy.brain.theta.grad.abs().max()) > 0 and float(hy.vnc.theta.grad.abs().max()) > 0


def test_rule_4_latency_zero_is_refused():
    hy, units, base, _ = make()
    with pytest.raises(ValueError):
        H.RateHybrid(hy.brain, hy.vnc, units, base, latency_intervals=0)


def _grads(device, ce=0):
    hy, _, _, u = make(device=device)
    st, sb, sv = hy.run(u, record=np.arange(150, 160), checkpoint_every=ce)
    st.square().sum().backward()
    return [x.detach().cpu() for x in (st, sb, sv, hy.brain.theta.grad, hy.brain.phi.grad, hy.vnc.theta.grad,
                                       hy.vnc.phi.grad)]


def test_checkpointing_is_identical():
    torch.use_deterministic_algorithms(True)
    try:
        a, b = _grads("cpu"), _grads("cpu", ce=7)
    finally:
        torch.use_deterministic_algorithms(False)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_equals_cpu():
    c, g = _grads("cpu"), _grads("cuda")
    for x, y in zip(c[:3], g[:3]):
        assert float((x - y).abs().max()) < 1e-5
    for x, y in zip(c[3:], g[3:]):
        assert float((x - y).abs().max() / x.abs().max()) < 1e-5
