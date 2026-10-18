"""prereg/18: every F5 training run uses edge_grad="sddmm", with gradient checkpointing every 50 intervals in stages B
and C (none in A). Checkpointing must not change any result in deterministic mode: states, losses, gradients and the
parameters after a few optimiser steps are bit-identical with and without it, for the stage-A engine (RateNet) and the
stage-B/C composite (RateHybrid), on the CPU and on the GPU. T = 130 covers full and partial 50-interval segments."""
import numpy as np
import pytest
import torch

from engine import rate_torch as R
from test_rate_hybrid import make
from test_rate_torch import drive, random_graph, randomise

CKPT, T = 50, 130
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def _train_net(device, ce, steps=3):
    g = random_graph()
    net = R.RateNet(g, edge_grad="sddmm")
    randomise(net)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, u = drive(T, 3, g)
    x, u = torch.from_numpy(x).to(device), torch.from_numpy(u).to(device)
    out = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        st, _ = net.run(x, u, record=np.arange(100, 140), checkpoint_every=ce)
        loss = st.square().mean()
        loss.backward()
        out += [st.detach().cpu(), loss.detach().cpu(), net.theta.grad.cpu().clone(), net.phi.grad.cpu().clone()]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return out + [net.theta.detach().cpu(), net.phi.detach().cpu()]


def _train_hybrid(device, ce, steps=3):
    hy, _, _, u = make(device=device)
    hy.brain.edge_grad = hy.vnc.edge_grad = "sddmm"
    u = torch.cat([u] * 4)[:T]                        # make() gives 40 intervals
    opt = torch.optim.Adam(hy.parameters(), lr=1e-3)
    out = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        st, sb, sv = hy.run(u, record=np.arange(150, 160), checkpoint_every=ce)
        loss = st.square().mean()
        loss.backward()
        out += [st.detach().cpu(), sb.detach().cpu(), sv.detach().cpu(), loss.detach().cpu()]
        out += [p.grad.cpu().clone() for p in hy.parameters()]
        torch.nn.utils.clip_grad_norm_(hy.parameters(), 1.0)
        opt.step()
    return out + [p.detach().cpu() for p in hy.parameters()]


@pytest.mark.parametrize("device", DEVICES)
def test_stage_a_engine_checkpointing_is_bit_identical(device):
    a, b = _train_net(device, 0), _train_net(device, CKPT)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_stage_bc_hybrid_checkpointing_is_bit_identical(device):
    a, b = _train_hybrid(device, 0), _train_hybrid(device, CKPT)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert float(a[-1].abs().max()) > 0
