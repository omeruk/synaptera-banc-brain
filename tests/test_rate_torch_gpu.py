"""engine/rate_torch.py and train/checkpoint.py on CUDA: forward and backward equal the CPU within float32 tolerance,
gradient checkpointing is exact, and a checkpoint resume on the GPU is bit-identical (deterministic algorithms; the
CUDA RNG state is part of the checkpoint). Synthetic graphs, no data files. Skipped without CUDA."""
import numpy as np
import pytest
import torch

from engine import rate_torch as R
from test_rate_torch import drive, random_graph, randomise
from train import checkpoint as C

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def graph():
    return random_graph()


def run_on(device, graph, x, u, coef, checkpoint_every=0, edge_grad="chunked"):
    net = R.RateNet(graph, edge_grad=edge_grad)
    randomise(net)
    net.to(device)
    st, last = net.run(torch.from_numpy(x), torch.from_numpy(u), checkpoint_every=checkpoint_every)
    (st * coef.to(device)).sum().backward()
    return dict(states=st.detach().cpu(), last=last.detach().cpu(), theta=net.theta.grad.cpu(), phi=net.phi.grad.cpu(),
                device=st.device.type)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.fixture(scope="module")
def cpu_and_gpu(graph):
    x, u = drive(300, 4, graph)
    coef = torch.from_numpy(np.random.default_rng(3).normal(size=(300, 4, graph.n)).astype(np.float32))
    return run_on("cpu", graph, x, u, coef), run_on("cuda", graph, x, u, coef), (x, u, coef)


def test_gpu_forward_equals_cpu(cpu_and_gpu):
    """Same tolerance as the gate-3 forward check against NumPy: max |GPU - CPU| < 1e-5 over 300 intervals."""
    c, g, _ = cpu_and_gpu
    assert g["device"] == "cuda"
    assert float((g["states"] - c["states"]).abs().max()) < 1e-5
    assert float((g["last"] - c["last"]).abs().max()) < 1e-5
    assert float(c["states"].abs().max()) > 0.1


def test_gpu_backward_equals_cpu(cpu_and_gpu):
    """float32 gradients over 300 intervals: max relative error < 1e-5 (theta and phi)."""
    c, g, _ = cpu_and_gpu
    assert rel(g["theta"], c["theta"]) < 1e-5
    assert rel(g["phi"], c["phi"]) < 1e-5
    assert float(c["theta"].abs().max()) > 0


def test_gpu_gradient_checkpointing_is_identical(graph, cpu_and_gpu):
    _, _, (x, u, coef) = cpu_and_gpu
    torch.use_deterministic_algorithms(True)
    try:
        a = run_on("cuda", graph, x[:60], u[:60], coef[:60])
        b = run_on("cuda", graph, x[:60], u[:60], coef[:60], checkpoint_every=25)
    finally:
        torch.use_deterministic_algorithms(False)
    for k in ("states", "last", "theta", "phi"):
        assert torch.equal(a[k], b[k]), k


def _train_gpu(run_dir, steps, resume, g, edge_grad="chunked"):
    """The toy loop of test_rate_torch._train on the GPU; the inputs come from the CUDA RNG."""
    net = R.RateNet(g, edge_grad=edge_grad).to("cuda")
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    data_rng = np.random.default_rng(7)
    start = 0
    if resume:
        ck = C.load(C.latest(run_dir), net, opt)
        start = ck["step"]
        data_rng.bit_generator.state = ck["extra"]["data_rng"]
    target = torch.from_numpy(np.random.default_rng(8).uniform(-0.5, 0.5, (16, 30, 8)).astype(np.float32)).cuda()
    rec = np.arange(200, 208)
    for step in range(start, steps):
        idx = data_rng.choice(16, size=4, replace=False)
        x = torch.rand(30, 4, len(g.inputs), device="cuda")     # CUDA RNG: restored from the checkpoint
        opt.zero_grad()
        st, _ = net.run(x, record=rec)
        loss = (st - target[idx].transpose(0, 1)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        C.save(run_dir, step + 1, net, opt, extra=dict(data_rng=data_rng.bit_generator.state))
    return net


@pytest.mark.parametrize("edge_grad", R.EDGE_GRADS)
def test_gpu_resume_is_bit_identical(tmp_path, edge_grad):
    """Gate 3, resume, on the GPU: interrupt at step k, resume, and the parameters at step 2k equal an uninterrupted
    run bit for bit."""
    g = random_graph(seed=11, n=400, n_in=10, n_drv=0, k_in=8)
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(0)
        full = _train_gpu(tmp_path / "a", 8, False, g, edge_grad)
        torch.manual_seed(0)
        _train_gpu(tmp_path / "b", 4, False, g, edge_grad)
        assert "cuda" in torch.load(C.latest(tmp_path / "b"), weights_only=False)["rng"]
        torch.manual_seed(12345)                # the resumed run must restore the CUDA RNG itself
        resumed = _train_gpu(tmp_path / "b", 8, True, g, edge_grad)
    finally:
        torch.use_deterministic_algorithms(False)
    assert full.theta.device.type == resumed.theta.device.type == "cuda"
    assert torch.equal(full.theta, resumed.theta) and torch.equal(full.phi, resumed.phi)
    assert not torch.equal(full.theta, torch.zeros_like(full.theta))


def test_gpu_sddmm_equals_cpu_chunked(graph, cpu_and_gpu):
    """The opt-in edge_grad="sddmm" on the GPU against the default on the CPU: float32 tolerance; repeated runs in
    deterministic mode are bit-identical."""
    c, _, (x, u, coef) = cpu_and_gpu
    torch.use_deterministic_algorithms(True)
    try:
        a = run_on("cuda", graph, x, u, coef, edge_grad="sddmm")
        b = run_on("cuda", graph, x, u, coef, edge_grad="sddmm")
    finally:
        torch.use_deterministic_algorithms(False)
    assert float((a["states"] - c["states"]).abs().max()) < 1e-5
    assert rel(a["theta"], c["theta"]) < 1e-5 and rel(a["phi"], c["phi"]) < 1e-5
    for k in ("states", "theta", "phi"):
        assert torch.equal(a[k], b[k]), k


def test_gpu_sddmm_equals_gpu_chunked(graph, cpu_and_gpu):
    """On the GPU, sddmm against the registered chunked default: identical forward, gradients within float32
    tolerance."""
    _, g, (x, u, coef) = cpu_and_gpu
    s = run_on("cuda", graph, x, u, coef, edge_grad="sddmm")
    assert torch.equal(s["states"], g["states"]) and torch.equal(s["last"], g["last"])
    assert rel(s["theta"], g["theta"]) < 1e-5 and rel(s["phi"], g["phi"]) < 1e-5
