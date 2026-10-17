"""engine/rate_torch.py and train/checkpoint.py: prereg/14 gate 3 (forward against NumPy, sparse backward against dense
float64, causality, resume), on synthetic 2,000-neuron graphs (no data files)."""
from collections import deque

import numpy as np
import pytest
import torch

from engine import rate_torch as R
from train import checkpoint as C

N, N_IN, N_DRV, K_IN = 2000, 40, 20, 12


def random_graph(seed=0, n=N, n_in=N_IN, n_drv=N_DRV, k_in=K_IN):
    """Every non-input neuron gets k_in distinct presynaptic partners; inputs receive nothing; one sign per neuron."""
    rng = np.random.default_rng(seed)
    inputs = np.arange(n_in)
    drives = np.arange(n_in, n_in + n_drv)
    sign_of = rng.choice([-1, 1], size=n, p=[0.4, 0.6])
    pre, post = [], []
    for i in range(n_in, n):
        cand = rng.choice(n - 1, size=k_in, replace=False)
        cand[cand >= i] += 1                      # no self-loop
        pre.append(cand)
        post.append(np.full(k_in, i))
    pre, post = np.concatenate(pre), np.concatenate(post)
    n_syn = rng.integers(1, 30, size=len(pre))
    return R.RateGraph.from_arrays(n, pre, post, n_syn, sign_of[pre], inputs, drives)


@pytest.fixture(scope="module")
def graph():
    return random_graph()


def drive(T, batch, g, seed=1, dtype=np.float32):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (T, batch, len(g.inputs))).astype(dtype)
    u = rng.normal(0, 0.5, (T, batch, len(g.drives))).astype(dtype)
    return x, u


def randomise(net, seed=2):
    """Gains and leaks away from their initial values, so the checks do not rest on g = 1."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.theta.copy_(torch.randn(net.theta.shape, generator=gen, dtype=net.theta.dtype))
        net.phi.copy_(torch.randn(net.phi.shape, generator=gen, dtype=net.phi.dtype))


def test_normalisation_signs_and_initialisation(graph):
    absum = np.bincount(graph.post, weights=np.abs(graph.w), minlength=graph.n)
    has_in = np.bincount(graph.post, minlength=graph.n) > 0
    assert np.allclose(absum[has_in], 1.0) and (absum[~has_in] == 0).all()
    assert (np.sign(graph.w) == graph.sign).all()
    net = R.RateNet(graph)
    assert torch.equal(net.gains(), torch.ones(graph.n_edges))
    assert torch.allclose(net.leaks(), torch.full((graph.n,), R.LAM_0))
    assert graph.n_trainable == graph.n_edges + N - N_IN
    assert net.record()["n_trainable"] == graph.n_trainable


@pytest.mark.parametrize("bad", ["dup", "self", "onto_input", "two_signs", "zero_syn", "driven_input"])
def test_graph_validation(bad):
    pre, post, n_syn, sign, inputs, drives = [2, 3], [3, 2], [1, 1], [1, 1], [0], [1]
    if bad == "dup":
        pre, post, n_syn, sign = [2, 2], [3, 3], [1, 1], [1, 1]
    elif bad == "self":
        pre, post = [2, 3], [2, 2]
    elif bad == "onto_input":
        pre, post = [2, 3], [0, 2]
    elif bad == "two_signs":
        pre, post, sign = [2, 2], [3, 1], [1, -1]
    elif bad == "zero_syn":
        n_syn = [0, 1]
    elif bad == "driven_input":
        drives = [0]
    with pytest.raises(ValueError):
        R.RateGraph.from_arrays(4, pre, post, n_syn, sign, inputs, drives)


def test_forward_matches_numpy_300_intervals(graph):
    """Gate 3: max |torch - NumPy| < 1e-5 in float32 over 300 intervals."""
    net = R.RateNet(graph)
    randomise(net)
    x, u = drive(300, 2, graph)
    with torch.no_grad():
        st, _ = net.run(torch.from_numpy(x), torch.from_numpy(u))
    ref = R.reference_numpy(graph, net.gains().detach().numpy(), net.leaks().detach().numpy(), x, u)
    err = float(np.abs(st.numpy() - ref).max())
    assert err < 1e-5, err
    assert float(np.abs(ref).max()) > 0.1                    # the network is actually active


def rel_err(a, b):
    a, b = a.detach().numpy(), b.detach().numpy()
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_sparse_backward_matches_dense_float64(graph):
    """Gate 3: custom sparse backward against dense autograd, float64, max relative error < 1e-6."""
    T, batch = 40, 2
    net = R.RateNet(graph, dtype=torch.float64)
    randomise(net)
    x, u = drive(T, batch, graph, dtype=np.float64)
    x, u = torch.from_numpy(x), torch.from_numpy(u)
    coef = torch.from_numpy(np.random.default_rng(3).normal(size=(T, batch, graph.n)))
    st, _ = net.run(x, u)
    (st * coef).sum().backward()
    th = net.theta.detach().clone().requires_grad_(True)
    ph = net.phi.detach().clone().requires_grad_(True)
    ref = R.dense_reference_torch(graph, th, ph, x, u)
    assert float((st - ref).detach().abs().max()) < 1e-12
    (ref * coef).sum().backward()
    assert rel_err(net.theta.grad, th.grad) < 1e-6
    assert rel_err(net.phi.grad, ph.grad) < 1e-6
    assert float(net.phi.grad[torch.from_numpy(graph.inputs)].abs().max()) == 0.0   # inputs have no leak


def test_autograd_gradcheck_small():
    g = random_graph(seed=5, n=60, n_in=5, n_drv=3, k_in=4)
    st = R._Structure.from_graph(g)
    vals = torch.from_numpy(g.w).clone().requires_grad_(True)
    s = torch.randn(g.n, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, z: R._SparseMM.apply(v, z, st), (vals, s))


def test_edge_chunking_does_not_change_the_gradient(graph, monkeypatch):
    net = R.RateNet(graph, dtype=torch.float64)
    randomise(net)
    x, u = drive(10, 2, graph, dtype=np.float64)

    def grad():
        net.zero_grad()
        st, _ = net.run(torch.from_numpy(x), torch.from_numpy(u))
        st.square().sum().backward()
        return net.theta.grad.clone()

    g_full = grad()
    monkeypatch.setattr(R, "EDGE_CHUNK", 1000)
    assert torch.equal(grad(), g_full)


def test_gradient_checkpointing_is_identical(graph):
    torch.use_deterministic_algorithms(True)
    try:
        x, u = drive(60, 2, graph)
        grads = []
        for ce in (0, 25):
            net = R.RateNet(graph)
            randomise(net)
            st, _ = net.run(torch.from_numpy(x), torch.from_numpy(u), record=np.arange(100, 140), checkpoint_every=ce)
            st.square().sum().backward()
            grads.append((st.detach(), net.theta.grad.clone(), net.phi.grad.clone()))
        for a, b in zip(*grads):
            assert torch.equal(a, b)
    finally:
        torch.use_deterministic_algorithms(False)


def hops_from(graph, sources):
    """Shortest synapse-hop distance from any source node (BFS on pre -> post)."""
    out = [[] for _ in range(graph.n)]
    for a, b in zip(graph.pre, graph.post):
        out[a].append(b)
    d = np.full(graph.n, -1)
    q = deque()
    for s in sources:
        d[s] = 0
        q.append(s)
    while q:
        a = q.popleft()
        for b in out[a]:
            if d[b] < 0:
                d[b] = d[a] + 1
                q.append(b)
    return d


def test_causality_clamp_and_drive(graph):
    """Gate 3, causality: a perturbation at interval k changes nothing before k; a neuron d hops downstream of the
    perturbed clamped input first changes at k + d - 1; a perturbed drive node changes at k. Any readout dA is a fixed
    linear combination of states of the same interval, so it inherits this."""
    T, k = 30, 10
    net = R.RateNet(graph, dtype=torch.float64)
    randomise(net)
    x, u = drive(T, 1, graph, dtype=np.float64)
    with torch.no_grad():
        base, _ = net.run(torch.from_numpy(x), torch.from_numpy(u))
        x2 = x.copy()
        x2[k, 0, 3] += 0.5
        pert, _ = net.run(torch.from_numpy(x2), torch.from_numpy(u))
        u2 = u.copy()
        u2[k, 0, 2] += 0.5
        pert_u, _ = net.run(torch.from_numpy(x), torch.from_numpy(u2))
    for p, src, lag in ((pert, graph.inputs[3], -1), (pert_u, graph.drives[2], 0)):
        diff = (p - base).abs()[:, 0].numpy()
        assert diff[:k].max() == 0.0
        d = hops_from(graph, [src])
        changed = diff.max(0) > 0
        first = np.where(changed, np.argmax(diff > 0, axis=0), -1)
        near = (d >= 1) & (d <= 3)
        if lag == 0:
            near |= d == 0
        assert near.sum() > 10
        expected = k + d[near] + lag
        assert (first[near] == expected).all()
        assert not changed[d < 0].any()          # unreachable neurons never change
    readout = np.random.default_rng(4).normal(size=(graph.n,))
    dA = (pert - base)[:, 0].numpy() @ readout
    assert np.abs(dA[:k]).max() == 0.0


def test_inputs_are_clamped_and_zero_input_stays_at_rest(graph):
    net = R.RateNet(graph)
    randomise(net)
    x, u = drive(20, 3, graph)
    with torch.no_grad():
        st, fin = net.run(torch.from_numpy(x), torch.from_numpy(u))
        assert torch.equal(st[:, :, torch.from_numpy(graph.inputs)], torch.from_numpy(x))
        z, _ = net.run(torch.zeros(20, 3, N_IN))
    assert float(z.abs().max()) == 0.0
    assert fin.shape == (3, graph.n)
    assert float(st.abs().max()) < 1.0


def test_induced_subgraph_renormalises(graph):
    keep = np.arange(0, graph.n, 2)
    sub = graph.induced(keep)
    assert sub.n == len(keep)
    absum = np.bincount(sub.post, weights=np.abs(sub.w), minlength=sub.n)
    assert np.allclose(absum[absum > 0], 1.0)
    assert len(sub.inputs) == N_IN // 2


def _train(run_dir, steps, resume, g):
    """A toy optimisation loop with the prereg/14 optimiser (Adam, clip 1.0) and a seeded data order."""
    net = R.RateNet(g)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    data_rng = np.random.default_rng(7)
    start = 0
    if resume:
        ck = C.load(C.latest(run_dir), net, opt)
        start = ck["step"]
        data_rng.bit_generator.state = ck["extra"]["data_rng"]
    target = torch.from_numpy(np.random.default_rng(8).uniform(-0.5, 0.5, (16, 30, 8)).astype(np.float32))
    rec = np.arange(200, 208)
    for step in range(start, steps):
        idx = data_rng.choice(16, size=4, replace=False)
        x = torch.rand(30, 4, len(g.inputs))     # torch RNG: restored from the checkpoint
        opt.zero_grad()
        st, _ = net.run(x, record=rec)
        loss = (st - target[idx].transpose(0, 1)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        C.save(run_dir, step + 1, net, opt, extra=dict(data_rng=data_rng.bit_generator.state))
    return net


def test_resume_is_bit_identical(tmp_path):
    """Gate 3, resume: interrupt at step k, resume, and the parameters at step 2k equal an uninterrupted run bit for
    bit (CPU, fixed thread count, deterministic algorithms)."""
    g = random_graph(seed=11, n=400, n_in=10, n_drv=0, k_in=8)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(0)
        full = _train(tmp_path / "a", 8, False, g)
        torch.manual_seed(0)
        _train(tmp_path / "b", 4, False, g)
        torch.manual_seed(12345)                # the resumed run must restore the RNG itself
        resumed = _train(tmp_path / "b", 8, True, g)
    finally:
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(threads)
    assert torch.equal(full.theta, resumed.theta) and torch.equal(full.phi, resumed.phi)
    assert not torch.equal(full.theta, torch.zeros_like(full.theta))


def test_checkpoint_keeps_last_three_and_best(tmp_path):
    net = R.RateNet(random_graph(seed=12, n=50, n_in=3, n_drv=0, k_in=3))
    opt = torch.optim.Adam(net.parameters())
    for s in range(1, 7):
        C.save(tmp_path, s, net, opt, best=(s == 2))
    assert [p.name for p in C.step_checkpoints(tmp_path)] == ["step_00000004.pt", "step_00000005.pt", "step_00000006.pt"]
    assert (tmp_path / "best.pt").exists() and not list(tmp_path.glob("*.tmp"))
    assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 2


def test_sddmm_edge_gradient_matches_dense_float64(graph):
    """The opt-in edge_grad="sddmm" passes the gate-3 backward check (dense float64 autograd, relative error < 1e-6)."""
    T, batch = 40, 2
    net = R.RateNet(graph, dtype=torch.float64, edge_grad="sddmm")
    randomise(net)
    x, u = drive(T, batch, graph, dtype=np.float64)
    x, u = torch.from_numpy(x), torch.from_numpy(u)
    coef = torch.from_numpy(np.random.default_rng(3).normal(size=(T, batch, graph.n)))
    st, _ = net.run(x, u)
    (st * coef).sum().backward()
    th = net.theta.detach().clone().requires_grad_(True)
    ph = net.phi.detach().clone().requires_grad_(True)
    (R.dense_reference_torch(graph, th, ph, x, u) * coef).sum().backward()
    assert rel_err(net.theta.grad, th.grad) < 1e-6
    assert rel_err(net.phi.grad, ph.grad) < 1e-6
    assert net.record()["edge_grad"] == "sddmm"


def test_sddmm_matches_chunked_float32(graph):
    """The opt-in edge_grad="sddmm" against the registered default "chunked" in float32 over 300 intervals: the forward
    is the same code (bit-identical), the gradients agree within float32 tolerance (max relative error < 1e-5)."""
    x, u = drive(300, 4, graph)
    coef = torch.from_numpy(np.random.default_rng(3).normal(size=(300, 4, graph.n)).astype(np.float32))
    out = {}
    for eg in R.EDGE_GRADS:
        net = R.RateNet(graph, edge_grad=eg)
        randomise(net)
        st, last = net.run(torch.from_numpy(x), torch.from_numpy(u))
        (st * coef).sum().backward()
        out[eg] = (st.detach(), last.detach(), net.theta.grad, net.phi.grad)
    c, s = out["chunked"], out["sddmm"]
    assert torch.equal(c[0], s[0]) and torch.equal(c[1], s[1])
    assert rel_err(s[2], c[2]) < 1e-5 and rel_err(s[3], c[3]) < 1e-5
    assert float(c[2].abs().max()) > 0


def test_sddmm_gradcheck_small():
    g = random_graph(seed=5, n=60, n_in=5, n_drv=3, k_in=4)
    st = R._Structure.from_graph(g, edge_grad="sddmm")
    vals = torch.from_numpy(g.w).clone().requires_grad_(True)
    s = torch.randn(g.n, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, z: R._SparseMM.apply(v, z, st), (vals, s))


def test_unknown_edge_gradient_is_refused(graph):
    with pytest.raises(ValueError):
        R.RateNet(graph, edge_grad="dense")
