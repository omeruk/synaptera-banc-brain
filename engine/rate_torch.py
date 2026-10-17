"""F5 sparse rate network (docs/10_TRAINING.md section 2.1, prereg/14 section 2). Label `TRAINED`.

Each neuron i has a signed state s_i in (-1, 1). One update per 5 ms coupling interval:

    s  <-  (1 - lam) * s  +  lam * tanh(W_eff s + u)
    W_eff[i, j] = g_ij * sign_j * n_ij / N_i        (only pairs with n_ij > 0; N_i = sum_j n_ij)
    g_ij  = G_MAX * sigmoid(theta_ij)                (theta init 0 -> g = 1)
    lam_i = LAM_MIN + (LAM_MAX - LAM_MIN) * sigmoid(phi_i)   (phi init -> lam = LAM_0)

Input nodes (the BANC DNs) are clamped: their state is set from outside each interval and they have no leak. Interval
k of :meth:`RateNet.run` is

    s~_k = s_{k-1} with the input nodes set to x_k           (the clamp of interval k)
    s_k  = (1 - lam) s~_k + lam tanh(W_eff s~_k + u_k)       on every other neuron; s_k = x_k on the input nodes

so a neuron d synapse hops downstream of an input node first responds to the clamp of interval k at interval
k + d - 1. The clamp itself must come from earlier intervals (the bridge latency); that is the caller's job (rule 4).

The sparse product has a custom backward (as in flyhard): the gradient with respect to the state uses a stored
transpose, and the per-edge gradient is computed in chunks of edges so that no E x batch matrix is ever held
(``edge_grad="chunked"``, the default and the one of every record so far). ``edge_grad="sddmm"`` computes the same
per-edge sums with one sampled dense-dense product (``torch.sparse.sampled_addmm``, cuSPARSE on CUDA); it is equal
within float rounding, not bit for bit, and is used only when chosen explicitly.

The state is kept as (neurons, batch) inside; the public API takes and returns (time, batch, ...).

Device: the network runs where its parameters are (``net.to("cuda")``); the CSR index arrays are non-persistent buffers
that move with it, and inputs are moved to that device. Checkpoints (state_dict) are the same on every device.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

G_MAX = 2.0                    # edge gain bound; gain init 1 (theta = 0)
LAM_MIN, LAM_MAX = 0.05, 0.95  # leak bounds (flyhard)
LAM_0 = 0.25                   # 5 ms update / Shiu membrane time constant 20 ms
UPDATES_PER_INTERVAL = 1       # one update per 5 ms coupling interval
R_IN = 100.0                   # Hz per unit of state or drive; R_OUT = R_IN
EDGE_CHUNK = 1 << 20           # edges per chunk of the per-edge gradient
EDGE_GRADS = ("chunked", "sddmm")

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


def phi_0() -> float:
    """phi such that lam = LAM_0."""
    p = (LAM_0 - LAM_MIN) / (LAM_MAX - LAM_MIN)
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class RateGraph:
    """A fixed, signed, input-normalised graph in CSR order (sorted by post, then pre).

    ``w`` = sign_j * n_ij / N_i per edge (float64). ``inputs`` are the clamped nodes, ``drives`` the nodes that receive
    an external drive u. No edge may end on an input node (an input node receives nothing)."""
    n: int
    pre: np.ndarray
    post: np.ndarray
    n_syn: np.ndarray
    sign: np.ndarray
    w: np.ndarray
    inputs: np.ndarray
    drives: np.ndarray

    @classmethod
    def from_arrays(cls, n, pre, post, n_syn, sign, inputs=(), drives=()) -> "RateGraph":
        pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        n_syn, sign = np.asarray(n_syn, np.int64), np.asarray(sign, np.int64)
        inputs, drives = np.asarray(inputs, np.int64), np.asarray(drives, np.int64)
        if not (len(pre) == len(post) == len(n_syn) == len(sign)):
            raise ValueError("edge arrays differ in length")
        if len(pre) and (pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= n):
            raise ValueError("edge index out of range")
        if (n_syn <= 0).any():
            raise ValueError("every edge needs n_syn > 0 (fixed sparsity)")
        if not np.isin(sign, (-1, 1)).all():
            raise ValueError("sign must be -1 or +1 per edge")
        if (pre == post).any():
            raise ValueError("self-loops are not allowed")
        key = post * n + pre
        o = np.argsort(key, kind="stable")
        pre, post, n_syn, sign, key = pre[o], post[o], n_syn[o], sign[o], key[o]
        if len(key) > 1 and (np.diff(key) == 0).any():
            raise ValueError("duplicate (pre, post) pairs")
        presign = np.zeros(n, np.int64)
        presign[pre] = sign
        if (presign[pre] != sign).any():
            raise ValueError("a presynaptic neuron has more than one sign (Dale's law)")
        if len(np.unique(inputs)) != len(inputs) or len(np.unique(drives)) != len(drives):
            raise ValueError("duplicate input or drive index")
        if np.isin(post, inputs).any():
            raise ValueError("an edge ends on a clamped input node")
        if np.isin(drives, inputs).any():
            raise ValueError("a node cannot be both clamped and driven")
        n_in = np.bincount(post, weights=n_syn, minlength=n)
        w = sign * n_syn / n_in[post]
        return cls(n, pre, post, n_syn, sign, w, inputs, drives)

    @classmethod
    def from_connectome(cls, cx, inputs=(), drives=()) -> "RateGraph":
        """From a signed :class:`connectome.schema.Connectome`. Silenced edges (sign 0) are refused."""
        pre, post = cx.edge_endpoints()
        e = cx.edges
        return cls.from_arrays(cx.n_neurons, pre, post, e["n_syn"].to_numpy(), e["sign"].to_numpy(), inputs, drives)

    @property
    def n_edges(self) -> int:
        return int(len(self.pre))

    @property
    def n_trainable(self) -> int:
        """One gain per edge plus one leak per non-input neuron (docs/10 section 2.2)."""
        return self.n_edges + self.n - int(len(self.inputs))

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in (np.int64(self.n), self.pre, self.post, self.n_syn, self.sign, self.inputs, self.drives):
            h.update(np.ascontiguousarray(a, np.int64).tobytes())
        return h.hexdigest()

    def induced(self, keep) -> "RateGraph":
        """The subgraph on the neurons ``keep`` (sorted), with N_i recomputed on the kept edges."""
        keep = np.unique(np.asarray(keep, np.int64))
        new = np.full(self.n, -1, np.int64)
        new[keep] = np.arange(len(keep))
        m = (new[self.pre] >= 0) & (new[self.post] >= 0)
        inputs = new[self.inputs][new[self.inputs] >= 0]
        drives = new[self.drives][new[self.drives] >= 0]
        return RateGraph.from_arrays(len(keep), new[self.pre[m]], new[self.post[m]], self.n_syn[m], self.sign[m],
                                     inputs, drives)


class _Structure:
    """CSR index arrays of W (rows = post) and of its stored transpose (rows = pre)."""
    NAMES = ("pre", "post", "crow", "perm_t", "col_t", "crow_t")

    def __init__(self, n, pre, post, crow, perm_t, col_t, crow_t, edge_grad="chunked"):
        if edge_grad not in EDGE_GRADS:
            raise ValueError(f"edge_grad must be one of {EDGE_GRADS}")
        self.n = n
        self.edge_grad = edge_grad
        self.pre, self.post, self.crow = pre, post, crow
        self.perm_t, self.col_t, self.crow_t = perm_t, col_t, crow_t

    @classmethod
    def from_graph(cls, g: RateGraph, device=None, edge_grad="chunked") -> "_Structure":
        return cls(g.n, *(cls.arrays(g)[k].to(device) for k in cls.NAMES), edge_grad=edge_grad)

    @staticmethod
    def arrays(g: RateGraph) -> dict:
        n = g.n
        perm = np.argsort(g.pre * n + g.post, kind="stable")
        a = dict(pre=g.pre, post=g.post, crow=np.concatenate([[0], np.cumsum(np.bincount(g.post, minlength=n))]),
                 perm_t=perm, col_t=g.post[perm], crow_t=np.concatenate([[0], np.cumsum(np.bincount(g.pre, minlength=n))]))
        return {k: torch.from_numpy(np.ascontiguousarray(v, np.int64)) for k, v in a.items()}

    def matmul(self, vals, s):
        w = torch.sparse_csr_tensor(self.crow, self.pre, vals, (self.n, self.n), check_invariants=False)
        return w @ s

    def matmul_t(self, vals, s):
        w = torch.sparse_csr_tensor(self.crow_t, self.col_t, vals[self.perm_t], (self.n, self.n), check_invariants=False)
        return w @ s


class _SparseMM(torch.autograd.Function):
    """y = W s with W[post_e, pre_e] = vals_e; s and y are (neurons, batch)."""

    @staticmethod
    def forward(ctx, vals, s, st):
        ctx.st = st
        ctx.save_for_backward(vals, s)
        return st.matmul(vals, s)

    @staticmethod
    def backward(ctx, gy):
        vals, s = ctx.saved_tensors
        st = ctx.st
        gy = gy.contiguous()
        g_vals = g_s = None
        if ctx.needs_input_grad[0]:
            if st.edge_grad == "sddmm":            # rows = post, columns = pre: the CSR edge order
                mask = torch.sparse_csr_tensor(st.crow, st.pre, torch.zeros_like(vals), (st.n, st.n),
                                               check_invariants=False)
                g_vals = torch.sparse.sampled_addmm(mask, gy, s.T, beta=0.0, alpha=1.0).values()
            else:
                g_vals = torch.empty_like(vals)
                for a in range(0, len(vals), EDGE_CHUNK):
                    b = min(a + EDGE_CHUNK, len(vals))
                    g_vals[a:b] = (gy[st.post[a:b]] * s[st.pre[a:b]]).sum(1)
        if ctx.needs_input_grad[1]:
            g_s = st.matmul_t(vals, gy)
        return g_vals, g_s, None


class RateNet(nn.Module):
    """The trainable rate network on a fixed :class:`RateGraph`. Parameters: ``theta`` (one per edge, gain) and
    ``phi`` (one per neuron, leak; the entries of input nodes are unused and receive no gradient)."""

    def __init__(self, graph: RateGraph, dtype=torch.float32, edge_grad: str = "chunked"):
        super().__init__()
        if edge_grad not in EDGE_GRADS:
            raise ValueError(f"edge_grad must be one of {EDGE_GRADS}")
        self.graph = graph
        self.dtype = dtype
        self.edge_grad = edge_grad
        for k, v in _Structure.arrays(graph).items():
            self.register_buffer(f"_st_{k}", v, persistent=False)
        self.theta = nn.Parameter(torch.zeros(graph.n_edges, dtype=dtype))
        self.phi = nn.Parameter(torch.full((graph.n,), phi_0(), dtype=dtype))
        self.register_buffer("w", torch.from_numpy(graph.w).to(dtype))
        self.register_buffer("inputs", torch.from_numpy(graph.inputs))
        self.register_buffer("drives", torch.from_numpy(graph.drives))

    @property
    def device(self):
        return self.theta.device

    @property
    def _st(self):
        return _Structure(self.graph.n, *(getattr(self, f"_st_{k}") for k in _Structure.NAMES), edge_grad=self.edge_grad)

    def gains(self):
        return G_MAX * torch.sigmoid(self.theta)

    def leaks(self):
        return LAM_MIN + (LAM_MAX - LAM_MIN) * torch.sigmoid(self.phi)

    def initial_state(self, batch: int):
        return torch.zeros(self.graph.n, batch, dtype=self.dtype, device=self.device)

    def _step(self, s, vals, lam, x, u, st=None):
        if x is not None and len(self.inputs):
            s = s.index_copy(0, self.inputs, x)
        h = _SparseMM.apply(vals, s, self._st if st is None else st)
        if u is not None and len(self.drives):
            h = h.index_add(0, self.drives, u)
        s_new = (1.0 - lam) * s + lam * torch.tanh(h)
        if x is not None and len(self.inputs):
            s_new = s_new.index_copy(0, self.inputs, x)
        return s_new

    def _segment(self, s, vals, lam, x, u, record):
        out = []
        st = self._st
        for k in range(len(x) if x is not None else len(u)):
            s = self._step(s, vals, lam, None if x is None else x[k], None if u is None else u[k], st)
            out.append(s[record])
        return s, torch.stack(out)

    def run(self, x=None, u=None, s0=None, record=None, checkpoint_every: int = 0):
        """Run T intervals.

        x: (T, batch, n_inputs) clamp values of the input nodes (in ``graph.inputs`` order), or None.
        u: (T, batch, n_drives) external drive of the drive nodes (in ``graph.drives`` order), or None.
        s0: (batch, n) start state (default 0). record: neuron indices to return (default all; array or tensor).
        checkpoint_every: if > 0, gradient checkpointing per segment of that many intervals.
        Returns (states (T, batch, len(record)), final state (batch, n))."""
        if x is None and u is None:
            raise ValueError("x or u is needed (it sets the number of intervals)")
        T, batch = (x if x is not None else u).shape[:2]
        if x is not None and x.shape[2] != len(self.inputs):
            raise ValueError("x must have one column per input node")
        if u is not None and u.shape[2] != len(self.drives):
            raise ValueError("u must have one column per drive node")
        dev = self.device
        xt = None if x is None else x.to(dev, self.dtype).transpose(1, 2)    # (T, n_in, batch)
        ut = None if u is None else u.to(dev, self.dtype).transpose(1, 2)
        s = self.initial_state(batch) if s0 is None else s0.to(dev, self.dtype).T.contiguous()
        rec = (torch.arange(self.graph.n, device=dev) if record is None
               else record.to(dev) if torch.is_tensor(record)
               else torch.as_tensor(np.asarray(record), dtype=torch.int64, device=dev))
        vals = self.w * self.gains()
        lam = self.leaks()[:, None]
        seg = checkpoint_every if checkpoint_every > 0 else T
        parts = []
        for a in range(0, T, seg):
            b = min(a + seg, T)
            xs = None if xt is None else xt[a:b]
            us = None if ut is None else ut[a:b]
            if checkpoint_every > 0 and torch.is_grad_enabled():
                s, r = checkpoint(self._segment, s, vals, lam, xs, us, rec, use_reentrant=False)
            else:
                s, r = self._segment(s, vals, lam, xs, us, rec)
            parts.append(r)
        return torch.cat(parts).transpose(1, 2), s.T

    def record(self) -> dict:
        """What every run record carries about the engine."""
        return dict(engine="rate_torch", label="TRAINED", graph_sha256=self.graph.sha256(), n_neurons=self.graph.n,
                    n_edges=self.graph.n_edges, n_inputs=int(len(self.graph.inputs)),
                    n_drives=int(len(self.graph.drives)), n_trainable=self.graph.n_trainable,
                    dtype=str(self.dtype).removeprefix("torch."), g_max=G_MAX, lam_min=LAM_MIN, lam_max=LAM_MAX,
                    lam_0=LAM_0, updates_per_interval=UPDATES_PER_INTERVAL, r_in_hz=R_IN,
                    clamp_latency_intervals=0, hop_latency_intervals=1, edge_grad=self.edge_grad)


def reference_numpy(graph: RateGraph, gains, leaks, x=None, u=None, dtype=np.float32):
    """Plain NumPy implementation of the same dynamics (prereg/14 gate 3, forward reference): dense scatter-add per
    edge, no torch. Returns the states (T, batch, n)."""
    gains, leaks = np.asarray(gains, dtype), np.asarray(leaks, dtype)
    T, batch = (x if x is not None else u).shape[:2]
    s = np.zeros((batch, graph.n), dtype)
    vals = (graph.w.astype(dtype) * gains).astype(dtype)
    out = np.empty((T, batch, graph.n), dtype)
    for k in range(T):
        if x is not None:
            s[:, graph.inputs] = x[k]
        h = np.zeros((batch, graph.n), dtype)
        for b in range(batch):
            np.add.at(h[b], graph.post, vals * s[b, graph.pre])
        if u is not None:
            h[:, graph.drives] += u[k]
        s = ((1 - leaks) * s + leaks * np.tanh(h)).astype(dtype)
        if x is not None:
            s[:, graph.inputs] = x[k]
        out[k] = s
    return out


def dense_reference_torch(graph: RateGraph, theta, phi, x=None, u=None):
    """Dense-matrix autograd implementation (prereg/14 gate 3, gradient reference). Use in float64 on small graphs."""
    n = graph.n
    w = torch.zeros(n, n, dtype=theta.dtype)
    post, pre = torch.from_numpy(graph.post), torch.from_numpy(graph.pre)
    w = w.index_put((post, pre), torch.from_numpy(graph.w).to(theta.dtype) * G_MAX * torch.sigmoid(theta))
    lam = LAM_MIN + (LAM_MAX - LAM_MIN) * torch.sigmoid(phi)
    inp, drv = torch.from_numpy(graph.inputs), torch.from_numpy(graph.drives)
    T, batch = (x if x is not None else u).shape[:2]
    s = torch.zeros(batch, n, dtype=theta.dtype)
    out = []
    for k in range(T):
        if x is not None:
            s = s.index_copy(1, inp, x[k])
        h = s @ w.T
        if u is not None:
            h = h.index_add(1, drv, u[k])
        s = (1 - lam) * s + lam * torch.tanh(h)
        if x is not None:
            s = s.index_copy(1, inp, x[k])
        out.append(s)
    return torch.stack(out)
