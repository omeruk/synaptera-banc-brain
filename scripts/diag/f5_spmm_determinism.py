"""Diagnostic (not registered, no metric): are the sparse products of the stage-A rate VNC bit-reproducible?

    env -u PYTHONPATH python scripts/f5_train_guard.py --log runs/f5/A_smoke/guard.log -- \\
        env -u PYTHONPATH python scripts/diag/f5_spmm_determinism.py

On the full stage-A VNC (runs/f5/A/setup.npz) with synthetic inputs (default_rng(0)) and deterministic algorithms on,
each product is called 5 times with the same arguments: W s (CSR, cuSPARSE on CUDA / MKL on the CPU), W^T s (the
state gradient), the sddmm edge gradient, and a segment-sum alternative of W s (torch.segment_reduce over the CSR edge
order). Reported: whether the 5 results are bit-identical, their largest difference, the float32 error against a
float64 product, and the GPU time per product. Output: runs/f5/A_smoke/spmm_determinism.json.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine import rate_torch as RT  # noqa: E402
from train import f5_stageA as A  # noqa: E402

OUT = A.SMOKE_OUT / "spmm_determinism.json"
REPEATS, TIMED = 5, 50


def probe(dev, batches):
    torch.use_deterministic_algorithms(True)
    g = A.Setup.load(A.OUT / "setup.npz").graph()
    net = RT.RateNet(g, edge_grad="sddmm").to(dev)
    st = net._st
    vals = (net.w * 1.3).detach()
    lengths = st.crow[1:] - st.crow[:-1]
    mask = torch.sparse_csr_tensor(st.crow, st.pre, torch.zeros_like(vals), (g.n, g.n), check_invariants=False)
    ops = {"W_s": lambda x, gy: st.matmul(vals, x), "Wt_s": lambda x, gy: st.matmul_t(vals, x),
           "sddmm": lambda x, gy: torch.sparse.sampled_addmm(mask, gy, x.T, beta=0.0, alpha=1.0).values(),
           "W_s_segment": lambda x, gy: torch.segment_reduce(vals[:, None] * x[st.pre], "sum", lengths=lengths, axis=0)}
    rng = np.random.default_rng(0)
    res = {}
    for B in batches:
        x = torch.from_numpy(rng.uniform(-1, 1, (g.n, B)).astype(np.float32)).to(dev)
        gy = torch.from_numpy(rng.normal(size=(g.n, B)).astype(np.float32)).to(dev)
        ref = st.matmul(vals.double(), x.double())
        for name, f in ops.items():
            outs = [f(x, gy) for _ in range(REPEATS)]
            r = dict(identical=bool(all(torch.equal(outs[0], o) for o in outs[1:])),
                     max_diff=float(max((o - outs[0]).abs().max().item() for o in outs)))
            if name in ("W_s", "W_s_segment"):
                r["max_abs_err_vs_float64"] = float((outs[0].double() - ref).abs().max().item())
            if dev == "cuda" and name in ("W_s", "W_s_segment"):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(TIMED):
                    f(x, gy)
                torch.cuda.synchronize()
                r["ms_per_product"] = (time.perf_counter() - t0) / TIMED * 1e3
            res[f"{name}_b{B}"] = r
    return dict(n_neurons=g.n, n_edges=g.n_edges, graph_sha256=g.sha256(), repeats=REPEATS, products=res)


def main():
    out = dict(note="diagnostic only: synthetic inputs, no metric", torch=torch.__version__,
               cpu=probe("cpu", (1, 16)))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(A.GPU_FRACTION)
        out["cuda"] = probe("cuda", (1, 16, 80))
        out["gpu"] = torch.cuda.get_device_name(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
