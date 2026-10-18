"""F5 stage A training (prereg/14 sections 2-9; prereg/17: machine-2 records; prereg/18: sddmm, no checkpointing in A).

**Data and the sealed test split.** The 2,400 machine-2 records of ``runs/f5/stageA`` (prereg/17). Every record is
checked against ``manifest.sha256`` before use. A record is opened only through :func:`open_record`, which takes the
data seed from the file name and refuses a test seed unless the caller is inside :func:`final_evaluation`; only
``scripts/f5_train.py final-eval`` enters it. :func:`load_trials` refuses the test split outside it as well. The splits
are those of ``configs/f5_splits.json``; its SHA-256 is recorded in every run.

**Trials.** Train and validation: per data seed rot_left / rot_right at 60 and 120 deg/s, static and grey (6 trials;
960 training and 240 validation trials). Test: the same 6 (primary metrics) plus rot_left / rot_right at 90 deg/s
(generalisation, primary) and 30 deg/s (secondary).

**Input.** Per trial, the bridge (prereg/11 `ensemble`, window 5 intervals, latency 1, no Poisson re-synthesis) turns
the recorded FlyWire DN spike counts into unit rates; BANC DN state = unit rate / R_IN (prereg/14 section 2).
``coupling.dn_bridge.DNBridge.unit_rates`` is the transform (the same code as the F3b runs).

**Arms** (prereg/14 section 5). `true`, `shuffled`, `derange`, `bridge_off` and `untrained` are a :class:`GraphArm`
(the rate VNC subgraph, prereg/08 with prereg/01 signs): the clamp of the BANC DNs is the DNg02 baseline (1, `HAND`)
with every bridge target set to its unit value. `shuffled` uses the degree-preserving shuffle of training seed t
(`default_rng(20261501 + t)`), `derange` the bridge of the prereg/11 derangement `random_(t+1)`, `bridge_off` no bridge
(every bridged DN 0, DNg02 kept). `mlp` and `linear` are a :class:`FilteredArm`: the 224-unit input of the `true`
bridge through exponential filters of 10, 25, 50 and 100 ms, then two ReLU layers of width h (mlp) or nothing (linear),
then one linear map to the 16 readout MN states. Every arm ends in the same fixed readout:

    dA = G1 * R_OUT * sum_m sigma_m (s_{m,L} - s_{m,R})      (Lindsay `lateral` set, G1 = 0.395 deg/Hz)

**Teacher** (section 4). dA*(k) = clip(K_P v, -A_MAX, +A_MAX), K_P = A_MAX / V_MAX, v = +speed for rot_right and
-speed for rot_left on the stimulus intervals k >= N_GREY; 0 in the grey lead-in, under static and under grey. The
open-loop target of interval k is that of the stimulus shown in interval k.

**Causal timing** (rule 4). The clamp of interval k comes from DN counts of intervals k-5 .. k-1 (bridge latency 1);
the VNC state of interval k gives dA_k (one synapse hop per interval, the clamp acts in its own interval); in a closed
loop the body would receive dA_k at k + 1 (command queue). Every latency is in :data:`LATENCIES` and in each record.

**Training** (section 6). Adam (0.9 / 0.999, eps 1e-8), no weight decay, gradient clipping at global norm 1.0, loss =
mean over every interval of ((dA - dA*) / A_MAX)^2, BPTT over the whole 300-interval trial, batch 16, 2,000 steps.
Data order: one permutation of the training trials per epoch from `default_rng(1000 x 1 + t)`. Validation every 50
steps on the 240 validation trials (loss, turn-sign accuracy, R2); checkpoints every 25 steps (train/checkpoint.py,
atomic, last 3 kept, plus ``best.pt`` = lowest validation loss among the evaluations at steps 50, 100, ...).
Deterministic algorithms are on. On the CPU a resumed run is bit-identical to an uninterrupted one; on the GPU the
cuSPARSE CSR products are not bit-reproducible on the full VNC, so GPU runs agree only to float32 rounding
(:data:`REPRODUCIBILITY`, recorded in every run). prereg/19: GPU training is accepted at that level; the `true` arm
at training seed 0 and the selected rate is trained a second time (``replicate``) to measure it; the final test
evaluation runs on the CPU from ``best.pt`` and is exactly reproducible.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine import rate_torch as RT
from train import checkpoint as CK

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # before the first cuBLAS call (deterministic GPU)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "configs" / "f5_splits.json"
RECORDS = ROOT / "runs" / "f5" / "stageA"            # prereg/17 records (brain/*.npz, manifest.sha256)
OUT = ROOT / "runs" / "f5" / "A"                     # stage-A training runs
SMOKE_OUT = ROOT / "runs" / "f5" / "A_smoke"         # smoke runs: never read by selection, test or report

# ── registered constants (configs/assumptions.json, f5.*) ─────────────────────────────────────────────────────────
STAGE, STAGE_CODE = "A", 1
A_MAX = 12.5                                         # deg; v1 turn range / 2 (docs/08 section 4.2)
V_MAX = 120.0                                        # deg/s; the fastest drum
K_P = A_MAX / V_MAX                                  # deg per deg/s
G1 = 0.395                                           # deg/Hz (docs/08 section 10.4)
SCALE = G1 * RT.R_IN                                 # deg per unit of sum sigma (s_L - s_R)
INTERVAL_MS = 5.0
N_GREY, T_TRIAL = 100, 300                           # grey lead-in, trial length (intervals)
WINDOW = (120, 300)                                  # analysis window: the stimulus after its first 100 ms
TRAIN_CONDITIONS = (("rot_left", 60), ("rot_right", 60), ("rot_left", 120), ("rot_right", 120), ("static", None),
                    ("grey", None))
HELD_OUT = {"primary": 90, "secondary": 30}
BATCH, STEPS = 16, 2000
LR_GRID = (1e-3, 3e-3, 1e-2)
BETAS, ADAM_EPS, CLIP_NORM = (0.9, 0.999), 1e-8, 1.0
EVAL_EVERY, CKPT_EVERY = 50, 25
FILTER_MS = (10.0, 25.0, 50.0, 100.0)
SHUFFLE_SEED0 = 20261501
SET_ACCURACY = 0.90
EDGE_GRAD, CHECKPOINT_EVERY = "sddmm", 0              # prereg/18 section 1, stage A
GPU_FRACTION = 0.7                                   # per-process cap on the GPU memory (machine-2 policy)
TEST_THREADS = 8                                     # CPU threads of the final test evaluation (prereg/19 section 1)
ARMS = ("true", "shuffled", "derange", "mlp", "linear", "bridge_off", "untrained")
GRAPH_ARMS = ("true", "shuffled", "derange", "bridge_off", "untrained")
TRAIN_SEEDS = {"true": range(10), "shuffled": range(10), "derange": range(10), "mlp": range(10), "linear": range(10),
               "bridge_off": range(5), "untrained": range(1)}
LATENCIES = dict(bridge_latency_intervals=1, bridge_window_intervals=5, clamp_latency_intervals=0,
                 hop_latency_intervals=1, command_latency_intervals=1)
LABELS = dict(dA="TRAINED", dng02_baseline="HAND", teacher="HAND", dn_input="BRAIN")
REPRODUCIBILITY = ("deterministic algorithms on; every RNG, the optimiser state and the data position are in each "
                   "checkpoint. CPU: a resumed run is bit-identical to an uninterrupted one. CUDA: the CSR products of "
                   "torch / cuSPARSE (W s and W^T s) are not bit-reproducible on the full VNC (largest differences "
                   "between identical calls about 1e-7 and 3e-6, scripts/diag/f5_spmm_determinism.py), so two GPU runs "
                   "and a resumed GPU run agree only to float32 rounding; sddmm is bit-reproducible")


# ── splits and the sealed test set ──────────────────────────────────────────────────────────────────────────────────
class SealedSplitError(RuntimeError):
    """A test seed reached a training or selection path."""


def splits(path=SPLIT_FILE) -> dict:
    """train / val / test data seeds; a seed in more than one split is refused (a test seed anywhere else is a breach
    of the sealed split)."""
    d = json.loads(Path(path).read_text())
    s = {k: tuple(d[k]) for k in ("train", "val", "test")}
    if set(s["test"]) & (set(s["train"]) | set(s["val"])):
        raise SealedSplitError(f"{path}: a test seed is also listed as a training or validation seed")
    if set(s["train"]) & set(s["val"]):
        raise ValueError(f"{path}: a seed is both a training and a validation seed")
    return s


def split_sha256(path=SPLIT_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_FINAL_EVAL = False


@contextlib.contextmanager
def final_evaluation():
    """The only context in which test records may be opened (``scripts/f5_train.py final-eval``)."""
    global _FINAL_EVAL
    prev, _FINAL_EVAL = _FINAL_EVAL, True
    try:
        yield
    finally:
        _FINAL_EVAL = prev


def in_final_evaluation() -> bool:
    return _FINAL_EVAL


_SEED_RE = re.compile(r"_s(\d{3})\.npz$")


def seed_of_file(path) -> int:
    m = _SEED_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"no data seed in the record name {Path(path).name}")
    return int(m.group(1))


def check_seed(seed: int, split_path=SPLIT_FILE) -> None:
    if int(seed) in splits(split_path)["test"] and not _FINAL_EVAL:
        raise SealedSplitError(f"data seed {seed} is a sealed test seed; only the final evaluation may read it")


def open_record(path, split_path=SPLIT_FILE):
    """np.load of one record, refused for a test seed outside :func:`final_evaluation`. The seed comes from the file
    name, so no caller can mislabel it."""
    check_seed(seed_of_file(path), split_path)
    return np.load(path, allow_pickle=False)


def record_name(stim, speed, seed) -> str:
    """As scripts/f5_stageA_records.py:record_name."""
    return f"{stim}_v{speed:03d}_s{seed:03d}.npz" if stim.startswith("rot") else f"{stim}_s{seed:03d}.npz"


def trial_specs(split, conditions=TRAIN_CONDITIONS, split_path=SPLIT_FILE) -> list[dict]:
    if split == "test" and not _FINAL_EVAL:
        raise SealedSplitError("the test split is read only by the final evaluation")
    return [dict(stimulus=st, speed=v, seed=s, file=f"brain/{record_name(st, v or 0, s)}")
            for s in splits(split_path)[split] for st, v in conditions]


def read_manifest(path) -> dict:
    out = {}
    for ln in Path(path).read_text().splitlines():
        h, f = ln.split(None, 1)
        out[f.strip()] = h
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── teacher ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def target_value(stim, speed) -> float:
    """dA* on the stimulus intervals (deg); rot_right (pattern moves to the fly's right) is positive."""
    if stim in ("static", "grey"):
        return 0.0
    v = {"rot_right": 1.0, "rot_left": -1.0}[stim] * float(speed)
    return float(np.clip(K_P * v, -A_MAX, A_MAX))


def target_series(stim, speed, T=T_TRIAL) -> np.ndarray:
    z = np.zeros(T, np.float32)
    z[N_GREY:] = target_value(stim, speed)
    return z


# ── setup (graph, readout, baseline, DN tables) ─────────────────────────────────────────────────────────────────────
@dataclass
class Setup:
    """Everything an arm needs besides the data. ``banc_*``: the BANC DN table (engine index = VNC node index);
    ``fw_*``: the FlyWire DN table, one row per column of a record's ``dn_counts``."""
    n: int
    pre: np.ndarray
    post: np.ndarray
    n_syn: np.ndarray
    sign: np.ndarray
    inputs: np.ndarray
    readout: np.ndarray
    sigma: np.ndarray
    base: np.ndarray
    banc_cell_type: np.ndarray
    banc_side: np.ndarray
    banc_index: np.ndarray
    fw_cell_type: np.ndarray
    fw_side: np.ndarray
    fw_index: np.ndarray
    families: tuple
    meta: dict = field(default_factory=dict)

    ARRAYS = ("pre", "post", "n_syn", "sign", "inputs", "readout", "sigma", "base", "banc_cell_type", "banc_side",
              "banc_index", "fw_cell_type", "fw_side", "fw_index")

    def graph(self, post=None) -> RT.RateGraph:
        return RT.RateGraph.from_arrays(self.n, self.pre, self.post if post is None else post, self.n_syn, self.sign,
                                        self.inputs)

    def content_sha256(self) -> str:
        h = hashlib.sha256(np.int64(self.n).tobytes())
        for k in self.ARRAYS:
            a = np.ascontiguousarray(getattr(self, k))
            h.update(k.encode() + str(a.dtype).encode() + a.tobytes())
        h.update(json.dumps(list(self.families)).encode())
        return h.hexdigest()

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, n=self.n, families=np.array(self.families), meta=json.dumps(self.meta),
                 **{k: getattr(self, k) for k in self.ARRAYS})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> "Setup":
        z = np.load(path, allow_pickle=False)
        return cls(n=int(z["n"]), families=tuple(str(f) for f in z["families"]), meta=json.loads(str(z["meta"])),
                   **{k: z[k] for k in cls.ARRAYS})

    def banc_table(self):
        import pandas as pd
        return pd.DataFrame(dict(cell_type=self.banc_cell_type.astype(str), side=self.banc_side.astype(str),
                                 index=self.banc_index.astype(np.int64)))

    def fw_table(self):
        import pandas as pd
        return pd.DataFrame(dict(cell_type=self.fw_cell_type.astype(str), side=self.fw_side.astype(str)))


def bridge_config(families, source=None):
    from coupling.dn_bridge import BridgeConfig
    return BridgeConfig(families=tuple(families), interval_ms=INTERVAL_MS,
                        latency_intervals=LATENCIES["bridge_latency_intervals"],
                        window_intervals=LATENCIES["bridge_window_intervals"], source=dict(source or {}))


def derangement_of(t: int, families) -> dict:
    """Target -> source family map of `derange` at training seed t: prereg/11 `random_(t+1)`."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import f3b3_common as C
    if tuple(families) == tuple(C.ENSEMBLE):
        return dict(C.permutations()[f"random_{t + 1}"])
    return C.derangement(list(families), C.PERM_SEEDS[t])          # synthetic families (unit tests): same generator


def make_bridge(setup: Setup, arm: str, t: int):
    """The DNBridge an arm reads its input through (None for `bridge_off`). `mlp` and `linear` use the `true` one."""
    from coupling.dn_bridge import DNBridge
    if arm == "bridge_off":
        return None
    src = derangement_of(t, setup.families) if arm == "derange" else None
    return DNBridge(setup.fw_table(), setup.banc_table(), bridge_config(setup.families, src))


def bridge_key(bridge) -> str:
    if bridge is None:
        return "none"
    return hashlib.sha256(json.dumps(bridge.record(), sort_keys=True, default=str).encode()).hexdigest()


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Trials:
    units: np.ndarray            # (T, N, U) float32: bridge unit rate / R_IN of the interval each value is applied in
    target: np.ndarray           # (T, N) float32: dA*
    stimulus: list
    speed: np.ndarray            # deg/s (0 for static and grey)
    seed: np.ndarray
    files: list
    sha256: list

    def __len__(self):
        return len(self.files)

    def rotation(self) -> np.ndarray:
        return np.array([s.startswith("rot") for s in self.stimulus])


def load_trials(split, setup: Setup, bridge, conditions=TRAIN_CONDITIONS, records=RECORDS, split_path=SPLIT_FILE,
                manifest=None) -> Trials:
    """The trials of one split through one bridge. Each record must match the manifest and come from the split."""
    if split != "test" and any(v in HELD_OUT.values() for _, v in conditions):
        raise SealedSplitError("the held-out drum speeds (90, 30 deg/s) are read only by the final evaluation")
    specs = trial_specs(split, conditions, split_path)
    allowed = set(splits(split_path)[split])
    records = Path(records)
    man = read_manifest(manifest or records / "manifest.sha256")
    U = 0 if bridge is None else len(bridge.units)
    units = np.zeros((T_TRIAL, len(specs), U), np.float32)
    target = np.zeros((T_TRIAL, len(specs)), np.float32)
    shas = []
    for b, sp in enumerate(specs):
        if sp["seed"] not in allowed:
            raise SealedSplitError(f"{sp['file']} is not in the {split} split")
        path = records / sp["file"]
        h = sha256_file(path)
        if man.get(sp["file"]) != h:
            raise ValueError(f"{sp['file']}: SHA-256 differs from the manifest")
        z = open_record(path, split_path)
        if not np.array_equal(z["dn_index"], setup.fw_index):
            raise ValueError(f"{sp['file']}: DN columns differ from the setup")
        if str(z["stimulus"]) != sp["stimulus"] or int(z["seed"]) != sp["seed"]:
            raise ValueError(f"{sp['file']}: stimulus or seed differs from its name")
        if bridge is not None:
            units[:, b] = bridge.unit_rates(z["dn_counts"]) / RT.R_IN
        target[:, b] = target_series(sp["stimulus"], sp["speed"])
        shas.append(h)
    return Trials(units, target, [s["stimulus"] for s in specs], np.array([s["speed"] or 0 for s in specs]),
                  np.array([s["seed"] for s in specs]), [s["file"] for s in specs], shas)


# ── arms ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class GraphArm(nn.Module):
    """Rate VNC with the bridge clamp and the fixed readout. ``bridge`` None: every bridged DN stays 0 (`bridge_off`)."""

    def __init__(self, setup: Setup, graph: RT.RateGraph, bridge, edge_grad=EDGE_GRAD, checkpoint_every=CHECKPOINT_EVERY):
        super().__init__()
        self.net = RT.RateNet(graph, edge_grad=edge_grad)
        self.checkpoint_every = int(checkpoint_every)
        col = {int(i): k for k, i in enumerate(graph.inputs)}
        if bridge is not None and len(bridge.units):
            tgt = np.array([col[int(i)] for i in bridge.target_index], np.int64)
            uot = bridge.unit_of_target().astype(np.int64)
            if np.any(np.asarray(setup.base)[tgt] != 0):
                raise ValueError("a bridge target is also a baseline DN")
        else:
            tgt, uot = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.n_units = 0 if bridge is None else len(bridge.units)
        self.register_buffer("base", torch.as_tensor(np.asarray(setup.base, np.float32)), persistent=False)
        self.register_buffer("tgt_col", torch.from_numpy(tgt), persistent=False)
        self.register_buffer("unit_of_target", torch.from_numpy(uot), persistent=False)
        self.register_buffer("readout", torch.as_tensor(np.asarray(setup.readout, np.int64)), persistent=False)
        self.register_buffer("sigma", torch.as_tensor(np.asarray(setup.sigma, np.float32)), persistent=False)

    def clamp(self, units):
        T, B = units.shape[:2]
        x = self.base.expand(T, B, -1).clone()
        if len(self.tgt_col):
            x[:, :, self.tgt_col] = units[:, :, self.unit_of_target]
        return x

    def forward(self, units):
        st, _ = self.net.run(self.clamp(units), record=self.readout, checkpoint_every=self.checkpoint_every)
        return SCALE * (st * self.sigma).sum(2)

    def n_trainable(self) -> int:
        return self.net.graph.n_trainable

    def record(self) -> dict:
        return dict(kind="graph", engine=self.net.record(), checkpoint_every=self.checkpoint_every,
                    n_units=self.n_units, n_targets=int(len(self.tgt_col)))


def mlp_param_count(n_features, h, n_out=16) -> int:
    return n_features * h + h + h * h + h + h * n_out + n_out


def mlp_width(n_features, n_target_params, n_out=16) -> int:
    """Smallest h with a parameter count >= n_target_params (prereg/14 section 5)."""
    b = n_features + n_out + 2
    h = max(1, int(math.floor((-b + math.sqrt(b * b - 4 * (n_out - n_target_params))) / 2)) - 2)
    while mlp_param_count(n_features, h, n_out) < n_target_params:
        h += 1
    return h


class FilteredArm(nn.Module):
    """`mlp` (``hidden`` = h) or `linear` (``hidden`` = None) on the exponentially filtered unit input."""

    def __init__(self, setup: Setup, n_units: int, hidden=None):
        super().__init__()
        self.n_units, self.hidden = int(n_units), hidden
        nf = n_units * len(FILTER_MS)
        if hidden is None:
            self.net = nn.Linear(nf, 16)
        else:
            self.net = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 16))
        self.register_buffer("decay", torch.tensor([math.exp(-INTERVAL_MS / tau) for tau in FILTER_MS]),
                             persistent=False)
        self.register_buffer("sigma", torch.as_tensor(np.asarray(setup.sigma, np.float32)), persistent=False)

    def features(self, units):
        """y_k = a y_{k-1} + (1 - a) u_k per filter (y_{-1} = 0); (T, B, U) -> (T, B, 4U), filter-major."""
        a = self.decay[:, None, None]
        y = torch.zeros((len(FILTER_MS),) + units.shape[1:], dtype=units.dtype, device=units.device)
        out = []
        for k in range(units.shape[0]):
            y = a * y + (1 - a) * units[k]
            out.append(y.permute(1, 0, 2).reshape(units.shape[1], -1))
        return torch.stack(out)

    def forward(self, units):
        return SCALE * (self.net(self.features(units)) * self.sigma).sum(2)

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def record(self) -> dict:
        return dict(kind="mlp" if self.hidden else "linear", hidden=self.hidden, n_features=self.n_units * len(FILTER_MS),
                    filters_ms=list(FILTER_MS), n_trainable=self.n_trainable(), output="linear (no output nonlinearity)")


def shuffled_post(setup: Setup, t: int, cache_dir=None):
    """Post array of the `shuffled` graph at training seed t (cached; the check of gate 4 is stored with it)."""
    from train.graphs import degree_preserving_shuffle, shuffle_check
    path = None if cache_dir is None else Path(cache_dir) / f"shuffled_t{t}.npz"
    seed = SHUFFLE_SEED0 + t
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=False)
        if int(z["seed"]) == seed and str(z["graph_sha256"]) == setup.graph().sha256():
            return z["post"], json.loads(str(z["info"]))
    post, info = degree_preserving_shuffle(setup.pre, setup.post, setup.n, seed)
    info["check"] = shuffle_check(setup.pre, setup.post, post, setup.n, setup.sign)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, post=post, seed=seed, graph_sha256=setup.graph().sha256(), info=json.dumps(info))
        os.replace(tmp, path)
    return post, info


def make_model(arm, t, setup: Setup, bridge, graph_cache=None, true_trainable=None):
    """(model, info). ``true_trainable``: the `true` arm's trainable count (sets the MLP width)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    info = dict(arm=arm, training_seed=int(t))
    if arm in GRAPH_ARMS:
        post = None
        if arm == "shuffled":
            post, sh = shuffled_post(setup, t, graph_cache)
            if not sh["check"]["passed"]:
                raise RuntimeError("shuffle check failed (gate 4)")
            info["shuffle"] = sh
        model = GraphArm(setup, setup.graph(post), bridge)
    else:
        torch.manual_seed(t)                          # the training seed sets the MLP / linear initialisation
        n_true = true_trainable if true_trainable is not None else setup.graph().n_trainable
        h = mlp_width(len(bridge.units) * len(FILTER_MS), n_true) if arm == "mlp" else None
        model = FilteredArm(setup, len(bridge.units), h)
        info.update(true_trainable=int(n_true))
    info.update(model.record())
    return model, info


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def r2(pred, target) -> float:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    sst = float(((target - target.mean()) ** 2).sum())
    return float(1.0 - ((pred - target) ** 2).sum() / sst) if sst > 0 else float("nan")


def metrics(dA, data: Trials) -> dict:
    """dA: (T, N) of every trial. loss = the training loss over all trials and intervals; sign accuracy over the
    rotation trials, R2 over all trials, both on the window mean (prereg/14 section 8.1)."""
    dA = np.asarray(dA, np.float64)
    tgt = data.target.astype(np.float64)
    loss = float((((dA - tgt) / A_MAX) ** 2).mean())
    wm, tm = dA[WINDOW[0]:WINDOW[1]].mean(0), tgt[WINDOW[0]:WINDOW[1]].mean(0)
    rot = data.rotation()
    acc = float((np.sign(wm[rot]) == np.sign(tm[rot])).mean()) if rot.any() else float("nan")
    return dict(loss=loss, sign_accuracy=acc, r2=r2(wm, tm), n_trials=int(len(data)), n_rotation=int(rot.sum()))


@torch.no_grad()
def predict(model, units, device, chunk=80) -> np.ndarray:
    out = []
    for a in range(0, units.shape[1], chunk):
        out.append(model(torch.from_numpy(units[:, a:a + chunk]).to(device)).double().cpu().numpy())
    return np.concatenate(out, axis=1)


def samples_to_set_accuracy(evals, batch=BATCH, budget=STEPS):
    """Training trials presented until the validation sign accuracy first reaches 0.90 (None and censored if never)."""
    for e in evals:
        if e["step"] > 0 and e["sign_accuracy"] >= SET_ACCURACY:
            return dict(samples=int(e["step"] * batch), censored=False)
    return dict(samples=int(budget * batch), censored=True)


# ── training ────────────────────────────────────────────────────────────────────────────────────────────────────────
def data_order(t, n_train, steps=STEPS, batch=BATCH) -> np.ndarray:
    """Epoch-wise shuffle (prereg/14 section 6): one permutation per epoch from default_rng(1000 x stage_code + t)."""
    rng = np.random.default_rng(1000 * STAGE_CODE + t)
    n_epochs = -(-steps * batch // n_train) + 1
    return np.concatenate([rng.permutation(n_train) for _ in range(n_epochs)])


def deterministic(device):
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)


class JsonLog:
    """JSON lines (one object per line). On resume the lines after the checkpoint step are dropped (atomic rewrite)."""

    def __init__(self, path):
        self.path = Path(path)

    def write(self, **rec):
        rec.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def truncate_after(self, step: int):
        keep = [r for r in self.read() if r.get("step", 0) <= step]
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
        os.replace(tmp, self.path)


@dataclass
class RunConfig:
    arm: str
    seed: int
    lr: float
    run_dir: Path
    steps: int = STEPS
    batch: int = BATCH
    eval_every: int = EVAL_EVERY
    ckpt_every: int = CKPT_EVERY
    device: str = "cpu"
    smoke: bool = False
    replicate: bool = False                          # prereg/19: second run of true / seed 0 / selected rate

    def record(self) -> dict:
        d = dict(self.__dict__)
        d["run_dir"] = str(self.run_dir)
        return d


def run_record(cfg: RunConfig, setup: Setup, model_info: dict, bridge, train: Trials, val: Trials | None,
               split_path=SPLIT_FILE) -> dict:
    """What config.json of a run holds (must be identical when a run is resumed)."""
    return dict(stage=STAGE, smoke=cfg.smoke, config=cfg.record(), model=model_info,
                optimiser=dict(name="Adam", betas=list(BETAS), eps=ADAM_EPS, weight_decay=0.0, clip_norm=CLIP_NORM),
                loss="mean over every interval of ((dA - dA*) / A_MAX)^2; BPTT over the 300-interval trial",
                teacher=dict(a_max_deg=A_MAX, v_max_deg_s=V_MAX, k_p=K_P, n_grey=N_GREY,
                             label=LABELS["teacher"]),
                readout=dict(g1_deg_per_hz=G1, r_out_hz=RT.R_IN, sigma=np.asarray(setup.sigma).tolist(),
                             readout=np.asarray(setup.readout).tolist(), names=setup.meta.get("readout_names")),
                labels=LABELS, latencies=LATENCIES, edge_grad=EDGE_GRAD, checkpoint_every=CHECKPOINT_EVERY,
                window=list(WINDOW), split_sha256=split_sha256(split_path), setup_sha256=setup.content_sha256(),
                graph_sha256=setup.graph().sha256(), bridge_sha256=bridge_key(bridge),
                bridge=None if bridge is None else dict(n_units=len(bridge.units),
                                                         n_targets=int(len(bridge.target_index)),
                                                         config=bridge.config.record()),
                train_records_sha256=hashlib.sha256("".join(train.sha256).encode()).hexdigest(),
                val_records_sha256=None if val is None else hashlib.sha256("".join(val.sha256).encode()).hexdigest(),
                n_train=len(train), n_val=None if val is None else len(val),
                data_order_seed=1000 * STAGE_CODE + cfg.seed, reproducibility=REPRODUCIBILITY)


def gpu_stats(device) -> dict:
    if not str(device).startswith("cuda"):
        return {}
    return dict(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def peak_rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def train_run(cfg: RunConfig, model, record: dict, train: Trials, val: Trials | None, log_fn=print,
              stop_after=None, on_eval=None) -> dict:
    """Train (or resume) one run. ``val`` None (smoke only): evaluations use the first 240 training trials and are
    marked so. ``stop_after``: stop after that many steps of this call (tests: an interruption). ``on_eval``: called
    after each evaluation and its checkpoint (the runner refreshes the learning-curve PNG). Returns DONE."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != json.loads(json.dumps(record)):
            raise SystemExit(f"{cfg_path}: the stored configuration differs from this one; use a new run directory")
    else:
        tmp = cfg_path.with_name("config.json.tmp")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, cfg_path)
    if (run_dir / "DONE").exists():
        return json.loads((run_dir / "DONE").read_text())
    dev = torch.device(cfg.device)
    deterministic(dev)
    model = model.to(dev)
    log = JsonLog(run_dir / "log.jsonl")
    params = list(model.parameters())
    trainable = cfg.arm != "untrained"
    opt = torch.optim.Adam(params, lr=cfg.lr if trainable else 0.0, betas=BETAS, eps=ADAM_EPS, weight_decay=0.0)
    ev_data = val if val is not None else _subset(train, 240)
    ev_name = "validation" if val is not None else "smoke_eval_on_training_trials"
    order = data_order(cfg.seed, len(train), cfg.steps, cfg.batch)
    start, best_step, best_loss = 0, None, math.inf
    last = CK.latest(run_dir)
    if last is not None:
        ck = CK.load(last, model, opt)
        start = ck["step"]
        best_step, best_loss = ck["extra"]["best_step"], ck["extra"]["best_val_loss"]
        if best_step == start and (not (run_dir / "best.pt").exists()
                                   or torch.load(run_dir / "best.pt", weights_only=False)["step"] != start):
            CK._atomic_save(torch.load(last, weights_only=False), run_dir / "best.pt")
        log.truncate_after(start)
        log.write(event="resume", step=start, from_checkpoint=last.name)
        log_fn(f"resumed at step {start}")
    else:
        log.truncate_after(-1)
        log.write(event="start", step=0, n_trainable=model.n_trainable(), device=str(dev), smoke=cfg.smoke)
        e = _evaluate(model, ev_data, dev)
        log.write(event="eval", step=0, trials_seen=0, set=ev_name, selectable=False, **e)
    targets = torch.from_numpy(train.target)
    units = torch.from_numpy(train.units)
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    steps_here = 0
    for step in range(start, cfg.steps if trainable else start):
        t0 = time.perf_counter()
        idx = torch.from_numpy(order[step * cfg.batch:(step + 1) * cfg.batch])
        u, tg = units[:, idx].to(dev), targets[:, idx].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = ((model(u) - tg) / A_MAX).square().mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
        opt.step()
        lv = float(loss.detach())
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        n = step + 1
        log.write(event="step", step=n, trials_seen=n * cfg.batch, loss=lv, grad_norm=float(gn),
                  step_s=time.perf_counter() - t0)
        is_eval = n % cfg.eval_every == 0
        if is_eval:
            e = _evaluate(model, ev_data, dev)
            if e["loss"] < best_loss:
                best_step, best_loss = n, e["loss"]
            log.write(event="eval", step=n, trials_seen=n * cfg.batch, set=ev_name, selectable=True,
                      best_step=best_step, **e, **gpu_stats(dev))
            log_fn(f"step {n}: evaluated ({ev_name}, {e['eval_s']:.1f} s)" if cfg.smoke else
                   f"step {n}: {ev_name} loss {e['loss']:.4f} sign {e['sign_accuracy']:.3f} r2 {e['r2']:.3f}")
        if is_eval or n % cfg.ckpt_every == 0:
            extra = dict(best_step=best_step, best_val_loss=best_loss, data_order_position=n * cfg.batch,
                         config_sha256=sha256_file(cfg_path))
            CK.save(run_dir, n, model, opt, extra=extra, best=is_eval and best_step == n)
            log.write(event="checkpoint", step=n, best=bool(is_eval and best_step == n))
        if is_eval and on_eval is not None:
            on_eval()
        steps_here += 1
        if stop_after is not None and steps_here >= stop_after:
            return dict(interrupted=True, step=n)
    evals = [r for r in log.read() if r.get("event") == "eval"]
    steps_log = [r for r in log.read() if r.get("event") == "step"]
    if not trainable:                                   # untrained: the initialisation is the selected checkpoint
        CK.save(run_dir, 0, model, opt, extra=dict(best_step=0, best_val_loss=evals[0]["loss"]), best=True)
        best_step = 0
    sel = [e for e in evals if e["selectable"]]
    done = dict(stage=STAGE, arm=cfg.arm, training_seed=cfg.seed, lr=cfg.lr if trainable else None, smoke=cfg.smoke,
                replicate=cfg.replicate,
                steps=cfg.steps if trainable else 0, eval_set=ev_name, selected_step=best_step,
                selected_val_loss=(min(e["loss"] for e in sel) if sel else evals[0]["loss"]),
                final_val_loss=evals[-1]["loss"], final_step=evals[-1]["step"],
                samples_to_set_accuracy=samples_to_set_accuracy(evals, cfg.batch, cfg.steps),
                median_step_s=(float(np.median([r["step_s"] for r in steps_log])) if steps_log else None),
                wall_s_this_session=time.time() - t_run, peak_rss_gib=peak_rss_gib(), **gpu_stats(dev),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = run_dir / "DONE.tmp"
    tmp.write_text(json.dumps(done, indent=1))
    os.replace(tmp, run_dir / "DONE")
    return done


def _subset(d: Trials, n) -> Trials:
    n = min(n, len(d))
    return Trials(d.units[:, :n], d.target[:, :n], d.stimulus[:n], d.speed[:n], d.seed[:n], d.files[:n], d.sha256[:n])


def _evaluate(model, data: Trials, dev) -> dict:
    t0 = time.perf_counter()
    m = metrics(predict(model, data.units, dev), data)
    m["eval_s"] = time.perf_counter() - t0
    return m


# ── learning-rate selection ─────────────────────────────────────────────────────────────────────────────────────────
def run_dir_of(arm, t, lr, out=OUT, replicate=False) -> Path:
    """runs/f5/A/<arm>/t<seed>_lr<lr> (`untrained`: t0; the prereg/19 replicate: t0_lr<lr>_replicate). Smoke runs live
    under runs/f5/A_smoke instead."""
    if replicate:
        return Path(out) / arm / f"t{t}_lr{lr:g}_replicate"
    return Path(out) / arm / (f"t{t}" if arm == "untrained" else f"t{t}_lr{lr:g}")


def select_lr(arm, out=OUT) -> dict:
    """prereg/14 section 6: per (stage, arm), the rate with the lowest final validation loss at training seed 0 (the
    evaluation at the last step); ties go to the smaller rate. Smoke runs are never read."""
    rows = []
    for lr in LR_GRID:
        p = run_dir_of(arm, 0, lr, out) / "DONE"
        if not p.exists():
            raise SystemExit(f"{p} missing: the seed-0 run at lr {lr:g} has not finished")
        d = json.loads(p.read_text())
        if d.get("smoke") or d["eval_set"] != "validation" or d["final_step"] != STEPS:
            raise SystemExit(f"{p}: not a complete validation run")
        rows.append(dict(lr=lr, final_val_loss=d["final_val_loss"], selected_val_loss=d["selected_val_loss"]))
    best = min(rows, key=lambda r: (r["final_val_loss"], r["lr"]))
    return dict(stage=STAGE, arm=arm, rule="lowest final validation loss at training seed 0 (prereg/14 section 6)",
                grid=rows, lr=best["lr"])


# ── final evaluation (the only reader of the test seeds) ────────────────────────────────────────────────────────────
TEST_GROUPS = {"training_speeds": TRAIN_CONDITIONS,
               "held_out_primary_90": (("rot_left", HELD_OUT["primary"]), ("rot_right", HELD_OUT["primary"])),
               "held_out_secondary_30": (("rot_left", HELD_OUT["secondary"]), ("rot_right", HELD_OUT["secondary"]))}


def final_evaluate(run_dir, setup: Setup, bridge, model, records=RECORDS, split_path=SPLIT_FILE) -> dict:
    """Test metrics of the selected checkpoint (best.pt) of a finished, non-smoke, non-replicate run, on the CPU with
    deterministic algorithms and TEST_THREADS threads (prereg/19 section 1: exactly reproducible from best.pt). Runs
    once: test.lock is created exclusively first, so a second call is refused even after a crash. Writes test.json."""
    run_dir = Path(run_dir)
    done = run_dir / "DONE"
    if not done.exists():
        raise SystemExit(f"{run_dir}: not finished")
    d = json.loads(done.read_text())
    if d.get("smoke") or d["eval_set"] != "validation":
        raise SystemExit("smoke runs are never evaluated on the test set")
    if d.get("replicate"):
        raise SystemExit("the prereg/19 replicate run is never evaluated on the test set")
    try:
        fd = os.open(run_dir / "test.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{run_dir}: the test evaluation has already run (test.lock); it runs once") from None
    os.write(fd, time.strftime("%Y-%m-%dT%H:%M:%S\n").encode())
    os.close(fd)
    ck = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")
    device = "cpu"
    threads = torch.get_num_threads()
    torch.set_num_threads(TEST_THREADS)
    model.load_state_dict(ck["model"])
    deterministic(device)
    model = model.to(device)
    res = dict(stage=STAGE, arm=d["arm"], training_seed=d["training_seed"], lr=d["lr"], selected_step=ck["step"],
               split_sha256=split_sha256(split_path), device=device, threads=torch.get_num_threads(),
               best_sha256=sha256_file(run_dir / "best.pt"), groups={})
    with final_evaluation():
        for name, conds in TEST_GROUPS.items():
            data = load_trials("test", setup, bridge, conds, records, split_path)
            m = metrics(predict(model, data.units, device), data)
            m["records_sha256"] = hashlib.sha256("".join(data.sha256).encode()).hexdigest()
            res["groups"][name] = m
    torch.set_num_threads(threads)
    tmp = run_dir / "test.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, run_dir / "test.json")
    return res
