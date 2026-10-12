"""Brian2 LIF engine: the Shiu et al. 2023 network as the *unmodified-model* reference arm.

Ported from synaptera@c94796d:flight/brain.py (FlightBrain). Changes with respect to v1:
  * NeuralEngine interface (reset / set_inputs / advance / rates / state_dict); ``advance(dt_ms)`` takes the
    coupling interval from the caller instead of the fixed 25 ms DECISION_INTERVAL.
  * M4: Poisson input has no synaptic delay (Shiu's PoissonInput adds to v directly); v1 used the 1.8 ms model
    delay. Poisson targets keep rfc = 0 as in model.py:poi().
  * M3 / L6: nulls and the DEV subnet are graph transforms (connectome.controls) applied before the engine; the
    engine warns on a DEV subnet and on DEV + control. ``match_weight_scale`` matches the activity level of a
    control to the intact graph (open-loop criterion, no behaviour involved).
  * The network is built from the common-schema Connectome (any dataset), with the model's own parameters,
    equations and construction recipe taken unchanged from the hashed brain_model/model.py.
  * Not carried: inputs="legacy", the --apl-graded / --olfaction-full / --leg-grn / --vision-boundary switches,
    the three NT editors (now connectome.sign_rules), shuffle_seed (now connectome.controls).
Brian2's SpikeMonitor still keeps every spike of a run (review L9); ``spike_sink`` receives each interval's
spikes so the recorder (coupling/recorder.py) can stream them to disk.

Randomness: Brian2 draws from one process-global generator, so several engines in one process share a noise
stream. ``reset(seed)`` re-seeds it: call it right before every run you want to be reproducible (the
coupling scheduler and ``measure_rate`` do).

Cost of a short interval (measured on this aarch64 machine, 2k-60k neurons, cython target): every
``Network.run`` call carries a fixed overhead of ~50-90 ms independent of network size, ~40 ms of it Brian's
debug logging (``quiet_logging=True`` raises the handlers to WARNING). At a 5 ms coupling interval 1 s of
simulated time is 200 calls, i.e. ~10-15 s of overhead before any synaptic work; GeNN or an in-run
network_operation coupling removes it: ``run_coupled`` runs all intervals inside ONE ``Network.run`` and calls
back at every boundary (measured 2-4 ms per interval instead of 56-61 ms, identical spike counts).
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import warnings
from collections import deque
from pathlib import Path

import numpy as np
from brian2 import Hz, Network, NetworkOperation, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses, defaultclock, ms
from brian2 import seed as brian_seed
from brian2.utils.logger import BrianLogger

from connectome.schema import Connectome

CLOCK_DT_MS = 0.1                 # configs/assumptions.json: engine.clock_dt_ms (the clock model.py was run at)
HISTORY_MS = 200.0                # recent-spike window kept for rates() (engine.history_ms)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "brain_model" / "model.py"
SHIU_MODEL_SHA256 = "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e"   # philshiu/Drosophila_brain_model@91bdd1e
_NUMERIC_PARAMS = ("v_0", "v_rst", "v_th", "t_mbr", "tau", "t_rfc", "t_dly", "w_syn", "r_poi", "f_poi")
_MODEL_CACHE: dict = {}


def load_shiu_model(path=DEFAULT_MODEL_PATH, check_hash: bool = True):
    """Import Shiu et al.'s model.py from ``path`` (cached); the SHA-256 must equal the pinned one."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if check_hash and digest != SHIU_MODEL_SHA256:
        raise ValueError(f"{path}: SHA-256 {digest} != pinned {SHIU_MODEL_SHA256}")
    key = (str(path.resolve()), digest)
    if key not in _MODEL_CACHE:
        spec = importlib.util.spec_from_file_location("shiu_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODEL_CACHE[key] = (mod, digest)
    return _MODEL_CACHE[key]


def _steps(dt_ms: float) -> int:
    k = dt_ms / CLOCK_DT_MS
    n = int(round(k))
    if n < 1 or abs(k - n) > 1e-6:
        raise ValueError(f"dt_ms={dt_ms} is not a positive multiple of the {CLOCK_DT_MS} ms clock")
    return n


class Brian2LIFEngine:
    """Shiu LIF network on a signed :class:`Connectome`.

    input_groups: name -> neuron indices driven by Poisson spikes (disjoint). groups: further named readout
    groups (indices). The group ``"all"`` always exists. Indices are engine indices of ``cx``."""

    def __init__(self, cx: Connectome, *, input_groups=None, groups=None, seed: int = 0,
                 history_ms: float = HISTORY_MS, model_path=DEFAULT_MODEL_PATH, check_model_hash: bool = True,
                 spike_sink=None, quiet_logging: bool = True, poisson_delay_v1: bool = False,
                 verbose: bool = False):
        if quiet_logging:
            for h in (BrianLogger.file_handler, BrianLogger.console_handler):
                if h is not None and h.level < logging.WARNING:
                    h.setLevel(logging.WARNING)
        if not cx.signed:
            raise ValueError(f"{cx.name}: the engine needs a signed connectome (apply a sign rule first)")
        log = print if verbose else (lambda *a, **k: None)
        self._warn_meta(cx.meta)
        model, self.model_sha256 = load_shiu_model(model_path, check_model_hash)
        self.params = dict(model.default_params)
        self.n_neurons = cx.n_neurons
        self.name = cx.name
        self.meta = dict(cx.meta)
        self.global_index = np.asarray(cx.meta.get("global_index", np.arange(cx.n_neurons)), dtype=np.int64)
        self._sink = spike_sink
        self.poisson_delay_v1 = bool(poisson_delay_v1)
        self._hist_steps = _steps(history_ms)
        defaultclock.dt = CLOCK_DT_MS * ms

        # groups
        self.groups = {"all": np.arange(self.n_neurons)}
        for name, idx in dict(groups or {}).items():
            self.groups[name] = np.asarray(idx, dtype=np.int64)
        self._input_names = tuple(dict(input_groups or {}))
        seen = np.zeros(self.n_neurons, np.int8)
        for name, idx in dict(input_groups or {}).items():
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) == 0:
                raise ValueError(f"input group {name!r} is empty")
            if idx.min() < 0 or idx.max() >= self.n_neurons:
                raise ValueError(f"input group {name!r} has indices outside the network")
            self.groups[name] = idx
            seen[idx] += 1
        if seen.max(initial=0) > 1:
            raise ValueError("a neuron belongs to more than one Poisson input group")

        # network: the recipe of model.py:create_model, from arrays
        p = self.params
        pre, post = cx.edge_endpoints()
        w_counts = cx.signed_weights().astype(np.float64)
        neu = NeuronGroup(N=self.n_neurons, model=p["eqs"], method="linear", threshold=p["eq_th"],
                          reset=p["eq_rst"], refractory="rfc", namespace=p)
        neu.v = p["v_0"]
        neu.g = 0
        neu.rfc = p["t_rfc"]
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p["t_dly"])
        syn.connect(i=pre, j=post)
        syn.w = w_counts * p["w_syn"]
        del w_counts
        spk = SpikeMonitor(neu)
        self.neu, self.syn, self.spk_mon = neu, syn, spk

        # Poisson inputs: no synaptic delay (M4), refractory period off for the targets (model.py:poi)
        w_in = p["w_syn"] * p["f_poi"]
        objs, self._poisson, self.poisson_synapses = [], {}, {}
        for name in self._input_names:
            tgt = self.groups[name]
            pg = PoissonGroup(len(tgt), rates=0 * Hz)
            if poisson_delay_v1:      # named variant "poisson_delay_v1": the v1 path (model delay on the input)
                s = Synapses(pg, neu, "w : volt", on_pre="v += w", delay=p["t_dly"])
            else:
                s = Synapses(pg, neu, "w : volt", on_pre="v += w")
            s.connect(i=np.arange(len(tgt)), j=tgt)
            s.w = w_in
            neu.rfc[tgt] = 0 * ms
            self._poisson[name] = pg
            self.poisson_synapses[name] = s
            objs += [pg, s]
        self.net = Network(neu, syn, spk, *objs)
        self._base_w = np.array(syn.w_[:])
        self._weight_scale = 1.0
        self._store()
        self._clear()
        self.seed = None
        self.reset(seed)
        log(f"  engine: {self.n_neurons:,} neurons, {len(syn):,} synapses, inputs {self._input_names}")

    # ── warnings (L6) ───────────────────────────────────────────────────────
    @staticmethod
    def _warn_meta(meta):
        if meta.get("dev_subnet") and meta.get("control"):
            warnings.warn("DEV subnet combined with the control "
                          f"{meta['control'].get('name')!r}: not a valid null control; never a result",
                          UserWarning, stacklevel=3)
        elif meta.get("dev_subnet"):
            warnings.warn("DEV subnet: development only, never a result", UserWarning, stacklevel=3)

    # ── state ───────────────────────────────────────────────────────────────
    def _store(self):
        self.net.store("fresh")

    def _clear(self):
        self._steps_done = 0
        self._n_seen = 0
        self._total = np.zeros(self.n_neurons, np.int64)
        self._hist = deque()                         # (step_end, spike step k int64, neuron i int32)
        self._rates = {name: 0.0 for name in self._input_names}

    def reset(self, seed: int) -> None:
        self.net.restore("fresh")
        brian_seed(int(seed))
        self.seed = int(seed)
        self._clear()
        self._n_seen = int(self.spk_mon.num_spikes)

    def set_weight_scale(self, scale: float) -> None:
        """Multiply every synaptic weight by ``scale`` (relative to the graph's own weights) and make this the
        new initial state. Used only by the activity-matched control; the reference arm keeps 1.0."""
        if not scale > 0:
            raise ValueError("scale must be positive")
        self.syn.w_[:] = self._base_w * float(scale)
        self._weight_scale = float(scale)
        self._store()
        self.reset(self.seed if self.seed is not None else 0)

    # ── NeuralEngine ────────────────────────────────────────────────────────
    def set_inputs(self, drive) -> None:
        for name, v in drive.items():
            if name not in self._poisson:
                raise KeyError(name)
            n = len(self.groups[name])
            if np.ndim(v) == 0:
                val = float(v)
                if val < 0:
                    raise ValueError(f"{name}: negative rate")
                self._rates[name] = val
                self._poisson[name].rates = val * Hz
            else:
                arr = np.asarray(v, dtype=float)
                if arr.shape != (n,):
                    raise ValueError(f"{name}: {arr.shape} rates for {n} neurons")
                if (arr < 0).any():
                    raise ValueError(f"{name}: negative rate")
                self._rates[name] = arr.copy()
                self._poisson[name].rates = arr * Hz

    def _collect(self, n_steps: int) -> None:
        """Account for ``n_steps`` just simulated: read only the new part of the monitor, update the totals and
        the recent-spike history, feed the spike sink."""
        self._steps_done += n_steps
        n1 = int(self.spk_mon.num_spikes)
        i_new = np.asarray(self.spk_mon.i[self._n_seen:n1], dtype=np.int64)
        t_new = np.asarray(self.spk_mon.t_[self._n_seen:n1], dtype=np.float64)
        self._n_seen = n1
        self._total += np.bincount(i_new, minlength=self.n_neurons)
        k = np.rint(t_new / (CLOCK_DT_MS * 1e-3)).astype(np.int64)
        self._hist.append((self._steps_done, k, i_new.astype(np.int32)))
        while self._hist and self._hist[0][0] <= self._steps_done - self._hist_steps:
            self._hist.popleft()
        if self._sink is not None:
            self._sink(t_new, self.global_index[i_new].astype(np.int32))

    def advance(self, dt_ms: float) -> None:
        n = _steps(dt_ms)
        self.net.run(n * defaultclock.dt)
        self._collect(n)

    def run_coupled(self, n_intervals: int, interval_ms: float, on_boundary) -> None:
        """Simulate ``n_intervals`` coupling intervals inside ONE ``Network.run`` call.

        ``on_boundary(j, engine)`` is called at t = j * interval for j = 0 .. n_intervals: j = 0 before anything
        ran, j >= 1 right after interval j-1 completed (the engine's spike counts, rates() and history already
        include it), j = n_intervals after the run. The callback may call ``set_inputs`` (takes effect for the
        next interval) and read ``rates`` / ``spike_counts``. This removes the fixed ~50-90 ms per-``run()``
        overhead (measured in coupling/benchmark and docs) that one ``advance`` call per interval carries."""
        n = _steps(interval_ms)
        state = {"j": 0}

        def op(t):
            j = state["j"]
            if j > 0:
                self._collect(n)
            state["j"] = j + 1
            on_boundary(j, self)

        no = NetworkOperation(op, dt=n * defaultclock.dt, when="start", name="coupling_boundary")
        self.net.add(no)
        try:
            self.net.run(n_intervals * n * defaultclock.dt)
        finally:
            self.net.remove(no)
        self._collect(n)                                   # the last interval completed with the run
        on_boundary(n_intervals, self)

    def rates(self, groups, window_ms: float) -> dict:
        w = _steps(window_ms)
        if w > self._hist_steps:
            raise ValueError(f"window {window_ms} ms exceeds the engine history {self._hist_steps * CLOCK_DT_MS} ms")
        w = min(w, self._steps_done)
        if w == 0:
            return {g: np.zeros(len(self.groups[g])) for g in groups}
        lo = self._steps_done - w
        counts = np.zeros(self.n_neurons, np.int64)
        for _, k, i in self._hist:
            counts += np.bincount(i[k >= lo], minlength=self.n_neurons)
        dur_s = w * CLOCK_DT_MS * 1e-3
        return {g: counts[self.groups[g]] / dur_s for g in groups}

    def spike_counts(self) -> np.ndarray:
        """Cumulative spike count per neuron since the last reset (copy)."""
        return self._total.copy()

    @property
    def t_ms(self) -> float:
        return self._steps_done * CLOCK_DT_MS

    def state_dict(self, include_dynamics: bool = False) -> dict:
        w = np.asarray(self.syn.w_[:])
        out = dict(
            engine="brian2_lif", dataset=self.name, t_ms=self.t_ms, steps=self._steps_done, seed=self.seed,
            n_neurons=self.n_neurons, n_synapses=int(len(w)), clock_dt_ms=CLOCK_DT_MS,
            weights_sha256=hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest(),
            weight_scale=self._weight_scale, model_sha256=self.model_sha256,
            params={k: float(self.params[k]) for k in _NUMERIC_PARAMS},
            input_rates={k: (v if np.ndim(v) == 0 else np.asarray(v).copy()) for k, v in self._rates.items()},
            variant=("poisson_delay_v1" if self.poisson_delay_v1 else None),
            spike_total=int(self._total.sum()), dev_subnet=bool(self.meta.get("dev_subnet", False)),
            control=self.meta.get("control"), sign_rules=self.meta.get("sign_rules", []))
        if include_dynamics:
            out["v"] = np.asarray(self.neu.v_[:]).copy()
            out["g"] = np.asarray(self.neu.g_[:]).copy()
        return out


# ── activity matching of a control (M3) ─────────────────────────────────────────────────────────────────────
def measure_rate(engine: Brian2LIFEngine, drive, *, group: str = "all", duration_ms: float = 500.0,
                 discard_ms: float = 100.0, interval_ms: float = 5.0, seed: int = 0) -> float:
    """Mean firing rate (Hz) of ``group`` under a constant ``drive``, after ``discard_ms`` of transient."""
    if not 0 <= discard_ms < duration_ms:
        raise ValueError("need 0 <= discard_ms < duration_ms")
    engine.reset(seed)
    engine.set_inputs(drive)
    t = 0.0
    start, t_start = (engine.spike_counts(), 0.0) if discard_ms == 0 else (None, None)
    while t < duration_ms - 1e-9:
        engine.advance(interval_ms)
        t += interval_ms
        if start is None and t >= discard_ms - 1e-9:
            start, t_start = engine.spike_counts(), t
    counts = engine.spike_counts() - start
    idx = engine.groups[group]
    return float(counts[idx].sum()) / (len(idx) * (t - t_start) * 1e-3)


def match_weight_scale(engine: Brian2LIFEngine, target_rate_hz: float, drive, *, group: str = "all",
                       duration_ms: float = 500.0, discard_ms: float = 100.0, interval_ms: float = 5.0,
                       seed: int = 0, rel_tol: float = 0.05, lo: float = 0.25, hi: float = 4.0,
                       max_iter: int = 16) -> dict:
    """Find the global weight scale at which ``engine`` fires at ``target_rate_hz`` (mean over ``group``) under
    ``drive``, by bisection on log(scale) (rate is assumed to increase with scale). Open-loop; the criterion
    is the reference network's rate, never a behaviour. Sets the engine to the found scale. Raises if the
    target cannot be bracketed within [1e-2, 1e2]."""
    if target_rate_hz <= 0:
        raise ValueError("target rate must be positive")
    hist = []

    def rate_at(s):
        engine.set_weight_scale(s)
        r = measure_rate(engine, drive, group=group, duration_ms=duration_ms, discard_ms=discard_ms,
                         interval_ms=interval_ms, seed=seed)
        hist.append((float(s), r))
        return r

    r_lo, r_hi = rate_at(lo), rate_at(hi)
    while r_lo > target_rate_hz and lo > 1e-2:
        lo, r_lo = lo / 2, rate_at(lo / 2)
    while r_hi < target_rate_hz and hi < 1e2:
        hi, r_hi = hi * 2, rate_at(hi * 2)
    if not (r_lo <= target_rate_hz <= r_hi):
        raise RuntimeError(f"target {target_rate_hz} Hz not bracketed: rate {r_lo:.3g} Hz at scale {lo}, "
                           f"{r_hi:.3g} Hz at scale {hi}")
    best_s, best_r = (lo, r_lo) if abs(r_lo - target_rate_hz) <= abs(r_hi - target_rate_hz) else (hi, r_hi)
    for _ in range(max_iter):
        if abs(best_r - target_rate_hz) <= rel_tol * target_rate_hz:
            break
        mid = float(np.sqrt(lo * hi))
        r = rate_at(mid)
        if abs(r - target_rate_hz) < abs(best_r - target_rate_hz):
            best_s, best_r = mid, r
        if r < target_rate_hz:
            lo = mid
        else:
            hi = mid
    engine.set_weight_scale(best_s)
    return dict(scale=float(best_s), rate_hz=float(best_r), target_hz=float(target_rate_hz),
                converged=abs(best_r - target_rate_hz) <= rel_tol * target_rate_hz, history=hist)
