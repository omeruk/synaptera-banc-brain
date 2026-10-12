"""Replay audit: re-apply the recorded engine inputs to a freshly built engine and compare the spikes.

Pattern taken from synaptera@c94796d:scripts/ladder_replay.py (replay of one recorded flight into a brain with
the flight's seed; ``dn_identical`` / ``all_neuron_counts_identical``). The engine is deterministic given its
seed and inputs, so a record made with coupling/recorder.py must be reproduced bit-exactly; any difference is
an engine / build change (or a different graph) and is reported, not hidden.
"""
from __future__ import annotations

import h5py
import numpy as np


def replay(engine, record_path, interval_ms: float | None = None, seed: int | None = None) -> dict:
    """Replay ``record_path`` into ``engine`` (freshly built, same graph). Returns a report dict:
    identical (bool), n_spikes_record / n_spikes_replay, n_intervals, spikes_differing."""
    with h5py.File(record_path, "r") as f:
        meta = f["meta"].attrs
        n = int(meta["n_intervals_done"])
        if interval_ms is None:
            interval_ms = float(meta["interval_ms"])
        if seed is None:
            seed = int(meta["seed"])
        names = list(f["inputs"])
        inputs = {k: f["inputs"][k][:n] for k in names}
        rec_t, rec_i = f["spikes/t"][:], f["spikes/i"][:]
    got_t, got_i = [], []
    sink_old = engine._sink
    engine._sink = lambda t, i: (got_t.append(np.asarray(t, np.float32)), got_i.append(np.asarray(i, np.int32)))
    try:
        engine.reset(seed)
        for k in range(n):
            engine.set_inputs({name: inputs[name][k] for name in names})
            engine.advance(interval_ms)
    finally:
        engine._sink = sink_old
    t = np.concatenate(got_t) if got_t else np.zeros(0, np.float32)
    i = np.concatenate(got_i) if got_i else np.zeros(0, np.int32)
    same = len(t) == len(rec_t) and np.array_equal(i, rec_i) and np.array_equal(t, rec_t)
    return dict(identical=bool(same), n_intervals=n, n_spikes_record=int(len(rec_t)), n_spikes_replay=int(len(t)),
                spikes_differing=int(abs(len(t) - len(rec_t)) if len(t) != len(rec_t) else (i != rec_i).sum()),
                seed=seed, interval_ms=interval_ms)
