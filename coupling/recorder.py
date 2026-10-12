"""HDF5 run record: labelled behaviour fields, streamed spikes, engine inputs, meta.

Adapted from synaptera@c94796d:flight/recorder.py (the HDF5 schema, behaviour arrays and meta idea). Changes:
  * every behaviour field carries a source LABEL (BRAIN, VNC, HAND, REFLEX, TRAINED, FLYVIS, STATE) stored as a
    dataset attribute (CLAUDE.md rule 1), and the run meta carries the coupling latency (rule 4);
  * an existing result file is never overwritten (review M1): FileExistsError unless ``overwrite=True`` is passed
    explicitly (recorded in the meta);
  * the file is written under ``<name>.part`` and renamed on a clean close, so a crash leaves no half-written
    file under a result-like name (L7);
  * spikes are streamed to disk interval by interval through ``spike_sink`` and never accumulate in RAM (L9);
  * the engine inputs (rate per group and interval) are stored, so ``coupling/replay.py`` can audit the engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

LABELS = ("BRAIN", "VNC", "HAND", "REFLEX", "TRAINED", "FLYVIS", "STATE", "REPLAY")
# REPLAY: a command term replayed open-loop from an earlier record (a regression test of the port, never a result)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple = ()
    dtype: type = np.float64
    label: str = "STATE"
    note: str = ""

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"{self.name}: label {self.label!r} not in {LABELS}")


def _f(name, shape=(), dtype=np.float64, label="STATE", note=""):
    return FieldSpec(name, tuple(shape), dtype, label, note)


# Behaviour fields of the flight record (v1 BEHAVIOR_FIELDS without the legacy olfaction / all-DN terms).
FLIGHT_FIELDS = (
    _f("t"), _f("pos", (3,)), _f("vel", (3,)), _f("speed"), _f("quat", (4,)), _f("omega", (3,)), _f("heading"),
    _f("dist_to_food"), _f("phase", (), np.int8, note="codes in meta phase_codes"),
    # turn / collective terms: one label each
    _f("turn_total", label="STATE", note="clipped sum of the commanded turn terms (the wing yaw command)"),
    _f("turn_brain", label="BRAIN"), _f("turn_hand", label="HAND"), _f("turn_flyvis", label="FLYVIS"),
    _f("turn_reflex", label="REFLEX", note="haltere yaw torque as the equivalent turn_bias; physics, not commanded"),
    _f("thrust_hand", label="HAND"), _f("pitch_hand", label="HAND"), _f("roll_hand", label="HAND"),
    _f("vz_des", label="HAND"), _f("v_des_h", label="HAND"), _f("lift_frac"), _f("wings_on", (), np.int8),
    _f("stroke_amp_L"), _f("stroke_amp_R"), _f("stroke_freq"), _f("force", (3,)), _f("torque", (3,)),
    _f("amp_clipped_substeps", (), np.int32, note="sub-steps with a clipped stroke amplitude (review H5)"),
    # brain readouts and the bridge
    _f("dn_readout_rate", (15, 2), label="BRAIN", note="filtered readout rate per type x L/R (Hz per neuron)"),
    _f("mn9_rate", label="BRAIN"), _f("proboscis_extended", (), np.int8, label="BRAIN"),
    _f("steer_raw", (2,), label="BRAIN"), _f("steer_norm_t", (2,), label="BRAIN"),
    _f("steer_norm", label="BRAIN"), _f("net_rate", label="BRAIN"),
    # vision
    _f("t45_rate_L", label="FLYVIS"), _f("t45_rate_R", label="FLYVIS"), _f("loom_L", label="FLYVIS"),
    _f("loom_R", label="FLYVIS"), _f("expansion_rate"), _f("height_above"), _f("platform_azimuth"),
    # contacts and timing
    _f("platform_contact", (), np.int8), _f("tower_contact", (), np.int8), _f("tower_penetration"),
    _f("min_tower_clearance"), _f("is_feeding", (), np.int8), _f("leg_pose", (), np.int8),
    _f("yaw_perturb_torque", note="external body-z torque pulse (uN mm); experiment design"),
    _f("step_time"), _f("rss_gb"),
    # coupling
    _f("cmd_source", (), np.int32, note="interval whose engine output was applied (-1 = neutral command)"),
)


class Recorder:
    def __init__(self, path, fields=FLIGHT_FIELDS, meta: dict | None = None, n_intervals: int = 0,
                 overwrite: bool = False, input_groups: dict | None = None):
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; result files are never overwritten (pass overwrite=True "
                                  "explicitly or use a new name)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.part.exists():
            self.part.unlink()
        self.fields = {f.name: f for f in fields}
        self.n = int(n_intervals)
        self.meta = dict(meta or {})
        self.meta["overwrite"] = bool(overwrite)
        self.f = h5py.File(self.part, "w")
        b = self.f.create_group("behavior")
        self._b = {}
        for f in fields:
            ds = b.create_dataset(f.name, shape=(self.n,) + f.shape, dtype=f.dtype)
            if f.dtype in (np.float64, np.float32):
                ds[...] = np.nan
            ds.attrs["label"] = f.label
            if f.note:
                ds.attrs["note"] = f.note
            self._b[f.name] = ds
        s = self.f.create_group("spikes")
        self._st = s.create_dataset("t", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(65536,),
                                    compression="gzip", compression_opts=1)
        self._si = s.create_dataset("i", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(65536,),
                                    compression="gzip", compression_opts=1)
        s.attrs["i"] = "engine neuron index (global index of the connectome when a DEV subnet is used)"
        self._inputs = self.f.create_group("inputs")
        self._in = {}
        for name, size in (input_groups or {}).items():
            self._in[name] = self._inputs.create_dataset(name, shape=(self.n, size), dtype=np.float32,
                                                         compression="gzip", compression_opts=1)
        self.n_done = 0
        self.closed = False

    # ── streaming ───────────────────────────────────────────────────────────
    def spike_sink(self, t, i) -> None:
        """Engine ``spike_sink``: append one interval's spikes."""
        if len(t) == 0:
            return
        n0 = self._st.shape[0]
        self._st.resize(n0 + len(t), axis=0)
        self._si.resize(n0 + len(t), axis=0)
        self._st[n0:] = np.asarray(t, np.float32)
        self._si[n0:] = np.asarray(i, np.int32)

    def put(self, k: int, **vals) -> None:
        for name, v in vals.items():
            if name not in self._b:
                raise KeyError(f"unknown behaviour field {name!r}")
            self._b[name][k] = v
        self.n_done = max(self.n_done, k + 1)

    def put_inputs(self, k: int, drive: dict) -> None:
        for name, v in drive.items():
            if name in self._in:
                self._in[name][k] = np.broadcast_to(np.asarray(v, np.float32), self._in[name].shape[1:])

    # ── finish ──────────────────────────────────────────────────────────────
    def close(self, extra_meta: dict | None = None) -> Path:
        if self.closed:
            return self.path
        meta = dict(self.meta, **(extra_meta or {}), n_intervals_done=int(self.n_done))
        m = self.f.create_group("meta")
        for k, v in meta.items():
            m.attrs[k] = json.dumps(v, default=_json_default) if isinstance(v, (dict, list, tuple)) else v
        # unfinished runs keep NaN / 0 padding after n_intervals_done (meta); `complete` says which
        self.f.attrs["complete"] = bool(self.n_done == self.n)
        self.f.close()
        os.replace(self.part, self.path)
        self.closed = True
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:                       # a crash leaves only <name>.part, never a result-like name (L7)
            try:
                self.f.close()
            finally:
                self.closed = True


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
