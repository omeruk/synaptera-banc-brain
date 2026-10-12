"""Causal brain-body coupling: one Brian2 ``run()``, a command queue with an explicit latency.

Written from docs/05_ARCHITECTURE_PLAN.md 3.5 (pattern of synaptera@c94796d:fly_flight_brain_body_simulation.py,
whose main loop ran ``brain.step()`` and then drove the body in the SAME interval with the spikes just produced:
review H1 / v1 bug Y1).

Dataflow per interval k (length ``interval``), boundary j = k + 1 reached after interval k ran:
    out_k  = decider.decide(k, engine)            neural output of interval k (readouts -> command)
    queue.push(k, out_k)
    cmd    = queue.pop(k)                         the command decided ``latency`` intervals earlier
    rec    = plant.act(k, cmd, dt)                body physics of interval k with that command
    drive  = plant.sense(k + 1)                   input for interval k + 1 from the body state at the boundary
    engine.set_inputs(drive)
With latency = 1 (the causal default) the body in interval k is driven by the engine output of interval k - 1
and never by spikes produced in the interval it is stepped over. latency = 0 reproduces the v1 same-interval
coupling; it is refused unless ``allow_acausal=True`` and is then labelled ACAUSAL in the record.

Two execution modes with the same dataflow: ``single_run`` (the engine runs all intervals inside one
``Network.run`` and the boundary logic is a network operation; no per-interval run() overhead) and
``per_interval`` (one ``advance`` call per interval; for engines without callbacks and for measuring the
overhead).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


class Plant(Protocol):
    """The world side: body + sensors."""

    def sense(self, k: int) -> dict:
        """Engine input drive (group name -> Hz) for interval k from the current body state."""

    def act(self, k: int, command: Any, dt_s: float) -> dict:
        """Step the body over interval k with ``command``; returns the per-interval record fields."""


class Decider(Protocol):
    """Neural output -> command (readouts, bridge, HAND terms)."""

    neutral: Any

    def decide(self, k: int, engine) -> Any:
        """Command from the engine output of interval k. May expose ``record`` (dict) for the recorder."""


class CommandQueue:
    """FIFO with a fixed latency in intervals. ``pop(k)`` returns (command, source interval); before any decided
    command exists it returns the neutral command with source -1."""

    def __init__(self, latency: int, neutral):
        if latency < 0:
            raise ValueError("latency must be >= 0")
        self.latency = int(latency)
        self.neutral = neutral
        self._items: dict[int, Any] = {}

    def push(self, k: int, command) -> None:
        self._items[int(k)] = command

    def pop(self, k: int):
        src = int(k) - self.latency
        cmd = self._items.get(src)
        # drop what can no longer be asked for
        for old in [i for i in self._items if i < src]:
            del self._items[old]
        return (self.neutral, -1) if cmd is None else (cmd, src)


@dataclass(frozen=True)
class LoopConfig:
    interval_ms: float = 5.0          # configs/assumptions.json: coupling.interval_ms
    latency_intervals: int = 1
    mode: str = "single_run"          # "single_run" | "per_interval"
    allow_acausal: bool = False

    def __post_init__(self):
        if self.mode not in ("single_run", "per_interval"):
            raise ValueError(self.mode)
        if self.latency_intervals < 0:
            raise ValueError("latency must be >= 0")
        if self.latency_intervals == 0 and not self.allow_acausal:
            raise ValueError("latency 0 drives the body with the spikes of the same interval (v1 bug Y1); "
                             "pass allow_acausal=True only for the labelled v1 reproduction")

    @property
    def interval_s(self) -> float:
        return self.interval_ms * 1e-3

    def record(self) -> dict:
        d = asdict(self)
        d["interval_s"] = self.interval_s
        d["causality"] = "ACAUSAL (v1 reproduction, same-interval)" if self.latency_intervals == 0 else \
            f"causal: body in interval k uses the engine output of interval k - {self.latency_intervals}"
        return d


class CoupledLoop:
    """Runs ``n_intervals`` of engine + plant under :class:`LoopConfig`. ``on_interval(k, rec)`` receives, after
    every interval, the plant record merged with ``cmd_source`` (the interval whose output was applied; -1 =
    neutral) and ``decided`` (the decider's own record)."""

    def __init__(self, engine, plant: Plant, decider: Decider, config: LoopConfig, on_interval=None):
        self.engine, self.plant, self.decider, self.cfg = engine, plant, decider, config
        self.queue = CommandQueue(config.latency_intervals, decider.neutral)
        self.on_interval = on_interval
        self.n_done = 0
        self._n = 0

    def _boundary(self, j: int, eng) -> None:
        cfg = self.cfg
        if j > 0:
            k = j - 1
            out = self.decider.decide(k, eng)
            self.queue.push(k, out)
            cmd, src = self.queue.pop(k)
            rec = dict(self.plant.act(k, cmd, cfg.interval_s))
            rec["cmd_source"] = src
            rec["decided"] = getattr(self.decider, "record", None)
            self.n_done = j
            if self.on_interval is not None:
                self.on_interval(k, rec)
        if j < self._n:
            eng.set_inputs(self.plant.sense(j))

    def run(self, n_intervals: int) -> None:
        self._n = int(n_intervals)
        cfg = self.cfg
        if cfg.mode == "single_run":
            self.engine.run_coupled(self._n, cfg.interval_ms, self._boundary)
        else:
            self._boundary(0, self.engine)
            for j in range(1, self._n + 1):
                self.engine.advance(cfg.interval_ms)
                self._boundary(j, self.engine)
