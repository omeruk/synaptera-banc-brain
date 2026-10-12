"""The NeuralEngine interface every neural arm implements.

Written from docs/05_ARCHITECTURE_PLAN.md section 3.1 (v1 had one concrete class, flight/brain.py:FlightBrain,
with a fixed 25 ms ``step``). The engine knows nothing about the body, the coupling interval or the command
queue: the caller (coupling/scheduler.py) decides how long each ``advance`` is, from the configuration, and
delays the engine's output by one interval (CLAUDE.md rule 4).
"""
from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class NeuralEngine(Protocol):
    n_neurons: int

    def reset(self, seed: int) -> None:
        """Back to the initial state at t = 0, all input rates 0, with a fresh noise stream ``seed``."""

    def set_inputs(self, drive: Mapping[str, "float | np.ndarray"]) -> None:
        """Input group name -> rate in Hz (scalar, or one value per neuron of the group). Groups not named keep
        their rate; an unknown name raises KeyError."""

    def advance(self, dt_ms: float) -> None:
        """Simulate ``dt_ms`` (one coupling interval; a whole number of engine time steps)."""

    def rates(self, groups: Sequence[str], window_ms: float) -> "dict[str, np.ndarray]":
        """Per-neuron firing rate (Hz) of each named group over the last ``window_ms``."""

    def state_dict(self) -> dict:
        """Everything needed to audit or replay: time, seed, input rates, weights hash, model hash, parameters."""
