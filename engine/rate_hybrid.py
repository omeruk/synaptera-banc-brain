"""F5 stages B and C (prereg/14 sections 2-3, docs/10 section 2.3-2.4): rate FlyWire brain -> DN bridge -> rate BANC
VNC, trained together. Label `TRAINED`.

Interval k of :meth:`RateHybrid.run` (docs/10 section 2.4; L = bridge latency, W = window, in intervals):

    BANC DN clamp k  <- base, with every bridged target set to its unit value
                        v_u(k) = mean over j = k-L-W+1 .. k-L of mean over the unit's FlyWire DNs of relu(s_brain(j))
                        (intervals before 0 count as 0; the divisor is always W, as coupling/dn_bridge.py)
    brain state k    <- brain update(brain state k-1, drive u_k)
    VNC state k      <- VNC update(VNC state k-1, clamp k)

The newest brain state a clamp uses is that of interval k - L, so with L >= 1 the VNC never sees the brain state of
its own interval (rule 4). Which visual drive u_k belongs to interval k (docs/10: that of interval k-1) is the
caller's job, as for :class:`engine.rate_torch.RateNet`. The bridge units (matching, side, source family of a
derangement arm) come from a :class:`coupling.dn_bridge.DNBridge`; gains are 1 (prereg/14). A bridge with no units
(arm `bridge_off`) leaves the base clamp.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from engine.rate_torch import RateNet


class RateHybrid(nn.Module):
    """``brain``, ``vnc``: :class:`RateNet` (the VNC's inputs are the BANC DNs). ``units``: list of (FlyWire brain
    indices, VNC engine indices of the BANC targets) per bridge unit. ``base``: clamp of every VNC input node (in
    ``vnc.graph.inputs`` order) when no unit sets it (DNg02 = 1, every other DN 0)."""

    def __init__(self, brain: RateNet, vnc: RateNet, units, base, latency_intervals: int = 1, window_intervals: int = 5):
        super().__init__()
        if latency_intervals < 1:
            raise ValueError("bridge latency must be >= 1 interval (rule 4)")
        if window_intervals < 1:
            raise ValueError("window must be >= 1 interval")
        self.brain, self.vnc = brain, vnc
        self.L, self.W = int(latency_intervals), int(window_intervals)
        col = {int(i): k for k, i in enumerate(vnc.graph.inputs)}
        fw = np.unique(np.concatenate([np.asarray(f, np.int64) for f, _ in units])) if units else np.zeros(0, np.int64)
        pos = {int(i): k for k, i in enumerate(fw)}
        avg = np.zeros((len(units), len(fw)), np.float64)
        tgt_col, tgt_unit = [], []
        for u, (f, t) in enumerate(units):
            f = np.asarray(f, np.int64)
            avg[u, [pos[int(i)] for i in f]] = 1.0 / len(f)
            for i in np.asarray(t, np.int64):
                if int(i) not in col:
                    raise ValueError(f"bridge target {int(i)} is not a VNC input node")
                tgt_col.append(col[int(i)])
                tgt_unit.append(u)
        if len(set(tgt_col)) != len(tgt_col):
            raise ValueError("a VNC input node is the target of more than one unit")
        base = np.asarray(base, np.float64)
        if base.shape != (len(vnc.graph.inputs),):
            raise ValueError("base needs one value per VNC input node")
        dt = vnc.dtype
        self.register_buffer("fw_index", torch.from_numpy(fw), persistent=False)
        self.register_buffer("avg", torch.from_numpy(avg).to(dt), persistent=False)
        self.register_buffer("tgt_col", torch.as_tensor(tgt_col, dtype=torch.int64), persistent=False)
        self.register_buffer("tgt_unit", torch.as_tensor(tgt_unit, dtype=torch.int64), persistent=False)
        self.register_buffer("base", torch.from_numpy(base).to(dt), persistent=False)

    @property
    def n_units(self) -> int:
        return int(self.avg.shape[0])

    def _clamp(self, hist):
        """hist: (L + W - 1, n_units, batch), oldest first; the clamp uses the oldest W entries."""
        x = self.base[:, None].expand(-1, hist.shape[2])
        if self.n_units == 0:
            return x
        v = hist[: self.W].sum(0) / self.W
        return x.index_copy(0, self.tgt_col, v[self.tgt_unit])

    def _units_now(self, sb):
        return self.avg @ torch.relu(sb[self.fw_index])

    def _segment(self, sb, sv, hist, vb, lb, vv, lv, u, rec):
        out = []
        stb, stv = self.brain._st, self.vnc._st
        for k in range(len(u)):
            x = self._clamp(hist)
            sb = self.brain._step(sb, vb, lb, None, u[k], stb)
            if self.n_units:
                hist = torch.cat([hist[1:], self._units_now(sb)[None]])
            sv = self.vnc._step(sv, vv, lv, x, None, stv)
            out.append(sv[rec])
        return sb, sv, hist, torch.stack(out)

    def run(self, u, record=None, checkpoint_every: int = 0):
        """u: (T, batch, n_drives of the brain). record: VNC neuron indices to return (default all).
        Returns (VNC states (T, batch, len(record)), final brain state (batch, n_brain), final VNC state)."""
        T, batch = u.shape[:2]
        if u.shape[2] != len(self.brain.drives):
            raise ValueError("u must have one column per brain drive node")
        dev, dt = self.vnc.device, self.vnc.dtype
        ut = u.to(dev, dt).transpose(1, 2)
        sb, sv = self.brain.initial_state(batch), self.vnc.initial_state(batch)
        hist = torch.zeros(self.L + self.W - 1, self.n_units, batch, dtype=dt, device=dev)
        rec = (torch.arange(self.vnc.graph.n, device=dev) if record is None
               else torch.as_tensor(np.asarray(record), dtype=torch.int64, device=dev))
        vb, lb = self.brain.w * self.brain.gains(), self.brain.leaks()[:, None]
        vv, lv = self.vnc.w * self.vnc.gains(), self.vnc.leaks()[:, None]
        seg = checkpoint_every if checkpoint_every > 0 else T
        parts = []
        for a in range(0, T, seg):
            b = min(a + seg, T)
            if checkpoint_every > 0 and torch.is_grad_enabled():
                sb, sv, hist, r = checkpoint(self._segment, sb, sv, hist, vb, lb, vv, lv, ut[a:b], rec,
                                             use_reentrant=False)
            else:
                sb, sv, hist, r = self._segment(sb, sv, hist, vb, lb, vv, lv, ut[a:b], rec)
            parts.append(r)
        return torch.cat(parts).transpose(1, 2), sb.T, sv.T

    @staticmethod
    def units_from_bridge(bridge, fw_brain_index) -> list:
        """(FlyWire brain indices, BANC target indices) per unit of a DNBridge; ``fw_brain_index`` maps the rows of the
        bridge's FlyWire DN table to brain engine indices."""
        fw_brain_index = np.asarray(fw_brain_index, np.int64)
        return [(fw_brain_index[u["fw_pos"]], np.asarray(u["banc_index"], np.int64)) for u in bridge.units]

    def record(self) -> dict:
        return dict(engine="rate_hybrid", label="TRAINED", brain=self.brain.record(), vnc=self.vnc.record(),
                    bridge=dict(n_units=self.n_units, n_targets=int(len(self.tgt_col)),
                                n_fw_neurons=int(len(self.fw_index)), latency_intervals=self.L,
                                window_intervals=self.W, gain=1.0, base_nonzero=int((self.base != 0).sum())))
