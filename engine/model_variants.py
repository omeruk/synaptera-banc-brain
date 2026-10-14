"""Neuron model v2: named model variants of the Shiu LIF engine (docs/09_NEURON_MODEL_V2.md, prereg/12).

The reference arm R (``engine/lif_brian2.py`` with ``model_variant=None``) is not modified (CLAUDE.md rule 8). Each
switch below is a named model variant; an arm is a set of switches:

    R    reference (Shiu et al. 2023, unchanged)
    A    adapt
    AC   adapt + cond
    ACG  adapt + cond + gap

``adapt``  spike-triggered and subthreshold adaptation of the AdEx model (Brette & Gerstner 2005, J Neurophysiol 94:3637,
           Table 1), written in the voltage units of the Shiu equation (dividing by the leak conductance g_L):
               t_mbr dv/dt = v_0 - v + I_syn - w_ad
               tau_w dw_ad/dt = a (v - v_0) - w_ad        at a spike: w_ad += b
           with a = a_BG / g_L,BG (dimensionless) and b = b_BG / g_L,BG (volt). The exponential spike term of AdEx is not
           added (the Shiu threshold and reset are kept); only the adaptation current is.
``cond``   conductance synapses: I_syn = g_e (E_exc - v) + g_i (E_inh - v); g_e, g_i are conductances relative to the
           leak (dimensionless), decay with the Shiu synaptic time constant tau, and jump by n_syn * dg_exc or
           n_syn * |dg_inh| at a presynaptic spike (after the Shiu delay t_dly). dg is set so that the peak PSP of ONE
           synapse at rest equals the peak Shiu PSP of one synapse (w_syn): ``calibrate_dg`` (deterministic, no run).
``gap``    electrical synapses only for the pairs registered in configs/gap_junctions.yaml (published physiological
           evidence). Rectifying (Phelan et al. 2008): current flows from the registered presynaptic cell to the
           postsynaptic one. The point model has no spike waveform, so the junction is reduced to its measured effect,
           1:1 sub-millisecond transmission of a presynaptic spike: an instantaneous (no delay) jump of the
           postsynaptic v by GAP_DV at each presynaptic spike.

Unchanged in every arm: v_0, v_rst, v_th, t_mbr, tau, t_rfc, t_dly, the sign rule, and the Poisson input path
(v += w_syn * f_poi, no delay, refractory period off for the targets).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GAP_CONFIG = REPO_ROOT / "configs" / "gap_junctions.yaml"

# adapt: Brette & Gerstner 2005 Table 1 (general value, origin P; no fitted Drosophila value was found)
BG_GL_S = 30e-9                     # leak conductance g_L (S)
BG_A_S = 4e-9                       # subthreshold adaptation a (S)
BG_B_A = 0.0805e-9                  # spike-triggered adaptation b (A)
ADAPT_A = BG_A_S / BG_GL_S          # a / g_L, dimensionless (0.1333)
ADAPT_B_V = BG_B_A / BG_GL_S        # b / g_L (V) (2.683 mV)
ADAPT_TAU_W_S = 0.144               # tau_w (s)

# cond: reversal potentials
COND_E_EXC_V = 0.0                  # nicotinic (cation) reversal, owner's value
COND_E_INH_V = -0.074               # Cl- reversal, adult Drosophila flight MN5 (Ryglewski et al. 2017 Neuron 93:632, Fig. 4E)
# peak single-synapse conductances (relative to leak), from calibrate_dg() with the pinned Shiu parameters (origin F;
# fitted to: peak PSP of one synapse at rest = Shiu peak PSP of one synapse). tests/test_model_variants.py recomputes them.
COND_DG_EXC = 0.0052912003
COND_DG_INH = 0.0125153142

# gap: postsynaptic jump per presynaptic spike = the model's own suprathreshold kick (w_syn * f_poi, 68.75 mV)
GAP_DV_V = 0.06875

ARMS = {"R": (), "A": ("adapt",), "AC": ("adapt", "cond"), "ACG": ("adapt", "cond", "gap")}
SWITCHES = ("adapt", "cond", "gap")


@dataclass(frozen=True)
class ModelVariant:
    """A set of v2 switches. ``ModelVariant.arm("ACG")``. The reference arm is ``None`` (not a ModelVariant)."""
    adapt: bool = False
    cond: bool = False
    gap: bool = False
    gap_pairs: tuple = ()            # ((pre engine index, post engine index), ...), filled by the caller

    @classmethod
    def arm(cls, name: str, gap_pairs=()):
        if name not in ARMS:
            raise KeyError(name)
        if name == "R":
            raise ValueError("R is the reference arm: pass model_variant=None")
        sw = ARMS[name]
        return cls(adapt="adapt" in sw, cond="cond" in sw, gap="gap" in sw,
                   gap_pairs=tuple((int(a), int(b)) for a, b in gap_pairs) if "gap" in sw else ())

    @property
    def name(self) -> str:
        for k, sw in ARMS.items():
            if sw and set(sw) == {s for s in SWITCHES if getattr(self, s)}:
                return k
        return "+".join(s for s in SWITCHES if getattr(self, s)) or "none"

    def record(self) -> dict:
        return dict(name=self.name, adapt=self.adapt, cond=self.cond, gap=self.gap, n_gap_pairs=len(self.gap_pairs),
                    params=dict(adapt_a=ADAPT_A, adapt_b_v=ADAPT_B_V, adapt_tau_w_s=ADAPT_TAU_W_S,
                                e_exc_v=COND_E_EXC_V, e_inh_v=COND_E_INH_V, dg_exc=COND_DG_EXC, dg_inh=COND_DG_INH,
                                gap_dv_v=GAP_DV_V))


def equations(variant: ModelVariant, params: dict):
    """(eqs, reset, method) of a variant, in the units of the Shiu equation. The variant constants and v_0, t_mbr, tau
    are written into the equations as numbers: with symbolic constants Brian2's exact ``linear`` solver (a symbolic
    matrix exponential of the coupled v / w_ad system) does not finish."""
    v0 = f"({float(params['v_0'])!r} * volt)"
    tm = f"({float(params['t_mbr'])!r} * second)"
    ts = f"({float(params['tau'])!r} * second)"
    syn = (f"g_e * ({COND_E_EXC_V!r} * volt - v) + g_i * ({COND_E_INH_V!r} * volt - v)" if variant.cond else "g")
    ad = " - w_ad" if variant.adapt else ""
    lines = [f"dv/dt = ({v0} - v + {syn}{ad}) / {tm} : volt (unless refractory)"]
    if variant.cond:
        lines += [f"dg_e/dt = -g_e / {ts} : 1 (unless refractory)", f"dg_i/dt = -g_i / {ts} : 1 (unless refractory)"]
    else:
        lines += [f"dg/dt = -g / {ts} : volt (unless refractory)"]
    if variant.adapt:
        lines += [f"dw_ad/dt = ({ADAPT_A!r} * (v - {v0}) - w_ad) / ({ADAPT_TAU_W_S!r} * second) : volt"]
    lines += ["rfc : second"]
    reset = ("v = v_rst; " + ("g_e = 0; g_i = 0" if variant.cond else "g = 0 * mV")
             + (f"; w_ad += {ADAPT_B_V!r} * volt" if variant.adapt else ""))
    method = "exponential_euler" if variant.cond else "linear"
    return "\n".join(lines) + "\n", reset, method


# ── conductance calibration (no network run) ─────────────────────────────────────────────────────────────────
def _peak(rhs, t_end=0.1, dt=1e-6):
    """Peak |v - v_0| of a single PSP: RK4 on (x = v - v_0, s = synaptic variable), s(0) = 1, x(0) = 0."""
    x, s, best = 0.0, 1.0, 0.0
    for _ in range(int(round(t_end / dt))):
        k1 = rhs(x, s)
        k2 = rhs(x + 0.5 * dt * k1[0], s + 0.5 * dt * k1[1])
        k3 = rhs(x + 0.5 * dt * k2[0], s + 0.5 * dt * k2[1])
        k4 = rhs(x + dt * k3[0], s + dt * k3[1])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        s += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        best = max(best, abs(x))
    return best


def shiu_psp_peak(p: dict) -> float:
    """Peak PSP (V) of one Shiu synapse at rest: g jumps by w_syn and decays with tau; t_mbr dv/dt = v_0 - v + g."""
    w, tm, ts = float(p["w_syn"]), float(p["t_mbr"]), float(p["tau"])
    return _peak(lambda x, s: ((-x + w * s) / tm, -s / ts))


def cond_psp_peak(dg: float, e_rev: float, p: dict) -> float:
    """Peak |PSP| (V) of one conductance synapse at rest: g jumps by dg, t_mbr dv/dt = v_0 - v + g (E - v)."""
    v0, tm, ts = float(p["v_0"]), float(p["t_mbr"]), float(p["tau"])
    return _peak(lambda x, s: ((-x + dg * s * (e_rev - (v0 + x))) / tm, -s / ts))


def calibrate_dg(p: dict, e_rev: float, tol=1e-10) -> float:
    """dg such that the peak single-synapse PSP at rest equals the Shiu one (bisection; monotone in dg)."""
    target = shiu_psp_peak(p)
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cond_psp_peak(mid, e_rev, p) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── gap junction pairs (configs/gap_junctions.yaml) ─────────────────────────────────────────────────────────────
def load_gap_config(path=GAP_CONFIG) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def gap_pairs(cx, config=None) -> list:
    """Engine-index pairs (pre, post) of the registered ``used`` junctions on connectome ``cx``.

    Pairing rule (registered in the yaml): a presynaptic cell of type ``pre`` is coupled to every cell of type ``post``
    on which it makes at least ``min_chemical_synapses`` chemical synapses in the connectome (mixed electrochemical
    synapse; the chemical contact marks the junction site). A dataset without the types gives no pair."""
    cfg = config or load_gap_config()
    n = cx.neurons
    ct = n["cell_type"].astype(str).to_numpy()
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy(np.int64)
    out = []
    for j in cfg["used"]:
        k = int(cfg["pairing"]["min_chemical_synapses"])
        for a in np.flatnonzero(ct == j["pre"]):
            for b in np.flatnonzero(ct == j["post"]):
                if int(w[(pre == a) & (post == b)].sum()) >= k:
                    out.append((int(a), int(b)))
    return sorted(set(out))
