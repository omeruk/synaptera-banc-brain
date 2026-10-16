"""FlyVis -> connectome visual input: per-neuron Poisson rates for T4a-d / T5a-d.

Ported from synaptera@c94796d:flight/visual_input.py. Pipeline per coupling interval:
  FlyGym compound-eye frames (``subframes`` per interval, (2, 721, 2) each)
  -> FlyVis connectome-constrained network (flow/0000/000), both eyes as a batch of 2, dt = interval / subframes
  -> activity of the T4/T5 node of each FlyVis column, averaged over the sub-frames
  -> r = R_MAX relu(a - a0[type]) / a_ref[type], clipped to R_CLIP
  -> Poisson rates of the connectome T4/T5 neuron assigned to that column.

Column alignment is anatomical (no behaviour involved). FlyWire columns use hex coordinates (p, q) with
neighbours +-(1,0), +-(0,1), +-(1,1); axial (a, b) = (p, -q). FlyVis uses axial (u, v) with neighbours
+-(1,0), +-(0,1), +-(1,-1). The lattice map is the point-group operation that best aligns the
synapse-weighted input offsets of the two connectomes (fit_alignment): rotation by +60 deg, no reflection.

Eye convention (measured in v1): the FlyGym left-eye image is already in the FlyVis eye frame; the right-eye
image has anterior on the left and is mirrored horizontally before FlyVis.

Changes with respect to v1: the column table and the neuron order come from the caller (dataset-agnostic; the
FlyWire column assignment is loaded by :func:`load_columns` only when vision is on, review M2); the number of
sub-frames and the interval come from the configuration (H4); the loom summary is the sub-frame mean, not the
last sub-frame, and an empty frame list is an error (L1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from configs import load

CFG = load("vision")
FLYVIS_MODEL = CFG["flyvis_model"]
FLYVIS_EXTENT = CFG["flyvis_extent"]
ALIGN_ROT = CFG["align_rot"]
ALIGN_REFLECT = CFG["align_reflect"]
R_EYE_FLIP = CFG["r_eye_flip"]
GREY_S = CFG["grey_s"]
R_MAX, R_CLIP = CFG["r_max"], CFG["r_clip"]
GRATING_PERIOD_PX, GRATING_TF_HZ, GRATING_CONTRAST = CFG["grating_period_px"], CFG["grating_tf_hz"], CFG["grating_contrast"]
DEFAULT_SUBFRAMES = CFG["subframes"]
T45_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
# Preferred direction of each subtype as a FlyGym image direction (x right, y down) in the FlyVis eye frame
PD_IMAGE = {"a": (-1, 0), "b": (1, 0), "c": (0, -1), "d": (0, 1)}

# FlyVis always runs on the CPU: every existing FlyVis record was made on a CPU-only machine, and FlyVis 1.2.0 breaks
# on CUDA (its initialisation draws from a CPU generator while it has made CUDA torch's default device).
FLYVIS_DEVICE = "cpu"

E1 = np.array([1.0, 0.0])
E2 = np.array([0.5, np.sqrt(3) / 2])


def import_flyvis():
    """Import FlyVis pinned to FLYVIS_DEVICE and return the module. Call before any flyvis or flygym vision import.

    `import flyvis` picks CUDA when available, makes it torch's process-wide default device and binds it as the default
    argument of its rendering/dataset functions, so the choice has to be made during the first import: CUDA is hidden
    from that import only. Raises if flyvis was already imported elsewhere on another device."""
    import torch
    if "flyvis" not in sys.modules:
        is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            import flyvis  # noqa: F401
        finally:
            torch.cuda.is_available = is_available
    import flyvis
    if flyvis.device.type != FLYVIS_DEVICE or torch.get_default_device().type != FLYVIS_DEVICE:
        raise RuntimeError(f"flyvis was imported on {flyvis.device} (torch default {torch.get_default_device()}); "
                           f"import it through sense.vision_flyvis.import_flyvis() first")
    return flyvis


def flyvis_dt(interval_s: float, subframes: int = DEFAULT_SUBFRAMES) -> float:
    return interval_s / subframes


def axial_to_cart(a, b):
    return np.outer(np.asarray(a, float), E1) + np.outer(np.asarray(b, float), E2)


def flywire_cart(p, q):
    return axial_to_cart(p, -np.asarray(q))


def lattice_op(rot, reflect):
    th = rot * np.pi / 3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return R @ (np.diag([1.0, -1.0]) if reflect else np.eye(2))


def flyvis_hex(extent=FLYVIS_EXTENT):
    import_flyvis()
    from flyvis.utils.hex_utils import get_hex_coords
    return get_hex_coords(extent)


def map_columns(p, q, rot=ALIGN_ROT, reflect=ALIGN_REFLECT, extent=FLYVIS_EXTENT):
    """FlyWire (p, q) -> nearest FlyVis (u, v). Returns u, v and the distance in column spacings (0 inside the
    FlyVis extent)."""
    xy = flywire_cart(p, q) @ lattice_op(rot, reflect).T
    u, v = flyvis_hex(extent)
    fv = axial_to_cart(u, v)
    d2 = ((xy[:, None, :] - fv[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    return u[j], v[j], np.sqrt(d2[np.arange(len(j)), j])


def load_columns(path, id_column: str = "root_id") -> pd.DataFrame:
    """Column assignment table (FlyWire Codex ``column_assignment.csv.gz``) as id, hemisphere, type, p, q.
    Loaded only by runs that use vision."""
    col = pd.read_csv(path)
    col = col.rename(columns={id_column: "id"})
    return col[["id", "hemisphere", "type", "p", "q"]]


def t45_columns(columns: pd.DataFrame) -> pd.DataFrame:
    return columns[columns["type"].isin(T45_TYPES)].reset_index(drop=True)


def fit_alignment(cx, columns: pd.DataFrame, flyvis_edges=None):
    """Score every hex point-group operation FlyWire -> FlyVis per hemisphere.

    For each (T4/T5 subtype, columnar input type) pair: synapse-weighted mean column offset input - target in both
    connectomes (FlyVis: edges onto the central column). Score = weighted mean dot product after the operation.
    ``cx``: a Connectome (edges pre, post, n_syn). Returns {hemisphere: [(score, rot, reflect), ...] best first}
    and the per-pair vectors of the best operation."""
    col = columns.set_index("id")
    e = cx.edges
    e = e[e["pre"].isin(col.index) & e["post"].isin(col.index)]
    pre, post = col.loc[e["pre"]], col.loc[e["post"]]
    t = pd.DataFrame(dict(src=pre["type"].to_numpy(), tgt=post["type"].to_numpy(),
                          hemi=post["hemisphere"].to_numpy(),
                          same=pre["hemisphere"].to_numpy() == post["hemisphere"].to_numpy(),
                          dp=pre["p"].to_numpy() - post["p"].to_numpy(), dq=pre["q"].to_numpy() - post["q"].to_numpy(),
                          w=e["n_syn"].to_numpy()))
    t = t[t["same"] & t["tgt"].isin(T45_TYPES)]
    if flyvis_edges is None:
        flyvis_edges = load_flyvis_net().connectome.edges
    E = flyvis_edges
    fv = pd.DataFrame(dict(src=E.source_type[:].astype(str), tgt=E.target_type[:].astype(str), du=E.du[:],
                           dv=E.dv[:], w=E.n_syn[:], tu=E.target_u[:], tv=E.target_v[:]))
    fv = fv[(fv["tu"] == 0) & (fv["tv"] == 0) & fv["tgt"].isin(T45_TYPES)]
    ref = {}
    for (tg, s), g in fv.groupby(["tgt", "src"]):
        ref[(tg, s)] = ((axial_to_cart(g["du"], g["dv"]) * g["w"].to_numpy()[:, None]).sum(0) / g["w"].sum(),
                        float(g["w"].sum()))
    out, vecs = {}, {}
    for h in ("left", "right"):
        fw = {}
        for (tg, s), g in t[t["hemi"] == h].groupby(["tgt", "src"]):
            fw[(tg, s)] = (flywire_cart(g["dp"], g["dq"]) * g["w"].to_numpy()[:, None]).sum(0) / g["w"].sum()
        res = []
        for rot in range(6):
            for refl in (False, True):
                M = lattice_op(rot, refl)
                keys = [k for k in ref if k in fw]
                sc = sum(ref[k][1] * float((M @ fw[k]) @ ref[k][0]) for k in keys) / sum(ref[k][1] for k in keys)
                res.append((sc, rot, refl))
        res.sort(reverse=True)
        out[h] = res
        M = lattice_op(res[0][1], res[0][2])
        vecs[h] = {k: (M @ fw[k], ref[k][0]) for k in ref if k in fw}
    return out, vecs


def load_flyvis_net():
    flyvis = import_flyvis()
    from flyvis import NetworkView
    net = NetworkView(flyvis.results_dir / FLYVIS_MODEL).init_network(checkpoint="best")
    net.eval()
    for prm in net.parameters():
        prm.requires_grad = False
    return net


def load_transduction(path):
    with open(path) as f:
        return json.load(f)


class FlyVisEyes:
    """Both eyes through FlyVis; per-neuron rates for the T4/T5 input groups.

    columns: id, hemisphere, type, p, q of the dataset's columnar neurons. group_ids: {"L": ids, "R": ids} in the
    order of the engine input groups; the rate arrays returned by :meth:`rates` follow that order."""

    def _init_flyvis(self, interval_s, subframes):
        import torch  # noqa: F401
        import_flyvis()
        from flygym.examples.vision.vision_network import RetinaMapper
        from flyvis.datasets.rendering import BoxEye

        self.subframes = int(subframes)
        self.dt = flyvis_dt(interval_s, self.subframes)
        self.net = load_flyvis_net()
        self.rm = RetinaMapper()
        rc = BoxEye(extent=FLYVIS_EXTENT).receptor_centers.cpu().numpy()
        mirrored = rc * np.array([1.0, -1.0])          # negate the horizontal (column) coordinate
        self.flip_h = ((mirrored[:, None, :] - rc[None, :, :]) ** 2).sum(-1).argmin(1)
        assert len(np.unique(self.flip_h)) == len(rc)

    def __init__(self, columns: pd.DataFrame, group_ids: dict, transduction: dict, interval_s: float = 0.025,
                 subframes: int = DEFAULT_SUBFRAMES, verbose: bool = True):
        self._init_flyvis(interval_s, subframes)
        nodes = self.net.connectome.nodes
        ntype = np.array(nodes.type[:]).astype(str)
        nu, nv = np.asarray(nodes.u[:]), np.asarray(nodes.v[:])
        lookup = {(t, int(a), int(b)): i for i, (t, a, b) in enumerate(zip(ntype, nu, nv)) if t in T45_TYPES}

        col = t45_columns(columns).copy()
        u, v, dist = map_columns(col["p"].to_numpy(), col["q"].to_numpy())
        col["node"] = [lookup[(t, int(a), int(b))] for t, a, b in zip(col["type"], u, v)]
        col["dist"] = dist
        col = col.set_index("id")
        self.node, self.type_code, self.n_outside = {}, {}, {}
        for s, sd in (("L", "left"), ("R", "right")):
            c = col[col["hemisphere"] == sd].loc[np.asarray(group_ids[s])]
            self.node[s] = c["node"].to_numpy()
            self.type_code[s] = np.array([T45_TYPES.index(t) for t in c["type"]])
            self.n_outside[s] = int((c["dist"] > 0.5).sum())
        if verbose:
            print(f"  FlyVis->connectome T4/T5: L {len(self.node['L'])}, R {len(self.node['R'])} neurons; "
                  f"outside FlyVis extent (nearest column): L {self.n_outside['L']}, R {self.n_outside['R']}")
        li = self.net.stimulus.layer_index
        self.t5a, self.t5b = li["T5a"], li["T5b"]
        a0_node = np.full(len(ntype), np.nan)
        for t in T45_TYPES:
            a0_node[li[t]] = transduction["a0"][t]
        self.a0 = {s: a0_node[self.node[s]] for s in ("L", "R")}
        self.a_ref = np.array([transduction["a_ref"][t] for t in T45_TYPES])
        self.state = None
        self.reset()

    # ── FlyVis ──────────────────────────────────────────────────────────────
    def _forward(self, frame):
        """frame: (2, 721) FlyVis-ordered intensities. Returns activity (2, n_nodes) numpy."""
        import torch
        with torch.no_grad():
            self.net.stimulus.zero(2, 1)
            self.net.stimulus.add_input(torch.as_tensor(frame, dtype=torch.float32).reshape(2, 1, 1, -1))
            self.state = self.net.forward(self.net.stimulus(), self.dt, state=self.state, as_states=True)[-1]
        return self.state.nodes.activity.numpy()

    def reset(self):
        self.state = None
        grey = np.full((2, 721), 0.5, np.float32)
        for _ in range(int(round(GREY_S / self.dt))):
            self._forward(grey)

    def to_flyvis(self, vision, flip_right=R_EYE_FLIP):
        """FlyGym ommatidia readout (2, 721, 2) -> FlyVis-ordered (2, 721), right eye mirrored."""
        g = vision.max(axis=2)
        L = self.rm.flygym_to_flyvis(g[0])
        R = self.rm.flygym_to_flyvis(g[1])
        if flip_right:
            R = R[self.flip_h]
        return np.stack([L, R]).astype(np.float32)

    def _run_frames(self, frames):
        """Activities (2, n_nodes) of every frame; an empty list is an error (L1)."""
        if len(frames) == 0:
            raise ValueError("no vision frames rendered in this interval")
        return [self._forward(self.to_flyvis(fr)) for fr in frames]

    def loom(self, acts):
        """T5a/b mean absolute activity per eye, averaged over the sub-frames (L1: not the last sub-frame)."""
        per = np.array([[np.abs(a[e, self.t5a]).mean() + np.abs(a[e, self.t5b]).mean() for e in (0, 1)]
                        for a in acts])
        return tuple(float(x) for x in per.mean(axis=0))

    def step(self, frames):
        """frames: FlyGym vision readouts rendered during the last interval. Returns (activity per T4/T5
        neuron {L, R}, loom (L, R))."""
        acts = self._run_frames(frames)
        mean = np.mean(acts, axis=0)
        a = {"L": mean[0, self.node["L"]], "R": mean[1, self.node["R"]]}
        return a, self.loom(acts)

    def rates(self, a):
        """Activity per neuron -> Poisson rate (Hz) per neuron."""
        out = {}
        for s in ("L", "R"):
            tc = self.type_code[s]
            out[s] = np.clip(R_MAX * np.maximum(a[s] - self.a0[s], 0.0) / self.a_ref[tc], 0.0, R_CLIP)
        return out

    def type_means(self, r):
        """(2, 8) mean rate per eye x subtype (T4a..T5d) for recording."""
        return np.array([[r[s][self.type_code[s] == k].mean() for k in range(len(T45_TYPES))] for s in ("L", "R")])


# ── transduction calibration (scripts/make_transduction.py) ─────────────────
def grating_frames(direction, n, period=GRATING_PERIOD_PX, tf=GRATING_TF_HZ, contrast=GRATING_CONTRAST, dt=0.0125):
    """Sinusoidal grating in the FlyGym image moving along ``direction`` (x right, y down); yields FlyGym-format
    readouts (2, 721, 2) (same image, both eyes)."""
    from flygym.vision import Retina
    ret = Retina()
    yy, xx = np.mgrid[0:512, 0:450].astype(float)
    dx, dy = direction
    for k in range(n):
        ph = 2 * np.pi * ((xx * dx + yy * dy) / period - tf * dt * k)
        img = np.clip(0.5 + 0.5 * contrast * np.sin(ph), 0, 1)
        img = (np.repeat(img[..., None], 3, axis=2) * 255).astype(np.uint8)
        hx = ret.raw_image_to_hex_pxls(img)
        yield np.stack([hx, hx])


def calibrate_transduction(net=None, interval_s: float = 0.025, subframes: int = DEFAULT_SUBFRAMES,
                           t_grating=1.0, t_skip=0.25):
    """a0: grey steady-state activity of every T4/T5 FlyVis node (list per subtype in layer_index order); a_ref:
    time- and column-mean of relu(a - a0) per subtype for the standard grating moving in the subtype's
    preferred direction, with the activity averaged over ``subframes`` frames per interval as at run time
    (canonical eye frame: no right-eye flip, both batch entries)."""
    import torch
    import_flyvis()
    from flygym.examples.vision.vision_network import RetinaMapper
    dt = flyvis_dt(interval_s, subframes)
    net = net if net is not None else load_flyvis_net()
    li = net.stimulus.layer_index

    def fwd(frame, state):
        with torch.no_grad():
            net.stimulus.zero(2, 1)
            net.stimulus.add_input(torch.as_tensor(frame, dtype=torch.float32).reshape(2, 1, 1, -1))
            return net.forward(net.stimulus(), dt, state=state, as_states=True)[-1]

    rm = RetinaMapper()
    grey = np.full((2, 721), 0.5, np.float32)
    state = None
    for _ in range(int(round(GREY_S / dt))):
        state = fwd(grey, state)
    act0 = state.nodes.activity.numpy()
    assert np.allclose(act0[0], act0[1])
    a0 = {t: act0[0, li[t]].astype(float) for t in T45_TYPES}
    state0 = state
    a_ref = {}
    n = int(round(t_grating / dt))
    k0 = int(round(t_skip / dt))
    for d, direction in PD_IMAGE.items():
        state = state0
        acc = {t: [] for t in T45_TYPES if t.endswith(d)}
        sub = 0.0
        for k, fr in enumerate(grating_frames(direction, n, dt=dt)):
            g = fr.max(axis=2)
            state = fwd(np.stack([rm.flygym_to_flyvis(g[0]), rm.flygym_to_flyvis(g[1])]), state)
            sub = sub + state.nodes.activity.numpy()
            if (k + 1) % subframes == 0:
                if k >= k0:
                    for t in acc:
                        acc[t].append(np.maximum(sub[:, li[t]] / subframes - a0[t], 0).mean())
                sub = 0.0
        for t, xs in acc.items():
            a_ref[t] = float(np.mean(xs))
    return dict(a0={t: v.tolist() for t, v in a0.items()}, a_ref=a_ref, model=FLYVIS_MODEL, dt=dt, grey_s=GREY_S,
                grating=dict(period_px=GRATING_PERIOD_PX, tf_hz=GRATING_TF_HZ, contrast=GRATING_CONTRAST,
                             t=t_grating, t_skip=t_skip, pd_image=PD_IMAGE),
                r_max=R_MAX, r_clip=R_CLIP)


def compare_transduction(new: dict, ref: dict, rtol: float = 1e-4, atol: float = 1e-6) -> list:
    """Numeric comparison of two transduction dicts (review M7: bit-level hashes of these files differ between
    platforms). Returns the list of mismatching keys (empty = equal within tolerance)."""
    bad = []

    def walk(a, b, path):
        if isinstance(a, dict):
            for k in a:
                if k not in b:
                    bad.append(f"{path}/{k}: missing")
                else:
                    walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, (list, tuple)) and a and isinstance(a[0], (int, float)):
            if len(a) != len(b) or not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                bad.append(path)
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                bad.append(path)
        elif a != b:
            bad.append(path)
    walk(new, ref, "")
    return bad


# The open-loop stimulus protocol (tasks/t0_stimuli.py) is defined at the v1 timing: 25 ms steps, 2 sub-frames.
VIS_SUBFRAMES = DEFAULT_SUBFRAMES
FLYVIS_DT = flyvis_dt(0.025, DEFAULT_SUBFRAMES)
