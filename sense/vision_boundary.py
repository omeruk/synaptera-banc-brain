"""FlyVis -> connectome boundary layer (v1 --vision-boundary, SPEC_SENSORY_INPUTS 2.1, Stage B).

Ported from synaptera@c94796d:flight/vision_boundary.py. Which neurons are driven (fixed before any behaviour):
a FlyVis-matched type is driven when >= boundary_frac of its outgoing synapses go to neurons that are NOT of a
FlyVis-matched type. Each driven neuron gets a FlyVis column from the column assignment, else the nearest column
to the synapse-weighted mean column of its columnar partners (validated), else its arbor centroid on a surface
fitted on a reference type. Graded activity -> Poisson rate:
    r = clip(R_MAX relu(a - a0[node]) / a_ref[FlyVis type], 0, R_CLIP)       (a_ref < a_ref_min: 0 Hz)
Display layer (not an input): FlyVis activity a - a0 of the NOT driven types, streamed to the record.

Changes: dataset-agnostic (a Connectome, the column table and a geometry dict are arguments; the FlyVis -> type
name map is a parameter, default the FlyWire one); the hard-coded ~/Downloads path is gone (review L4); the
runtime class takes the interval / sub-frame count from the configuration (H4) and shares the L1 / empty-frame
fixes of :class:`sense.vision_flyvis.FlyVisEyes`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from configs import load
from connectome.schema import Connectome
from sense import vision_flyvis as V

CFG = load("vision_boundary")
BOUNDARY_FRAC = CFG["boundary_frac"]
A_REF_MIN = CFG["a_ref_min"]
R_CENTRAL = CFG["r_central"]
CENTROID_REF = {k: tuple(v) for k, v in CFG["centroid_ref"].items()}
SURFACE_DEG = CFG["surface_deg"]
STEP_ON, STEP_OFF = CFG["step_on"], CFG["step_off"]

# FlyVis type -> FlyWire primary_type(s) (Codex v783). Not in v783: Mi3, Mi11, Mi12, Tm5Y, Tm28, Tm30, TmY13,
# TmY18. R1..R6 = R1-6, CT1(Lo1)/CT1(M10) = compartments of CT1, Am = Am1.
FLYVIS_TO_FLYWIRE = {**{f"R{k}": ("R1-6",) for k in range(1, 7)}, "CT1(Lo1)": ("CT1",), "CT1(M10)": ("CT1",),
                     "TmY9": ("TmY9q", "TmY9q__perp"), "Am": ("Am1",)}


def flywire_names(fv_type, name_map=None):
    return (FLYVIS_TO_FLYWIRE if name_map is None else name_map).get(fv_type, (fv_type,))


def flywire_to_flyvis(fv_types, present, name_map=None):
    """{dataset type: FlyVis type}; one FlyVis type per dataset type (R1-6 -> R1, CT1 -> CT1(Lo1))."""
    out = {}
    for t in fv_types:
        for n in flywire_names(t, name_map):
            if n in present and n not in out:
                out[n] = t
    return out


def hex_dist_flywire(dp, dq):
    """Hex distance in FlyWire (p, q) (neighbours +-(1,0), +-(0,1), +-(1,1))."""
    dp, dq = np.asarray(dp), np.asarray(dq)
    return np.where(dp * dq >= 0, np.maximum(np.abs(dp), np.abs(dq)), np.abs(dp) + np.abs(dq))


def hex_radius_flyvis(u, v):
    u, v = np.asarray(u), np.asarray(v)
    return (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2


# ── selection (scripts/make_vision_boundary.py) ─────────────────────────────
def load_codex_types(ids, path) -> np.ndarray:
    """Codex ``consolidated_cell_types.csv.gz`` primary_type per neuron id (path required, review L4)."""
    ct = pd.read_csv(path).set_index("root_id")["primary_type"]
    return ct.reindex(ids).to_numpy(dtype=object)


def boundary_fractions(cell_type, fv_types, cx: Connectome, name_map=None):
    """Per FlyVis-matched type: n neurons, outgoing synapses, share going to neurons that are not of a matched
    type. cell_type: per engine index (object array)."""
    cell_type = np.asarray(cell_type, dtype=object)
    present = set(pd.Series(cell_type).dropna().unique())
    matched = flywire_to_flyvis(fv_types, present, name_map)
    is_fv = pd.Series(cell_type).isin(list(matched)).to_numpy()
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy().astype(float)
    m = is_fv[pre]
    df = pd.DataFrame(dict(t=cell_type[pre[m]], out=~is_fv[post[m]], w=w[m]))
    tot = df.groupby("t")["w"].sum()
    outw = df[df["out"]].groupby("t")["w"].sum().reindex(tot.index).fillna(0.0)
    n = pd.Series(cell_type[is_fv]).value_counts()
    res = pd.DataFrame(dict(n=n.reindex(tot.index).astype(int), syn_out=tot, frac_out_nonflyvis=outw / tot))
    res["flyvis_type"] = [matched[t] for t in res.index]
    res["driven"] = res["frac_out_nonflyvis"] >= BOUNDARY_FRAC
    missing = [t for t in fv_types if not any(n_ in present for n_ in flywire_names(t, name_map))]
    return res.sort_values("frac_out_nonflyvis", ascending=False), missing


def _poly(pq, deg=SURFACE_DEG):
    p, q = pq[:, 0].astype(float), pq[:, 1].astype(float)
    return np.stack([p ** i * q ** j for i in range(deg + 1) for j in range(deg + 1 - i)], axis=1)


class ColumnSurfaces:
    """Per hemisphere x neuropil: cubic (p, q) -> arbor centroid fitted on the reference type."""

    def __init__(self, col: pd.DataFrame, xyz: np.ndarray, index):
        self.fit = {}
        for hemi in ("left", "right"):
            for npl, types in CENTROID_REF.items():
                c = col[(col["hemisphere"] == hemi) & col["type"].isin(types) & col["id"].isin(index.ids)]
                X = xyz[index.index_of(c["id"].to_numpy())].astype(float)
                pq = c[["p", "q"]].to_numpy()
                coef, *_ = np.linalg.lstsq(_poly(pq), X, rcond=None)
                cols = np.unique(pq, axis=0)
                pred = _poly(cols) @ coef
                resid = np.linalg.norm(_poly(pq) @ coef - X, axis=1)
                self.fit[(hemi, npl)] = dict(cols=cols, pred=pred, resid_med_um=float(np.median(resid)) / 1e3,
                                             n_ref=len(c))

    def assign(self, hemi, npl, xyz):
        """Nearest reference column; npl None -> nearest over the three neuropils. (p, q, distance um, npl)."""
        cand = [npl] if npl in CENTROID_REF else list(CENTROID_REF)
        best = None
        for n in cand:
            f = self.fit[(hemi, n)]
            d = np.linalg.norm(f["pred"] - xyz[None, :], axis=1)
            j = int(d.argmin())
            if best is None or d[j] < best[2]:
                best = (int(f["cols"][j, 0]), int(f["cols"][j, 1]), float(d[j]) / 1e3, n)
        return best


def partner_columns(cx: Connectome, idx, col: pd.DataFrame):
    """For the neurons idx: nearest column (same hemisphere) to the synapse-weighted mean column (cartesian) of
    their columnar partners, pre and post. Returns ({i: (p, q)}, {i: summed partner synapses}, {i: hemisphere});
    hemisphere = column assignment, else the neuron's side."""
    n = cx.n_neurons
    ci = cx.index.index_of(col["id"].to_numpy(), missing="-1")
    keep = ci >= 0
    col = col[keep]
    ci = ci[keep]
    h = np.full(n, -1)
    h[ci] = (col["hemisphere"] == "right").to_numpy(int)
    side = cx.neurons["side"].map({"left": 0, "right": 1}).fillna(-1).to_numpy(int)
    side[ci] = h[ci]
    if (side[idx] < 0).any():
        raise ValueError("driven neuron without side")
    xy = np.full((n, 2), np.nan)
    xy[ci] = V.flywire_cart(col["p"].to_numpy(), col["q"].to_numpy())
    ok = h >= 0
    isb = np.zeros(n, bool)
    isb[idx] = True
    pre, post = cx.edge_endpoints()
    w = cx.edges["n_syn"].to_numpy().astype(float)
    acc, ws = np.zeros((n, 2)), np.zeros(n)
    for a, b in ((pre, post), (post, pre)):
        m = isb[a] & ok[b] & (h[b] == side[a])
        np.add.at(acc, a[m], xy[b[m]] * w[m, None])
        np.add.at(ws, a[m], w[m])
    out = {}
    for hh, name in ((0, "left"), (1, "right")):
        cols = np.unique(col.loc[col["hemisphere"] == name, ["p", "q"]].to_numpy(), axis=0)
        cxy = V.flywire_cart(cols[:, 0], cols[:, 1])
        sel = idx[(side[idx] == hh) & (ws[idx] > 0)]
        mxy = acc[sel] / ws[sel, None]
        j = ((mxy[:, None, :] - cxy[None]) ** 2).sum(-1).argmin(1)
        out.update({int(i): (int(cols[k, 0]), int(cols[k, 1])) for i, k in zip(sel, j)})
    return out, ws, {int(i): ("left", "right")[side[i]] for i in idx}


def build_table(cx: Connectome, cell_type, driven_types: dict, columns: pd.DataFrame, xyz: np.ndarray,
                dominant_neuropil):
    """One row per driven neuron: id, fw_type, flyvis_type, hemisphere, p, q, col_source, centroid_dist_um,
    u, v, extent_dist. Also returns the validation of the centroid rule on the columnar neurons of the driven
    non-reference types. ``xyz``: arbor centroid per engine index (nm); ``dominant_neuropil``: per engine index."""
    cell_type = np.asarray(cell_type, dtype=object)
    dom = np.asarray(dominant_neuropil, dtype=object)
    fw2fv = dict(driven_types)
    idx = np.flatnonzero(pd.Series(cell_type).isin(list(fw2fv)).to_numpy())
    col = columns[columns["id"].isin(cx.index.ids)]
    cc = col.set_index("id")
    surf = ColumnSurfaces(col, xyz, cx.index)
    pm_cols, pm_w, pm_side = partner_columns(cx, idx, col)
    ids = cx.index.ids

    rows = []
    for i in idx:
        r = int(ids[i])
        t = cell_type[i]
        if r in cc.index:
            hemi, p, q = cc.at[r, "hemisphere"], int(cc.at[r, "p"]), int(cc.at[r, "q"])
            src, dist = "column_assignment", 0.0
        elif pm_w[i] > 0:
            hemi, (p, q), dist = pm_side[i], pm_cols[i], 0.0
            src = "partner_mean"
        else:
            hemi = pm_side[i]
            p, q, dist, npl = surf.assign(hemi, dom[i] if dom[i] in CENTROID_REF else None, xyz[i])
            src = f"centroid_{npl}"
        rows.append(dict(id=r, fw_type=t, flyvis_type=fw2fv[t], hemisphere=hemi, p=p, q=q, col_source=src,
                         centroid_dist_um=dist))
    tab = pd.DataFrame(rows)
    u, v, d = V.map_columns(tab["p"].to_numpy(), tab["q"].to_numpy())
    tab["u"], tab["v"], tab["extent_dist"] = u, v, d

    ref_types = {t for ts in CENTROID_REF.values() for t in ts}
    val = []
    for i in idx:
        r = int(ids[i])
        if r not in cc.index or cell_type[i] in ref_types:
            continue
        hemi, p0, q0 = cc.at[r, "hemisphere"], cc.at[r, "p"], cc.at[r, "q"]
        p, q, _, _ = surf.assign(hemi, dom[i] if dom[i] in CENTROID_REF else None, xyz[i])
        pp, pq = pm_cols[i] if pm_w[i] > 0 else (10 ** 6, 10 ** 6)
        val.append(dict(fw_type=cell_type[i], centroid=int(hex_dist_flywire(p - p0, q - q0)),
                        partner_mean=int(hex_dist_flywire(pp - p0, pq - q0))))
    val = pd.DataFrame(val)

    def st(dd):
        return dict(exact=float((dd == 0).mean()), le1=float((dd <= 1).mean()), le2=float((dd <= 2).mean()),
                    median=float(dd.median()),
                    per_type_le1={t: float((g <= 1).mean()) for t, g in dd.groupby(val["fw_type"])})
    vstats = dict(n=len(val), centroid=st(val["centroid"]), partner_mean=st(val["partner_mean"]),
                  surfaces={f"{h}/{n}": dict(n_ref=f["n_ref"], resid_med_um=f["resid_med_um"])
                            for (h, n), f in surf.fit.items()})
    return tab, vstats


def load_table(path):
    return pd.read_csv(path)


def load_types(path):
    with open(path) as f:
        return json.load(f)


# ── transduction (scripts/make_transduction.py) ─────────────────────────────
def standard_stimuli(dt: float):
    """name -> callable(n) yielding FlyVis-ordered frames (2, 721) (both eyes, canonical frame)."""
    from flygym.examples.vision.vision_network import RetinaMapper
    rm = RetinaMapper()

    def grating(direction):
        def gen(n):
            for fr in V.grating_frames(direction, n, dt=dt):
                g = fr.max(axis=2)
                yield np.stack([rm.flygym_to_flyvis(g[0]), rm.flygym_to_flyvis(g[1])]).astype(np.float32)
        return gen

    def step(level):
        def gen(n):
            for _ in range(n):
                yield np.full((2, 721), level, np.float32)
        return gen

    st = {f"grating_{d}": grating(v) for d, v in V.PD_IMAGE.items()}
    st["on_step"] = step(STEP_ON)
    st["off_step"] = step(STEP_OFF)
    return st


def calibrate(net=None, types=None, interval_s: float = 0.025, subframes: int = V.DEFAULT_SUBFRAMES,
              t_stim=1.0, t_skip=0.25):
    """a0 of every FlyVis node (grey steady state) and a_ref per FlyVis type: the type's largest mean positive
    response to the standard stimulus set, central columns (hex radius <= R_CENTRAL), both eyes, activity
    averaged over ``subframes`` as at run time."""
    import torch
    net = net if net is not None else V.load_flyvis_net()
    dt = V.flyvis_dt(interval_s, subframes)
    nodes = net.connectome.nodes
    ntype = np.array(nodes.type[:]).astype(str)
    nu, nv = np.asarray(nodes.u[:]), np.asarray(nodes.v[:])
    central = hex_radius_flyvis(nu, nv) <= R_CENTRAL
    types = types or list(dict.fromkeys(ntype))

    def fwd(frame, state):
        with torch.no_grad():
            net.stimulus.zero(2, 1)
            net.stimulus.add_input(torch.as_tensor(frame, dtype=torch.float32).reshape(2, 1, 1, -1))
            return net.forward(net.stimulus(), dt, state=state, as_states=True)[-1]

    grey = np.full((2, 721), 0.5, np.float32)
    state = None
    for _ in range(int(round(V.GREY_S / dt))):
        state = fwd(grey, state)
    act0 = state.nodes.activity.numpy()
    assert np.allclose(act0[0], act0[1])
    a0 = act0[0].astype(float)
    state0 = state
    sel = {t: np.flatnonzero((ntype == t) & central) for t in types}
    n = int(round(t_stim / dt))
    k0 = int(round(t_skip / dt))
    resp = {}
    for name, gen in standard_stimuli(dt).items():
        state = state0
        acc = {t: [] for t in types}
        sub = 0.0
        for k, fr in enumerate(gen(n)):
            state = fwd(fr, state)
            sub = sub + state.nodes.activity.numpy()
            if (k + 1) % subframes == 0:
                if k >= k0:
                    a = sub / subframes
                    for t in types:
                        acc[t].append(float(np.maximum(a[:, sel[t]] - a0[sel[t]], 0).mean()))
                sub = 0.0
        resp[name] = {t: float(np.mean(x)) for t, x in acc.items()}
    a_ref = {t: max(resp[s][t] for s in resp) for t in types}
    best = {t: max(resp, key=lambda s: resp[s][t]) for t in types}
    return dict(a0=a0.tolist(), node_type=ntype.tolist(), a_ref=a_ref, a_ref_stimulus=best, responses=resp,
                model=V.FLYVIS_MODEL, dt=dt, grey_s=V.GREY_S, r_central=R_CENTRAL, window_s=[t_skip, t_stim],
                stimuli=dict(grating=dict(period_px=V.GRATING_PERIOD_PX, tf_hz=V.GRATING_TF_HZ,
                                          contrast=V.GRATING_CONTRAST, pd_image=V.PD_IMAGE),
                             on_step=[0.5, STEP_ON], off_step=[0.5, STEP_OFF]),
                r_max=V.R_MAX, r_clip=V.R_CLIP)


def load_transduction(path):
    with open(path) as f:
        return json.load(f)


# ── groups ──────────────────────────────────────────────────────────────────
def boundary_groups(cx: Connectome, table: pd.DataFrame) -> dict:
    """{"vbnd_L", "vbnd_R"}: sorted engine indices of the driven neurons."""
    return {f"vbnd_{h[0].upper()}": cx.index.group(table.loc[table["hemisphere"] == h, "id"].to_numpy())
            for h in ("left", "right")}


# ── run time ────────────────────────────────────────────────────────────────
def node_of(lookup, ntype, nu, nv, t, u, v):
    """FlyVis node of type t at column (u, v); types on a sparse FlyVis lattice (Lawf1/Lawf2: 123 of 721
    columns) take the type's nearest node."""
    k = lookup.get((t, int(u), int(v)))
    if k is not None:
        return k
    cand = np.flatnonzero(ntype == t)
    d = ((V.axial_to_cart(nu[cand], nv[cand]) - V.axial_to_cart([u], [v])) ** 2).sum(1)
    return int(cand[d.argmin()])


class BoundaryEyes(V.FlyVisEyes):
    """Both eyes through FlyVis; per-neuron rates for the boundary layer (vbnd_L / R order) and the
    display-layer activity of the not-driven FlyVis types. ``vbnd_ids``: {"L": ids, "R": ids} in the engine
    group order; ``table``: build_table output / v1 vision_boundary csv (``root_id`` accepted for ``id``);
    ``columns``: column assignment (for the display layer); ``index``: the connectome's IdIndex."""

    def __init__(self, table: pd.DataFrame, vbnd_ids: dict, transduction: dict, columns: pd.DataFrame, index,
                 interval_s: float = 0.025, subframes: int = V.DEFAULT_SUBFRAMES, name_map=None,
                 verbose: bool = True):
        self._init_flyvis(interval_s, subframes)
        tab = table.rename(columns={"root_id": "id"}).copy()
        nodes = self.net.connectome.nodes
        ntype = np.array(nodes.type[:]).astype(str)
        nu, nv = np.asarray(nodes.u[:]), np.asarray(nodes.v[:])
        lookup = {(t, int(a), int(b)): i for i, (t, a, b) in enumerate(zip(ntype, nu, nv))}
        self.fw_types = tuple(sorted(tab["fw_type"].unique(), key=lambda t: (not t.startswith(("T4", "T5")), t)))
        a0_all = np.asarray(transduction["a0"], float)
        assert len(a0_all) == len(ntype) and list(transduction["node_type"]) == ntype.tolist()
        self.a0_all = a0_all
        self.node, self.tcode, self.a0, self.aref, self.n_outside = {}, {}, {}, {}, {}
        self.t45_mask, self.t45_code = {}, {}
        tab_i = tab.set_index("id")
        for s, sd in (("L", "left"), ("R", "right")):
            c = tab_i[tab_i["hemisphere"] == sd].loc[np.asarray(vbnd_ids[s])]
            self.node[s] = np.array([node_of(lookup, ntype, nu, nv, t, a, b)
                                     for t, a, b in zip(c["flyvis_type"], c["u"], c["v"])])
            self.tcode[s] = np.array([self.fw_types.index(t) for t in c["fw_type"]])
            self.a0[s] = a0_all[self.node[s]]
            ar = np.array([transduction["a_ref"][t] for t in c["flyvis_type"]])
            self.aref[s] = np.where(ar < A_REF_MIN, np.inf, ar)               # unresponsive -> 0 Hz
            self.n_outside[s] = int((c["extent_dist"] > 0.5).sum())
            m = c["fw_type"].isin(V.T45_TYPES).to_numpy()
            self.t45_mask[s] = m
            self.t45_code[s] = np.array([V.T45_TYPES.index(t) for t in c["fw_type"][m]])
        self.unresponsive = tuple(sorted({t for t in tab["fw_type"]
                                          if transduction["a_ref"][tab.loc[tab["fw_type"] == t, "flyvis_type"].iloc[0]]
                                          < A_REF_MIN}))
        driven_fv = set(tab["flyvis_type"])
        self.display_types = tuple(t for t in dict.fromkeys(ntype) if t not in driven_fv)
        self.display_nodes = np.flatnonzero(np.isin(ntype, self.display_types))
        self.display_type = ntype[self.display_nodes]
        self.display_u, self.display_v = nu[self.display_nodes], nv[self.display_nodes]
        # connectome neurons drawn by the display layer: columnar neurons whose type is a display FlyVis type with
        # a unique dataset name (R1-6, CT1 skipped: several FlyVis nodes per neuron)
        col = columns[columns["id"].isin(index.ids)]
        fw2disp = {flywire_names(t, name_map)[0]: t for t in self.display_types
                   if len(flywire_names(t, name_map)) == 1 and flywire_names(t, name_map)[0] not in ("R1-6", "CT1")}
        col = col[col["type"].isin(list(fw2disp))]
        u, v, _ = V.map_columns(col["p"].to_numpy(), col["q"].to_numpy())
        pos = {n: k for k, n in enumerate(self.display_nodes)}
        self.display_fw = dict(idx=index.index_of(col["id"].to_numpy()).astype(np.int32),
                               eye=(col["hemisphere"] == "right").to_numpy(np.int8),
                               pos=np.array([pos[lookup[(fw2disp[t], int(a), int(b))]]
                                             for t, a, b in zip(col["type"], u, v)], np.int32))
        self.display = None
        if verbose:
            print(f"  FlyVis -> connectome boundary layer: {len(self.fw_types)} types, L {len(self.node['L'])}, "
                  f"R {len(self.node['R'])} neurons; outside FlyVis extent: L {self.n_outside['L']}, "
                  f"R {self.n_outside['R']}; display layer {len(self.display_types)} FlyVis types, "
                  f"{len(self.display_nodes)} nodes/eye; unresponsive (0 Hz): {self.unresponsive}")
        li = self.net.stimulus.layer_index
        self.t5a, self.t5b = li["T5a"], li["T5b"]
        self.state = None
        self.reset()

    def step(self, frames):
        acts = self._run_frames(frames)
        full = np.mean(acts, axis=0)
        a = {"L": full[0, self.node["L"]], "R": full[1, self.node["R"]]}
        self.display = (full[:, self.display_nodes] - self.a0_all[self.display_nodes]).astype(np.float16)
        return a, self.loom(acts)

    def rates(self, a):
        return {s: np.clip(V.R_MAX * np.maximum(a[s] - self.a0[s], 0.0) / self.aref[s], 0.0, V.R_CLIP)
                for s in ("L", "R")}

    def type_means(self, r):
        """(2, n_types) mean rate per eye x boundary type (self.fw_types order)."""
        return np.array([[r[s][self.tcode[s] == k].mean() for k in range(len(self.fw_types))] for s in ("L", "R")])

    def t45_view(self, r):
        """T4/T5 members only: {L, R} rates and the (2, 8) per-subtype means."""
        v = {s: r[s][self.t45_mask[s]] for s in ("L", "R")}
        tm = np.array([[v[s][self.t45_code[s] == k].mean() for k in range(len(V.T45_TYPES))] for s in ("L", "R")])
        return v, tm


class DisplayWriter:
    """Streams /flyvis/activity (steps x 2 eyes x display nodes, float16) into a run file, one chunk per interval;
    nothing accumulates in RAM. Refuses an existing file (CLAUDE.md rule 13)."""

    def __init__(self, path, eyes: BoundaryEyes):
        import os

        import h5py
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; result files are never overwritten")
        self.f = h5py.File(path, "w")
        g = self.f.create_group("flyvis")
        nd = len(eyes.display_nodes)
        self.ds = g.create_dataset("activity", shape=(0, 2, nd), maxshape=(None, 2, nd), dtype=np.float16,
                                   chunks=(1, 2, nd), compression="gzip", compression_opts=1)
        self.ds.attrs["content"] = ("FlyVis node activity a - a0 (grey steady state) of the NOT driven FlyVis types, "
                                    "interval mean; display only, never a simulation input")
        self.ds.attrs["axes"] = "step_row, eye (0 = left, 1 = right; right eye image mirrored), node"
        g.create_dataset("node_index", data=eyes.display_nodes.astype(np.int32))
        g.create_dataset("node_type", data=np.array(eyes.display_type, dtype="S16"))
        g.create_dataset("node_u", data=eyes.display_u.astype(np.int16))
        g.create_dataset("node_v", data=eyes.display_v.astype(np.int16))
        m = g.create_group("flywire_map")
        m.attrs["content"] = "neuron (engine index) -> eye and column of /flyvis/activity (display position)"
        for k, val in eyes.display_fw.items():
            m.create_dataset(k, data=val)
        self.steps = []

    def add(self, step, display):
        n = self.ds.shape[0]
        self.ds.resize(n + 1, axis=0)
        self.ds[n] = display
        self.steps.append(int(step))

    def close(self):
        self.f["flyvis"].create_dataset("step_idx", data=np.asarray(self.steps, np.int32))
        self.f["flyvis"]["step_idx"].attrs["content"] = "interval index of each row (negative = perch / input cut)"
        self.f.close()
