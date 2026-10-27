"""Visualisation scaffold (not an experiment, no claim): neuron positions of the FlyWire brain model and the BANC VNC
subgraph in nm, DN landmarks, and a FlyWire -> BANC transform. Origin class E (engineering scaffold, viewing only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_positions.py [--overwrite]

Outputs (runs/viz/, not committed; an existing output is not overwritten without --overwrite):
- positions_brain.npz: the 138,639 brain-model neurons in Completeness order; xyz_nm, source = soma (FlyWire soma,
  voxels x (4, 4, 40) nm) where present, else centroid_synapses / centroid_fallback (data/neuron_arbor_centroids.npz).
- positions_vnc.npz: the 25,818 stage-A VNC nodes in subgraph order; xyz_nm, source = root_position (BANC
  root_position_nm) or filled (synapse-weighted mean position of the subgraph partners that have a root position,
  one pass); a neuron with no such partner is dropped (kept = False, xyz NaN).
- landmarks.csv: one control point per DN cell type and side matched by type in the stage-A bridge crosswalk
  (family matches are not used); the FlyWire mean soma and the BANC mean root position of the type's neurons with
  a nucleus id (neurons within a type have no one-to-one match). A landmark without such a BANC neuron is dropped.
- transforms.json: FlyWire nm -> BANC nm, a 12-parameter affine and a 7-parameter similarity (orthogonal matrix,
  reflection allowed and reported through the determinant), both least squares with iterative 3-sigma rejection
  (a control point is removed while its residual norm exceeds 3 x the RMSE; refit until none is). 20 % of the
  landmark cell types (both sides together; default_rng(0)) are held out as check points. Per transform: the RMSE
  in micrometres on control and check points, the outliers removed, the determinant, and the side check (fraction of
  the FlyWire left / right DNs that land on the BANC left / right; BANC side of a point = majority side label of its
  15 nearest BANC neurons with a nucleus id and a left / right label).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "runs" / "viz"
META = ROOT / "banc" / "banc_888_meta.feather"
CENTROIDS = ROOT / "data" / "neuron_arbor_centroids.npz"
FW_VOXEL_NM = np.array([4.0, 4.0, 40.0])
HOLDOUT_FRAC, SPLIT_SEED = 0.2, 0
SIGMA_K = 3.0
SIDE_K = 15
FILES = ("positions_brain.npz", "positions_vnc.npz", "landmarks.csv", "transforms.json")


def parse_xyz(values) -> np.ndarray:
    """'x, y, z' strings (BANC meta) -> (n, 3) float; NaN where missing."""
    out = np.full((len(values), 3), np.nan)
    for i, s in enumerate(values):
        if isinstance(s, str) and s.strip():
            out[i] = [float(t) for t in s.split(",")]
    return out


def banc_meta() -> pd.DataFrame:
    m = pd.read_feather(META, columns=["banc_888_id", "root_position_nm", "nucleus_id", "side"])
    m["bid"] = m["banc_888_id"].astype(np.int64)
    m = m.drop_duplicates("bid").set_index("bid")
    xyz = parse_xyz(m["root_position_nm"].to_numpy())
    m["x"], m["y"], m["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    m["has_nucleus"] = m["nucleus_id"].notna() & (m["nucleus_id"].astype(str).str.strip() != "")
    return m


# ── position tables ─────────────────────────────────────────────────────────────────────────────────────────────────
def brain_positions(neurons: pd.DataFrame) -> dict:
    soma = neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * FW_VOXEL_NM
    c = np.load(CENTROIDS)
    if str(c["unit"]) != "nm" or not np.array_equal(c["root_id"], neurons["id"].to_numpy()):
        raise SystemExit("neuron_arbor_centroids.npz: unit or neuron order differs from the brain model")
    has = np.isfinite(soma).all(1)
    xyz = np.where(has[:, None], soma, c["xyz"].astype(float))
    src = np.where(has, "soma", np.where(c["source"] == 1, "centroid_synapses", "centroid_fallback"))
    return dict(id=neurons["id"].to_numpy(np.int64), xyz_nm=xyz, source=src.astype("U17"))


def vnc_positions(sub, meta: pd.DataFrame) -> dict:
    ids = sub.neurons["id"].to_numpy(np.int64)
    r = meta.reindex(ids)
    xyz = r[["x", "y", "z"]].to_numpy(float)
    has = np.isfinite(xyz).all(1)
    pre, post = sub.edge_endpoints()
    w = sub.edges["n_syn"].to_numpy(float)
    # partners of each missing neuron, both directions, weighted by synapse count; only partners with a root position
    a = np.concatenate([pre, post])
    b = np.concatenate([post, pre])
    ww = np.concatenate([w, w])
    k = ~has[a] & has[b]
    n = len(ids)
    den = np.bincount(a[k], weights=ww[k], minlength=n)
    fill = np.stack([np.bincount(a[k], weights=ww[k] * xyz[b[k], j], minlength=n) for j in range(3)], 1)
    can = ~has & (den > 0)
    xyz[can] = fill[can] / den[can, None]
    src = np.where(has, "root_position", np.where(can, "filled", "dropped"))
    return dict(id=ids, xyz_nm=xyz, source=src.astype("U13"), kept=has | can,
                region=sub.neurons["region"].astype(str).to_numpy().astype("U8"))


# ── landmarks ───────────────────────────────────────────────────────────────────────────────────────────────────────
def landmarks(setup, bridge, brain: dict, vnc_ids: np.ndarray, meta: pd.DataFrame) -> tuple:
    r = meta.reindex(vnc_ids)
    bxyz = r[["x", "y", "z"]].to_numpy(float)
    nuc = r["has_nucleus"].fillna(False).to_numpy(bool) & np.isfinite(bxyz).all(1)
    fw_soma = brain["source"] == "soma"
    rows, dropped = [], []
    for u in bridge.units:
        if u["match"] != "type":
            continue
        t, side = u["fw_types"][0], u["side"]
        fi = setup.fw_index[np.asarray(u["fw_pos"], np.int64)]
        fi = fi[fw_soma[fi]]
        bi = np.asarray(u["banc_index"], np.int64)
        bi = bi[nuc[bi]]
        if not len(fi) or not len(bi):
            dropped.append(dict(cell_type=t, side=side, n_fw=int(len(fi)), n_banc=int(len(bi))))
            continue
        f, b = brain["xyz_nm"][fi].mean(0), bxyz[bi].mean(0)
        rows.append(dict(cell_type=t, side=side, n_fw=len(fi), n_banc=len(bi),
                         fw_x=f[0], fw_y=f[1], fw_z=f[2], banc_x=b[0], banc_y=b[1], banc_z=b[2]))
    lm = pd.DataFrame(rows)
    types = np.array(sorted(lm["cell_type"].unique()))
    rng = np.random.default_rng(SPLIT_SEED)
    check = set(rng.permutation(types)[:int(round(HOLDOUT_FRAC * len(types)))])
    lm["set"] = np.where(lm["cell_type"].isin(check), "check", "control")
    return lm, dropped


# ── transforms ──────────────────────────────────────────────────────────────────────────────────────────────────────
def fit_affine(X, Y):
    """Y ~ X @ A.T + t: (A (3, 3), t (3,))."""
    H = np.c_[X, np.ones(len(X))]
    P, *_ = np.linalg.lstsq(H, Y, rcond=None)
    return P[:3].T, P[3]


def fit_similarity(X, Y):
    """Y ~ s R X + t with R orthogonal (Umeyama 1991, reflection allowed): (s R, t)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    R = U @ Vt
    s = S.sum() / (Xc ** 2).sum(1).mean()
    M = s * R
    return M, my - M @ mx


def apply(M, t, X):
    return X @ M.T + t


def rmse_um(M, t, X, Y):
    return float(np.sqrt(((apply(M, t, X) - Y) ** 2).sum(1).mean()) / 1e3)


def robust_fit(fit, X, Y):
    keep = np.ones(len(X), bool)
    while True:
        M, t = fit(X[keep], Y[keep])
        r = np.linalg.norm(apply(M, t, X) - Y, axis=1)
        lim = SIGMA_K * np.sqrt((r[keep] ** 2).mean())
        out = keep & (r > lim)
        if not out.any():
            return M, t, keep
        keep &= ~out


class BancSide:
    """Majority left / right label of the SIDE_K nearest BANC neurons with a nucleus id."""

    def __init__(self, meta: pd.DataFrame):
        from scipy.spatial import cKDTree
        m = meta[meta["has_nucleus"] & meta["side"].isin(["left", "right"]) & meta["x"].notna()]
        self.tree = cKDTree(m[["x", "y", "z"]].to_numpy(float))
        self.left = (m["side"] == "left").to_numpy()

    def __call__(self, xyz):
        _, ii = self.tree.query(xyz, k=SIDE_K)
        return np.where(self.left[ii].mean(1) > 0.5, "left", "right")


def evaluate(name, fit, lm, fw_dn_xyz, fw_dn_side, side_of):
    X = lm[["fw_x", "fw_y", "fw_z"]].to_numpy(float)
    Y = lm[["banc_x", "banc_y", "banc_z"]].to_numpy(float)
    ctl = (lm["set"] == "control").to_numpy()
    M, t, keep = robust_fit(fit, X[ctl], Y[ctl])
    land = side_of(apply(M, t, fw_dn_xyz))
    side = {s: dict(n=int((fw_dn_side == s).sum()), fraction_same_side=float((land[fw_dn_side == s] == s).mean()))
            for s in ("left", "right")}
    return dict(name=name, n_params=12 if name == "affine" else 7, matrix=M.tolist(), translation=t.tolist(),
                determinant=float(np.linalg.det(M)), reflection=bool(np.linalg.det(M) < 0),
                scale=float(abs(np.linalg.det(M)) ** (1 / 3)),
                n_control=int(ctl.sum()), n_check=int((~ctl).sum()), outliers_removed=int((~keep).sum()),
                outliers=lm[ctl][~keep][["cell_type", "side"]].to_dict("records"),
                rmse_control_um=rmse_um(M, t, X[ctl][keep], Y[ctl][keep]),
                rmse_control_all_um=rmse_um(M, t, X[ctl], Y[ctl]),
                rmse_check_um=rmse_um(M, t, X[~ctl], Y[~ctl]), side_check=side)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    existing = [f for f in FILES if (OUT / f).exists()]
    if existing and not a.overwrite:
        raise SystemExit(f"{OUT}: {existing} exist; pass --overwrite")
    import f3a_vnc_run as F
    import f5_train as T
    from connectome.adapters import flywire783 as FW
    from f3a3_subgraph import vnc_subgraph
    from train import f5_stageA as A

    neurons = FW.load_neurons()
    brain = brain_positions(neurons)
    meta = banc_meta()
    sub, _ = vnc_subgraph(F.load())
    setup = T.load_setup()
    if sub.n_neurons != setup.n:
        raise SystemExit("the VNC subgraph differs from the stage-A setup")
    vnc = vnc_positions(sub, meta)
    lm, dropped = landmarks(setup, A.make_bridge(setup, "true", 0), brain, vnc["id"], meta)

    dn = (neurons["super_class"] == "descending").to_numpy() & (brain["source"] == "soma")
    fw_side = neurons["side"].to_numpy()
    sel = dn & np.isin(fw_side, ["left", "right"])
    side_of = BancSide(meta)
    res = dict(note="visualisation scaffold (origin class E); FlyWire nm -> BANC nm",
               landmarks=dict(n=len(lm), n_types=int(lm["cell_type"].nunique()),
                              n_control=int((lm["set"] == "control").sum()), n_check=int((lm["set"] == "check").sum()),
                              n_check_types=int(lm.loc[lm["set"] == "check", "cell_type"].nunique()),
                              dropped=dropped, holdout_frac=HOLDOUT_FRAC, split_seed=SPLIT_SEED),
               side_check_dns=int(sel.sum()), sigma_k=SIGMA_K, side_k=SIDE_K,
               transforms={nm: evaluate(nm, f, lm, brain["xyz_nm"][sel], fw_side[sel], side_of)
                           for nm, f in (("affine", fit_affine), ("similarity", fit_similarity))},
               brain=dict(n=len(brain["id"]), **{s: int((brain["source"] == s).sum())
                                                 for s in ("soma", "centroid_synapses", "centroid_fallback")}),
               vnc=dict(n=len(vnc["id"]), **{s: int((vnc["source"] == s).sum())
                                             for s in ("root_position", "filled", "dropped")}))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "positions_brain.npz", **brain)
    np.savez_compressed(OUT / "positions_vnc.npz", **vnc)
    lm.to_csv(OUT / "landmarks.csv", index=False)
    (OUT / "transforms.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "transforms"}, indent=1))
    for nm, r in res["transforms"].items():
        print(nm, json.dumps({k: v for k, v in r.items() if k not in ("matrix", "translation")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
