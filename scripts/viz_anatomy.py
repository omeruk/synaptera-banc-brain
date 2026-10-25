"""Visualisation (demonstration, no claim): publication-style anatomy figure of the stage-A circuit with neuron
skeletons and neuropil outlines from the public BANC bucket, and a check of our affine FlyWire -> BANC transform
(runs/viz/transforms.json) against the BANC team's warp. Origin class E (engineering scaffold, viewing only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/viz_anatomy.py [ids|fetch|check|figure|all] [--overwrite]

Source: gs://lee-lab_brain-and-nerve-cord-fly-connectome (public, no login; README.txt and documentation/ describe
units and axes; mirrored at Harvard Dataverse doi:10.7910/DVN/7WTH1N). Downloads mirror the bucket paths under
data/viz_cache/bucket/ (not committed), are stored as served (gzip where the bucket stores gzip) and are checked
against the bucket size and MD5. A single object over 2 GB or a download total over 10 GB stops the run before
anything is fetched.

Steps (outputs under runs/viz/anatomy/, not committed; an existing output is not overwritten without --overwrite):
- ids: the circuit and its IDs. FlyWire: the brain-model root IDs and the bridged DNs of the stage-A bridge
  (make_bridge(setup, "true", 0)), looked up in compiled_data/fafb_783/fafb_783_meta.feather. BANC: the ID version of
  the VNC subgraph (matches per root_* column of compiled_data/banc_888/banc_888_meta.feather), the bridge-target DNs,
  the 16 steering MNs (setup.readout) and at most 150 VNC interneurons (bucket super_class
  ventral_nerve_cord_intrinsic, neither target nor MN) on the strongest two-hop paths target -> interneuron -> MN in
  the stage-A subgraph. Path strength = summed synapse count n_syn of its two edges; an interneuron ranks by its
  strongest path (ties: all two-hop synapses through it, then root ID). -> ids.json, circuit.csv
- fetch: the SWCs of the circuit (FlyWire DNs from fafb_783/fafb_banc_space_swc (warped into BANC space by the BANC
  team; first of <id>_skeleton.swc, <id>_l2.swc, <id>.swc) and fafb_783/fafb_fafb_space_swc (native space, <id>.swc);
  BANC neurons from banc_888/banc_banc_space_swc, <id>_skeleton.swc, else <id>_l2.swc) and the outline meshes
  (region_outlines, legacy Neuroglancer precomputed; segment IDs looked up by label in
  region_outlines/segment_properties/info). -> fetch_manifest.csv, fetch.json
- check: (1) the BANC axes from region meshes (antennal lobes, mushroom-body calyces, GNG) and the brain / VNC
  outlines; (2) per bridged FlyWire DN, the root node of its native-space skeleton against our FlyWire soma (unit
  of the native SWCs taken as the one that puts more roots within 5 um: nm or FlyWire voxels x (4, 4, 40) nm); for
  the DNs within 5 um whose BANC-space file is the native tree node for node (same parents; otherwise it is another
  skeleton whose root is not the soma), our affine-transformed soma against the root node of the BANC-space
  skeleton: median, RMSE and 95th percentile in um, and the dependence on the distance from the centroid of the
  fitted control landmarks (Spearman, least-squares slope, tertile medians). -> axes.json, warp_check.csv,
  warp_check.json
- figure: panels a-d (frontal, dorsal, lateral, posterior; orthographic, one scale), brain and VNC outlines as
  semi-opaque light grey fills, the circuit as thin semi-transparent lines in four Okabe-Ito colours (the subset with
  the largest colour-vision-deficiency separation). Each neuron is split at the posterior end of the brain outline
  and brain and VNC are drawn far to near (outline, then neurons), so in the frontal and posterior views the nearer
  region dims the farther one; BANC DNs are faint in the brain (alpha ramp over 40 um behind the cut). Skeletons are
  reduced for drawing: terminal twigs < 2 um removed in one pass (skeletor files carry sub-micron hairs below the line
  width), nodes >= 1 um apart along each unbranched chain, chain ends kept, straight edges > 20 um (healing jumps)
  dropped. PDF (vector) + PNG (600 dpi) + caption.txt in the preview folder; figure.json in runs/viz/anatomy/.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BUCKET = "lee-lab_brain-and-nerve-cord-fly-connectome"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
DOI = "10.7910/DVN/7WTH1N"
CACHE = ROOT / "data" / "viz_cache" / "bucket"
VIZ = ROOT / "runs" / "viz"
OUT = VIZ / "anatomy"
FIG_DIR = Path("/mnt/c/Users/omerf/Desktop/synaptera_preview/figure")
FIG_NAME = "stageA_circuit_anatomy"
DOCS = ("README.txt",) + tuple(f"documentation/{n}.md" for n in (
    "banc_swc_skeletons", "fafb_783_banc_space_swc", "banc_neuropil_meshes", "banc_888_meta", "neck_connective_y92500",
    "acknowledgements"))
FAFB_META = "compiled_data/fafb_783/fafb_783_meta.feather"
BANC_META = "compiled_data/banc_888/banc_888_meta.feather"
SEGPROPS = "region_outlines/segment_properties/info"
MESHES = "region_outlines/meshes/"
SWC = {"fafb_banc": ("compiled_data/fafb_783/fafb_banc_space_swc/{}_skeleton.swc",     # served names carry a suffix,
                      "compiled_data/fafb_783/fafb_banc_space_swc/{}_l2.swc",           # unlike fafb_783_banc_space_
                      "compiled_data/fafb_783/fafb_banc_space_swc/{}.swc"),             # swc.md states
       "fafb_fafb": ("compiled_data/fafb_783/fafb_fafb_space_swc/{}.swc",),
       "banc": ("compiled_data/banc_888/banc_banc_space_swc/{}_skeleton.swc",
                "compiled_data/banc_888/banc_banc_space_swc/{}_l2.swc")}
OUTLINES = ("BANC_brain_neuropil", "BANC_vnc_neuropil")
AXIS_REGIONS = ("ITO_midbrain_AL_L", "ITO_midbrain_AL_R", "ITO_midbrain_MB_CA_L", "ITO_midbrain_MB_CA_R",
                "ITO_midbrain_GNG")
BANC_ROOT_COLUMNS = ("banc_888_id", "root_id", "root_626", "root_850", "root_888", "root_890")
MAX_OBJECT_BYTES, MAX_TOTAL_BYTES = 2e9, 10e9
THREADS = 16
N_INTERNEURONS = 150
INTERNEURON_CLASS = "ventral_nerve_cord_intrinsic"
ROOT_TOL_UM = 5.0
FW_VOXEL_NM = np.array([4.0, 4.0, 40.0])
FILES = {"ids": ("ids.json", "circuit.csv"), "fetch": ("fetch_manifest.csv", "fetch.json"),
         "check": ("axes.json", "warp_check.csv", "warp_check.json"), "figure": ("figure.json",)}
FIG_FILES = (f"{FIG_NAME}.pdf", f"{FIG_NAME}.png", "caption.txt")

# BANC axes (README / documentation and the check step): +x the fly's left, +y posterior, +z dorsal
AXIS_ENDS = {0: ("R", "L"), 1: ("A", "P"), 2: ("V", "D")}
# view -> (screen right, screen up) as BANC unit vectors; each pair with right x up = toward the viewer (a rotation)
VIEWS = {"frontal": ((1, 0, 0), (0, 0, 1)), "dorsal": ((-1, 0, 0), (0, -1, 0)),
         "lateral": ((0, 0, 1), (0, -1, 0)), "posterior": ((-1, 0, 0), (0, 0, 1))}
ROLES = ("fw_dn", "banc_dn", "interneuron", "mn")
DRAW_ORDER = ("banc_dn", "fw_dn", "interneuron", "mn")               # back to front within a part
# Okabe-Ito sky blue, blue, orange, vermillion: colour, line width (pt), alpha in the brain part, alpha in the VNC
# part. BANC DNs are faint in the brain, where the FlyWire DNs of the same types run.
STYLE = {"fw_dn": ("#56B4E9", 0.15, 0.15, 0.15), "banc_dn": ("#0072B2", 0.15, 0.05, 0.3),
         "interneuron": ("#E69F00", 0.15, 0.35, 0.35), "mn": ("#D55E00", 0.22, 0.5, 0.5)}
RAMP_UM, N_RAMP = 40.0, 4                  # brain -> VNC alpha ramp behind the neck cut, in bands
PART_BANDS = {"brain": (0,), "vnc": tuple(range(1, N_RAMP + 2))}
OUTLINE_FILL, OUTLINE_ALPHA, OUTLINE_EDGE, OUTLINE_LW = "#E4E4E4", 0.75, "#C8C8C8", 0.3
MIN_CONTOUR_UM2 = 50.0
MAX_EDGE_UM = 20.0
KEY_BOX_MM = 9.0
TWIG_UM, STEP_UM = 2.0, 1.0
SIL_RES_UM = 0.5
FIG_W_MM, MARGIN_MM, GAP_MM, PAD_UM = 180.0, 1.5, 5.0, 4.0
FONT_PT, LETTER_PT = 7, 8
SCALE_UM = 100
DPI = 600
MM = 1 / 25.4


# ── bucket access ───────────────────────────────────────────────────────────────────────────────────────────────────
def _http(url, headers=None, tries=4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from None
            if (e.code < 500 and e.code != 429) or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
        time.sleep(2 ** k)


def stat(name):
    """Bucket metadata of one object, or None if it does not exist."""
    q = "?fields=name,size,md5Hash,generation,contentEncoding"
    try:
        d = json.loads(_http(API + urllib.parse.quote(name, safe="") + q))
    except FileNotFoundError:
        return None
    return dict(object=d["name"], size=int(d["size"]), md5=d.get("md5Hash", ""), generation=d.get("generation", ""),
                encoding=d.get("contentEncoding", ""))


def stat_first(names):
    """The first of the candidate names that exists, else None."""
    for n in names:
        o = stat(n)
        if o is not None:
            return o
    return None


def _md5(b: bytes) -> str:
    return base64.b64encode(hashlib.md5(b).digest()).decode()


def _cached(o) -> bool:
    p = CACHE / o["object"]
    return p.exists() and p.stat().st_size == o["size"] and (not o["md5"] or _md5(p.read_bytes()) == o["md5"])


def _download(o) -> str:
    if _cached(o):
        return "cached"
    b = _http(MEDIA + urllib.parse.quote(o["object"]), {"Accept-Encoding": "gzip"})   # stored bytes, as hashed
    if len(b) != o["size"] or (o["md5"] and _md5(b) != o["md5"]):
        raise RuntimeError(f"{o['object']}: size or MD5 differs from the bucket")
    p = CACHE / o["object"]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(b)
    os.replace(tmp, p)
    return "downloaded"


def download(objs: list) -> list:
    """Download (or confirm in the cache) the objects; stops first if a size limit would be exceeded. Returns the
    status per object: downloaded or cached."""
    big = [o["object"] for o in objs if o["size"] > MAX_OBJECT_BYTES]
    total = sum(o["size"] for o in objs)
    if big or total > MAX_TOTAL_BYTES:
        raise SystemExit(f"size limit: objects over 2 GB {big}, total {total / 1e9:.2f} GB; ask before downloading")
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(_download, objs))


def stat_all(names: list) -> list:
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(stat, names))


def read_object(name) -> bytes:
    b = (CACHE / name).read_bytes()
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


def fetch_named(names) -> list:
    objs = stat_all(list(names))
    missing = [n for n, o in zip(names, objs) if o is None]
    if missing:
        raise SystemExit(f"not in the bucket: {missing}")
    return [dict(o, status=s) for o, s in zip(objs, download(objs))]


# ── readers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def segment_ids(labels) -> dict:
    d = json.loads(read_object(SEGPROPS))["inline"]
    values = next(p for p in d["properties"] if p["type"] == "label")["values"]
    m = dict(zip(values, d["ids"]))
    return {lab: m[lab] for lab in labels}


def read_mesh(seg_id: str):
    """Legacy Neuroglancer precomputed mesh: manifest '<id>:0' lists fragments; a fragment is uint32 n_vertices,
    float32 vertices (n, 3), uint32 triangle indices. Returns (vertices nm, faces)."""
    V, F, off = [], [], 0
    for frag in json.loads(read_object(f"{MESHES}{seg_id}:0"))["fragments"]:
        b = read_object(MESHES + frag)
        nv = int(np.frombuffer(b, "<u4", 1)[0])
        V.append(np.frombuffer(b, "<f4", 3 * nv, 4).reshape(nv, 3).astype(float))
        F.append(np.frombuffer(b, "<u4", offset=4 + 12 * nv).reshape(-1, 3).astype(np.int64) + off)
        off += nv
    return np.concatenate(V), np.concatenate(F)


def read_swc(name):
    """(xyz (n, 3) in file units, parent index (n,), -1 for a root). The file's first root is node 0 if listed first."""
    t = pd.read_csv(io.BytesIO(read_object(name)), sep=r"\s+", comment="#", header=None, usecols=range(7))
    t = t.to_numpy(float)
    ids, par = t[:, 0].astype(np.int64), t[:, 6].astype(np.int64)
    order = np.argsort(ids)
    pos = np.clip(np.searchsorted(ids[order], par), 0, len(ids) - 1)
    found = ids[order][pos] == par
    parent = np.where((par >= 0) & found, order[pos], -1)
    return t[:, 2:5], parent


def swc_root(parent) -> tuple:
    roots = np.flatnonzero(parent < 0)
    return int(roots[0]), len(roots)


def _root_distance(xyz, parent):
    """Cable distance of each node from its root (pointer doubling, no node order assumed)."""
    has = parent >= 0
    dist = np.zeros(len(xyz))
    dist[has] = np.linalg.norm(xyz[has] - xyz[parent[has]], axis=1)
    anc = parent.copy()
    while (anc >= 0).any():
        m = anc >= 0
        d_new, a_new = dist.copy(), anc.copy()
        d_new[m] += dist[anc[m]]
        a_new[m] = anc[anc[m]]
        dist, anc = d_new, a_new
    return dist


def _chain_tops(parent):
    """Top node of each node's unbranched chain (a chain runs up through parents that have one child and are not
    roots), and the child count per node."""
    n = len(parent)
    has = parent >= 0
    nchild = np.bincount(parent[has], minlength=n)
    p = np.maximum(parent, 0)
    link = np.where(has & (nchild[p] == 1) & (parent[p] >= 0), parent, np.arange(n))
    while True:
        nxt = link[link]
        if np.array_equal(nxt, link):
            return link, nchild
        link = nxt


def polylines(xyz, parent, twig, step):
    """Drawing polylines of a skeleton: terminal twigs shorter than `twig` removed (one pass); the unbranched chains
    are joined into polylines (at each branch point the child chain with the lowest node index continues the incoming
    line, the others start at the branch point), each chain keeping its ends and the nodes where the cable distance
    crosses a multiple of `step`. Returns vertices (m, 3) and the polyline-start mask (m,)."""
    dist = _root_distance(xyz, parent)
    top, nchild = _chain_tops(parent)
    base = parent[top]
    leaf = np.flatnonzero(nchild == 0)
    short = leaf[(base[leaf] >= 0) & (dist[leaf] - dist[np.maximum(base[leaf], 0)] < twig)]
    keep = ~np.isin(top, top[short])
    idx = np.cumsum(keep) - 1
    parent = np.where(parent[keep] >= 0, idx[np.maximum(parent[keep], 0)], -1)
    xyz, dist = xyz[keep], dist[keep]
    n = len(parent)
    top, _ = _chain_tops(parent)
    base = parent[top]
    tops = np.unique(top)
    tops = tops[parent[tops] >= 0]
    cont = tops[np.unique(parent[tops], return_index=True)[1]]       # lowest child top per branch point
    prv = np.full(n, -1)
    prv[cont] = top[parent[cont]]                                    # the chain that ends at the branch point
    head = np.where(prv >= 0, prv, np.arange(n))
    rank, a = (prv >= 0).astype(np.int64), prv.copy()
    while True:                                                      # list ranking by pointer doubling
        nxt = head[head]
        m = a >= 0
        if np.array_equal(nxt, head) and not m.any():
            break
        r2 = rank.copy()
        r2[m] += rank[a[m]]
        a2 = a.copy()
        a2[m] = a[a[m]]
        head, rank, a = nxt, r2, a2
    o = np.lexsort((dist, rank[top], head[top]))                     # polyline by polyline, chain by chain
    t, d = top[o], dist[o]
    first = np.r_[True, t[1:] != t[:-1]]
    last = np.r_[t[1:] != t[:-1], True]
    sel = first | last | (np.floor(d / step) != np.floor(np.r_[-1.0, d[:-1]] / step))
    s = o[sel]
    start = first[sel] & (prv[top[s]] < 0)
    b = base[s]
    ins = np.flatnonzero(start & (b >= 0))                           # a new polyline starts at its branch point
    return np.insert(xyz[s], ins, xyz[b[ins]], axis=0), np.insert(start & (b < 0), ins, True)


def bucket_meta(name, cols) -> pd.DataFrame:
    return pd.read_feather(io.BytesIO(read_object(name)), columns=list(cols))


# ── step: ids ───────────────────────────────────────────────────────────────────────────────────────────────────────
def two_hop_interneurons(setup, targets, mns, super_class):
    n = setup.n
    pre, post = setup.pre.astype(np.int64), setup.post.astype(np.int64)
    w = setup.n_syn.astype(float)
    is_t, is_m = np.zeros(n, bool), np.zeros(n, bool)
    is_t[targets], is_m[mns] = True, True
    mid = ~is_t & ~is_m
    e1, e2 = is_t[pre] & mid[post], is_m[post] & mid[pre]
    w_in, w_out = np.zeros(n), np.zeros(n)
    np.maximum.at(w_in, post[e1], w[e1])
    np.maximum.at(w_out, pre[e2], w[e2])
    via_t, via_m = np.full(n, -1), np.full(n, -1)
    for e, src, dst, via in ((e1, pre, post, via_t), (e2, post, pre, via_m)):
        i = np.flatnonzero(e)
        i = i[np.lexsort((src[i], -w[i]))]                           # strongest edge first, then lowest node
        first = np.unique(dst[i], return_index=True)[1]
        via[dst[i][first]] = src[i][first]
    total = np.bincount(post[e1], w[e1], n) + np.bincount(pre[e2], w[e2], n)
    two_hop = (w_in > 0) & (w_out > 0)
    ok = two_hop & (super_class == INTERNEURON_CLASS)
    return dict(w_in=w_in, w_out=w_out, total=total, via_t=via_t, via_m=via_m, two_hop=two_hop, ok=ok)


def step_ids(a):
    import f5_train as T
    from train import f5_stageA as A

    fetch_named(DOCS + (FAFB_META, BANC_META, SEGPROPS))
    setup = T.load_setup()
    bridge = A.make_bridge(setup, "true", 0)
    brain = np.load(VIZ / "positions_brain.npz")
    vnc = np.load(VIZ / "positions_vnc.npz")
    fw_ids, vnc_ids = brain["id"].astype(np.int64), vnc["id"].astype(np.int64)
    if len(vnc_ids) != setup.n:
        raise SystemExit("positions_vnc.npz differs from the stage-A setup")

    fafb = bucket_meta(FAFB_META, ["fafb_783_id"])
    fafb_ids = fafb["fafb_783_id"].astype(np.int64).to_numpy()
    cols = ["banc_888_id", "root_id", "root_626", "root_850", "root_888", "root_890", "super_class", "cell_type",
            "side"]
    banc = bucket_meta(BANC_META, cols)
    banc["bid"] = banc["banc_888_id"].astype(np.int64)
    bm = banc.drop_duplicates("bid").set_index("bid")
    version = {c: int(np.isin(vnc_ids, pd.to_numeric(banc[c], errors="coerce").dropna().astype(np.int64)).sum())
               for c in BANC_ROOT_COLUMNS}

    fw_pos = np.unique(np.concatenate([np.asarray(u["fw_pos"], np.int64) for u in bridge.units]))
    fw_node = setup.fw_index[fw_pos].astype(np.int64)
    targets = np.unique(np.concatenate([np.asarray(u["banc_index"], np.int64) for u in bridge.units]))
    mns = np.asarray(setup.readout, np.int64)
    sc = bm["super_class"].reindex(vnc_ids).astype(object).to_numpy()
    hop = two_hop_interneurons(setup, targets, mns, sc)
    cand = np.flatnonzero(hop["ok"])
    score = hop["w_in"][cand] + hop["w_out"][cand]
    cand = cand[np.lexsort((vnc_ids[cand], -hop["total"][cand], -score))][:N_INTERNEURONS]
    score = hop["w_in"][cand] + hop["w_out"][cand]

    bt = setup.banc_table().set_index("index")
    rows = [dict(role="fw_dn", dataset="fafb_783", id=int(fw_ids[i]), node=int(i), cell_type=str(setup.fw_cell_type[p]),
                 side=str(setup.fw_side[p])) for p, i in zip(fw_pos, fw_node)]
    rows += [dict(role="banc_dn", dataset="banc_888", id=int(vnc_ids[i]), node=int(i), cell_type=bt.loc[i, "cell_type"],
                  side=bt.loc[i, "side"]) for i in targets]
    names = setup.meta["readout_names"]
    rows += [dict(role="mn", dataset="banc_888", id=int(vnc_ids[i]), node=int(i), cell_type=names[k].split("|")[0],
                  side=names[k].split("|")[1]) for k, i in enumerate(mns)]
    rows += [dict(role="interneuron", dataset="banc_888", id=int(vnc_ids[i]), node=int(i),
                  cell_type=bm["cell_type"].get(vnc_ids[i]), side=bm["side"].get(vnc_ids[i]),
                  w_in=hop["w_in"][i], w_out=hop["w_out"][i], path_syn=hop["w_in"][i] + hop["w_out"][i],
                  two_hop_syn=hop["total"][i], via_target_id=int(vnc_ids[hop["via_t"][i]]),
                  via_mn_id=int(vnc_ids[hop["via_m"][i]])) for i in cand]
    circ = pd.DataFrame(rows)
    b = circ["dataset"] == "banc_888"
    circ.loc[b, "banc_super_class"] = bm["super_class"].reindex(circ.loc[b, "id"]).to_numpy()
    circ.loc[b, "banc_cell_type"] = bm["cell_type"].reindex(circ.loc[b, "id"]).to_numpy()
    circ["in_meta"] = np.where(b, circ["id"].isin(bm.index), circ["id"].isin(fafb_ids))

    miss_fw = fw_ids[~np.isin(fw_ids, fafb_ids)]
    two = hop["two_hop"]
    res = dict(
        note="visualisation (origin class E); IDs of the stage-A circuit against the bucket meta tables",
        fafb_783_meta=dict(object=FAFB_META, rows=len(fafb_ids)),
        banc_888_meta=dict(object=BANC_META, rows=len(banc)),
        flywire=dict(model_neurons=len(fw_ids), in_fafb_783_meta=int(np.isin(fw_ids, fafb_ids).sum()),
                     missing=[int(i) for i in miss_fw], bridged_dns=len(fw_node),
                     bridged_dns_in_meta=int(np.isin(fw_ids[fw_node], fafb_ids).sum())),
        banc=dict(vnc_neurons=len(vnc_ids), matches_per_column=version,
                  version=[c for c, k in version.items() if k == len(vnc_ids)],
                  missing_in_banc_888_id=[int(i) for i in vnc_ids[~np.isin(vnc_ids, bm.index)]],
                  bridge_targets=len(targets), mns=len(mns), bridge_units=len(bridge.units)),
        interneurons=dict(rule="two-hop target -> k -> MN in the stage-A subgraph; path strength = n_syn(T->k) + "
                               "n_syn(k->MN); k ranked by its strongest path, ties by all two-hop synapses through k, "
                               "then root ID", super_class=INTERNEURON_CLASS, cap=N_INTERNEURONS,
                          two_hop_intermediates=int(two.sum()),
                          two_hop_by_super_class=pd.Series(sc[two]).fillna("none").value_counts().to_dict(),
                          eligible=int(hop["ok"].sum()), selected=len(cand),
                          path_syn_min_selected=float(score.min()) if len(cand) else None,
                          selected_w_in_below_5=int((hop["w_in"][cand] < 5).sum()),
                          selected_w_out_below_5=int((hop["w_out"][cand] < 5).sum())),
        circuit=circ["role"].value_counts().to_dict(),
        circuit_in_meta=circ.groupby("role")["in_meta"].sum().astype(int).to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    circ.to_csv(OUT / "circuit.csv", index=False)
    (OUT / "ids.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ── step: fetch ─────────────────────────────────────────────────────────────────────────────────────────────────────
def step_fetch(a):
    circ = pd.read_csv(OUT / "circuit.csv")
    jobs = []                                                        # (role, id, kind, candidate names)
    for r in circ.itertuples():
        kinds = ("fafb_banc", "fafb_fafb") if r.dataset == "fafb_783" else ("banc",)
        jobs += [(r.role, int(r.id), k, tuple(t.format(int(r.id)) for t in SWC[k])) for k in kinds]
    seg = segment_ids(OUTLINES + AXIS_REGIONS)
    manifests = fetch_named([f"{MESHES}{s}:0" for s in seg.values()])
    frags = [MESHES + f for m in manifests for f in json.loads(read_object(m["object"]))["fragments"]]
    with ThreadPoolExecutor(THREADS) as ex:
        found = list(ex.map(stat_first, [j[3] for j in jobs]))
    frag_objs = fetch_named(frags)
    objs = [o for o in found if o is not None]
    status = download(objs)
    st = iter(status)
    rows = [dict(role=j[0], id=j[1], kind=j[2], object=o["object"] if o else "", size=o["size"] if o else 0,
                 md5=o["md5"] if o else "", generation=o["generation"] if o else "",
                 status=next(st) if o else "missing") for j, o in zip(jobs, found)]
    rows += [dict(role="outline", id=0, kind="mesh", object=o["object"], size=o["size"], md5=o["md5"],
                  generation=o["generation"], status=o["status"]) for o in manifests + frag_objs]
    man = pd.DataFrame(rows)
    swc = man[man["kind"] != "mesh"]
    res = dict(
        note="SWC and mesh downloads into data/viz_cache/bucket/ (not committed)",
        segment_ids=seg,
        swc={f"{role}/{kind}": dict(n=int(len(g)), found=int((g["status"] != "missing").sum()),
                                    skeleton=int(g["object"].str.endswith("_skeleton.swc").sum()),
                                    l2=int(g["object"].str.endswith("_l2.swc").sum()),
                                    bytes=int(g["size"].sum()),
                                    missing=[int(i) for i in g.loc[g["status"] == "missing", "id"]])
             for (role, kind), g in swc.groupby(["role", "kind"])},
        mesh_bytes=int(man.loc[man["kind"] == "mesh", "size"].sum()),
        total_bytes=int(man["size"].sum()), downloaded=int((man["status"] == "downloaded").sum()),
        cached=int((man["status"] == "cached").sum()))
    man.to_csv(OUT / "fetch_manifest.csv", index=False)
    (OUT / "fetch.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ── step: check ─────────────────────────────────────────────────────────────────────────────────────────────────────
def axes_check(seg) -> dict:
    c = {lab: read_mesh(seg[lab])[0].mean(0) / 1e3 for lab in AXIS_REGIONS}
    lim = {lab: (lambda v: dict(min=(v.min(0) / 1e3).round(1).tolist(), max=(v.max(0) / 1e3).round(1).tolist()))(
        read_mesh(seg[lab])[0]) for lab in OUTLINES}
    al = (c["ITO_midbrain_AL_L"] + c["ITO_midbrain_AL_R"]) / 2
    ca = (c["ITO_midbrain_MB_CA_L"] + c["ITO_midbrain_MB_CA_R"]) / 2
    gng = c["ITO_midbrain_GNG"]
    d = dict(x_AL_L_minus_AL_R_um=float(c["ITO_midbrain_AL_L"][0] - c["ITO_midbrain_AL_R"][0]),
             x_MB_CA_L_minus_R_um=float(c["ITO_midbrain_MB_CA_L"][0] - c["ITO_midbrain_MB_CA_R"][0]),
             y_MB_CA_minus_AL_um=float(ca[1] - al[1]), z_MB_CA_minus_GNG_um=float(ca[2] - gng[2]),
             z_AL_minus_GNG_um=float(al[2] - gng[2]),
             y_vnc_min_minus_brain_max_um=float(lim[OUTLINES[1]]["min"][1] - lim[OUTLINES[0]]["max"][1]))
    return dict(note="centroids (mean vertex) of region_outlines meshes, um; BANC space",
                plus_x_is_left=d["x_AL_L_minus_AL_R_um"] > 0 and d["x_MB_CA_L_minus_R_um"] > 0,
                plus_y_is_posterior=d["y_MB_CA_minus_AL_um"] > 0 and d["y_vnc_min_minus_brain_max_um"] > 0,
                plus_z_is_dorsal=d["z_MB_CA_minus_GNG_um"] > 0 and d["z_AL_minus_GNG_um"] > 0,
                differences=d, outline_bounds_um=lim,
                documented=dict(units="SWC nodes in nm in BANC space (README.txt, banc_swc_skeletons.md)",
                                posterior="+y: neck planes y = 92500 (anterior) and 121000 (posterior) voxels "
                                          "(neck_connective_y92500.md)"))


def step_check(a):
    fetch = json.loads((OUT / "fetch.json").read_text())
    axes = axes_check(fetch["segment_ids"])
    circ = pd.read_csv(OUT / "circuit.csv")
    man = pd.read_csv(OUT / "fetch_manifest.csv").fillna({"object": ""})
    obj = {(r.id, r.kind): r.object for r in man.itertuples() if r.object}
    brain = np.load(VIZ / "positions_brain.npz")
    tr = json.loads((VIZ / "transforms.json").read_text())["transforms"]["affine"]
    M, t = np.array(tr["matrix"]), np.array(tr["translation"])
    lm = pd.read_csv(VIZ / "landmarks.csv")
    out = {(o["cell_type"], o["side"]) for o in tr["outliers"]}
    lm["fitted"] = (lm["set"] == "control") & ~pd.Series(list(zip(lm["cell_type"], lm["side"]))).isin(out)
    centroid = lm.loc[lm["fitted"], ["banc_x", "banc_y", "banc_z"]].to_numpy(float).mean(0)
    lm_set = {(r.cell_type, r.side): ("control" if r.fitted else "control_outlier" if r.set == "control" else "check")
              for r in lm.itertuples()}

    dn = circ[circ["role"] == "fw_dn"].reset_index(drop=True)
    rows = []
    for r in dn.itertuples():
        node = int(r.node)
        row = dict(id=int(r.id), cell_type=r.cell_type, side=r.side, soma=str(brain["source"][node]) == "soma",
                   landmark=lm_set.get((r.cell_type, r.side), "none"))
        soma = brain["xyz_nm"][node]
        ours = soma @ M.T + t
        nat, war = obj.get((int(r.id), "fafb_fafb")), obj.get((int(r.id), "fafb_banc"))
        if nat and war:
            xn, pn = read_swc(nat)
            xw, pw = read_swc(war)
            rn, n_roots = swc_root(pn)
            rw, _ = swc_root(pw)
            row.update(n_roots_native=n_roots, nodes_native=len(xn), nodes_warped=len(xw),
                       same_tree=len(xn) == len(xw) and np.array_equal(pn, pw),
                       root_native=xn[rn].tolist(), root_warped=xw[rw].tolist(),
                       nearest_warped_node_um=float(np.linalg.norm(xw - ours, axis=1).min() / 1e3))
            for unit, f in (("nm", 1.0), ("voxel", FW_VOXEL_NM)):
                row[f"root_soma_um_{unit}"] = float(np.linalg.norm(xn[rn] * f - soma) / 1e3)
        rows.append(row)
    d = pd.DataFrame(rows)
    has = d["root_native"].notna() & d["soma"]
    n_ok = {u: int((has & (d[f"root_soma_um_{u}"] <= ROOT_TOL_UM)).sum()) for u in ("nm", "voxel")}
    unit = max(n_ok, key=n_ok.get)
    d["root_soma_um"] = d[f"root_soma_um_{unit}"]
    d["root_ok"] = has & (d["root_soma_um"] <= ROOT_TOL_UM)
    # the warped root is the warped soma only where the warped file is the native tree node for node; the BANC team
    # warped FlyWire's L2 skeleton where no split skeleton exists, and its root is not at the soma
    d["used"] = d["root_ok"] & d["same_tree"].fillna(False).astype(bool)
    soma = brain["xyz_nm"][dn["node"].to_numpy(int)]
    ours_all = soma @ M.T + t
    ok = d["root_ok"].to_numpy()
    warped_all = np.full_like(ours_all, np.nan)
    warped_all[ok] = np.array(d.loc[ok, "root_warped"].tolist())
    err_all = np.linalg.norm(ours_all - warped_all, axis=1) / 1e3
    d["root_vs_root_um"] = err_all
    u = d[d["used"]]
    ours, warped = ours_all[u.index], warped_all[u.index]
    err = err_all[u.index]
    rad = np.linalg.norm(warped - centroid, axis=1) / 1e3
    d.loc[u.index, "affine_vs_warp_um"] = err
    d.loc[u.index, "dist_from_landmark_centroid_um"] = rad
    for k, ax in enumerate("xyz"):
        d.loc[u.index, f"diff_{ax}_um"] = (ours[:, k] - warped[:, k]) / 1e3

    from scipy.stats import spearmanr
    rho, p = spearmanr(rad, err)
    slope, icpt = np.polyfit(rad, err, 1)
    q = np.quantile(rad, [0, 1 / 3, 2 / 3, 1])
    bin_ = np.minimum(np.searchsorted(q, rad, side="right") - 1, 2)
    tert = [dict(dist_um=[float(q[i]), float(q[i + 1])], n=int((bin_ == i).sum()),
                 median_err_um=float(np.median(err[bin_ == i]))) for i in range(3)]

    def summary(e):
        return dict(n=int(len(e)), median_um=float(np.median(e)), rmse_um=float(np.sqrt((e ** 2).mean())),
                    p95_um=float(np.quantile(e, 0.95)), max_um=float(e.max())) if len(e) else dict(n=0)

    other = d[d["root_ok"] & ~d["used"]]
    res = dict(
        note="visualisation (origin class E); our affine FlyWire -> BANC soma vs the root node of the BANC team's "
             "warped skeleton (FAFB14 -> JRC2018F -> mirror -> BANC, elastix + thin-plate spline)",
        bridged_dns=len(d), with_flywire_soma=int(d["soma"].sum()),
        with_both_skeletons=int(d["root_native"].notna().sum()),
        native_unit=unit, native_unit_votes=n_ok, root_tol_um=ROOT_TOL_UM,
        root_soma_um_median=float(d.loc[has, "root_soma_um"].median()),
        native_multiple_roots=int((d["n_roots_native"] > 1).sum()),
        root_within_tol=int(d["root_ok"].sum()), used=int(d["used"].sum()),
        error=summary(err),
        error_by_landmark_set={k: summary(err[(u["landmark"] == k).to_numpy()])
                               for k in ("control", "control_outlier", "check", "none")},
        mean_signed_diff_um=dict(zip("xyz", np.mean((ours - warped) / 1e3, 0).round(2).tolist())),
        landmark_centroid_banc_um=(centroid / 1e3).round(1).tolist(), n_fitted_landmarks=int(lm["fitted"].sum()),
        distance_dependence=dict(spearman_rho=float(rho), spearman_p=float(p),
                                 slope_um_per_100um=float(100 * slope), intercept_um=float(icpt), tertiles=tert),
        excluded_other_tree=dict(
            n=int(len(other)), note="warped file is not the native tree node for node (other skeleton; its root is "
                                    "not the soma)",
            median_node_ratio_native_to_warped=float((other["nodes_native"] / other["nodes_warped"]).median()),
            root_vs_root=summary(other["root_vs_root_um"].to_numpy()),
            nearest_warped_node_um=summary(other["nearest_warped_node_um"].to_numpy())),
        all_root_within_tol_root_vs_root=summary(err_all[ok]))
    d.drop(columns=["root_native", "root_warped"]).to_csv(OUT / "warp_check.csv", index=False)
    (OUT / "warp_check.json").write_text(json.dumps(res, indent=1))
    (OUT / "axes.json").write_text(json.dumps(axes, indent=1))
    print(json.dumps(axes, indent=1))
    print(json.dumps(res, indent=1))


# ── step: figure ────────────────────────────────────────────────────────────────────────────────────────────────────
def silhouette(V, F, view):
    """Outer contours of a mesh projected to the view (holes filled): list of (k, 2) arrays in um. Pinholes between
    the rasterised triangles are closed (5 x 5 px closing) and contours under MIN_CONTOUR_UM2 are dropped."""
    import cv2
    R, U = (np.array(v, float) for v in VIEWS[view])
    P = np.c_[V @ R, V @ U] / 1e3
    lo = P.min(0) - 4
    px = np.round((P - lo) / SIL_RES_UM).astype(np.int32)
    img = np.zeros((px[:, 1].max() + 9, px[:, 0].max() + 9), np.uint8)
    cv2.fillPoly(img, list(px[F]), 255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [cv2.approxPolyDP(c, 1.0, True)[:, 0, :] * SIL_RES_UM + lo for c in cs
            if cv2.contourArea(c) * SIL_RES_UM ** 2 >= MIN_CONTOUR_UM2]


def runs(verts, start, keep):
    """The polyline pieces made of the segments (i-1 -> i) with keep[i]: vertices and start mask."""
    keep = keep & ~start
    first = np.flatnonzero(keep & ~np.r_[False, keep[:-1]]) - 1
    idx = np.sort(np.r_[np.flatnonzero(keep), first])
    return verts[idx], np.isin(idx, first)


def band_alpha(role, band) -> float:
    _, _, a_brain, a_vnc = STYLE[role]
    return a_brain + (a_vnc - a_brain) * band / (N_RAMP + 1)


def load_circuit_lines(y_cut_um):
    """Per role, per neuron and per band (0: brain, y < y_cut; 1..N_RAMP: RAMP_UM / N_RAMP slices behind the cut;
    N_RAMP + 1: the rest of the VNC part): drawing vertices (m, 3) in um and the polyline-start mask. Edges longer
    than MAX_EDGE_UM (straight skeleton-healing jumps) are dropped. Also the neuron count per role and the number of
    dropped edges."""
    circ = pd.read_csv(OUT / "circuit.csv")
    man = pd.read_csv(OUT / "fetch_manifest.csv").fillna({"object": ""})
    man = man[man["kind"].isin(["fafb_banc", "banc"]) & (man["object"] != "")]
    obj = dict(zip(zip(man["role"], man["id"]), man["object"]))

    def one(name):
        v, st = polylines(*read_swc(name), TWIG_UM * 1e3, STEP_UM * 1e3)
        v = v / 1e3
        ok = np.r_[np.inf, np.linalg.norm(np.diff(v, axis=0), axis=1)] <= MAX_EDGE_UM
        y = v[:, 1] - y_cut_um
        band = np.where(y < 0, 0, np.minimum(1 + (np.maximum(y, 0) / RAMP_UM * N_RAMP).astype(int), N_RAMP + 1))
        return {k: runs(v, st, ok & (band == k)) for k in range(N_RAMP + 2)}, int((~st & ~ok).sum())

    lines, n, dropped = {}, {}, 0
    for role in ROLES:
        ids = circ.loc[circ["role"] == role, "id"].astype(np.int64)
        names = [obj[(role, i)] for i in ids if (role, i) in obj]
        with ThreadPoolExecutor(THREADS) as ex:
            parts = list(ex.map(one, names))
        lines[role], n[role] = [p for p, _ in parts], len(names)
        dropped += sum(k for _, k in parts)
    return lines, n, dropped


def font_setup():
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    fam = None
    for want in ("Arial", "Liberation Sans"):
        if any(f.name == want for f in fm.fontManager.ttflist):
            fam = want
            break
    if fam is None:                                                  # the Windows copy of Arial on this machine
        win = Path("/mnt/c/Windows/Fonts")
        for f in ("arial.ttf", "arialbd.ttf"):
            if (win / f).exists():
                fm.fontManager.addfont(str(win / f))
        fam = "Arial" if any(f.name == "Arial" for f in fm.fontManager.ttflist) else None
    if fam is None:
        raise SystemExit("neither Arial nor Liberation Sans is available")
    mpl.rcParams.update({"font.family": fam, "font.size": FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none"})
    return fam


def orientation_key(fig, x, y, view, arm_mm=2.6):
    """Plus-shaped key at figure position (x, y) mm (centre), letters at the four arm ends."""
    from matplotlib.lines import Line2D
    R, U = VIEWS[view]
    ends = {}
    for vec, (dx, dy) in ((R, (1, 0)), (U, (0, 1))):
        i = int(np.flatnonzero(vec)[0])
        neg, pos = AXIS_ENDS[i] if vec[i] > 0 else AXIS_ENDS[i][::-1]
        ends[(dx, dy)], ends[(-dx, -dy)] = pos, neg
    W, H = fig.get_size_inches() / MM
    kw = dict(transform=fig.transFigure, color="black", lw=0.5, solid_capstyle="butt")
    fig.add_artist(Line2D([(x - arm_mm) / W, (x + arm_mm) / W], [y / H, y / H], **kw))
    fig.add_artist(Line2D([x / W, x / W], [(y - arm_mm) / H, (y + arm_mm) / H], **kw))
    off = arm_mm + 1.1
    for (dx, dy), lab in ends.items():
        fig.text((x + dx * off) / W, (y + dy * off) / H, lab, ha="center", va="center", fontsize=FONT_PT - 1)


def key_centre(view, b, pts, contours, s):
    """Centre (u, v) in um of the orientation key: the panel corner whose key box holds the fewest drawn vertices
    (outline area counts as well); ties in the order bottom left, bottom right, top right, top left."""
    from matplotlib.path import Path as MPath
    h = KEY_BOX_MM / s / 2
    cands = [(b[0] + h, b[1] + h), (b[2] - h, b[1] + h), (b[2] - h, b[3] - h), (b[0] + h, b[3] - h)]
    g = np.stack(np.meshgrid(np.linspace(-h, h, 7), np.linspace(-h, h, 7)), -1).reshape(-1, 2)
    paths = [MPath(c) for c in contours]
    score = []
    for cu, cv in cands:
        ink = int((np.abs(pts - (cu, cv)) <= h).all(1).sum())
        grey = sum(int(pa.contains_points(g + (cu, cv)).sum()) for pa in paths)
        score.append(ink + 10 * grey)
    return cands[int(np.argmin(score))]


def render(lines, meshes, paths: dict, dpi: int) -> dict:
    """Draw panels a-d into the files in `paths` (extension -> path); returns layout numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    sil, depth, uv, bounds = {}, {}, {}, {}
    for view, (R, U) in VIEWS.items():
        R, U = np.array(R, float), np.array(U, float)
        toward = np.cross(R, U)
        sil[view] = {part: silhouette(V, F, view) for part, (V, F) in meshes.items()}
        depth[view] = {part: float((V @ toward).mean()) for part, (V, _) in meshes.items()}
        uv[view] = {role: [{part: (np.c_[v @ R, v @ U], st) for part, (v, st) in nr.items()} for nr in lines[role]]
                    for role in ROLES}
        pts = [np.concatenate(c) for c in sil[view].values()]
        pts += [p for role in ROLES for nr in uv[view][role] for p, _ in nr.values() if len(p)]
        pts = np.concatenate(pts)
        bounds[view] = np.r_[pts.min(0) - PAD_UM, pts.max(0) + PAD_UM]      # umin, vmin, umax, vmax
        uv[view]["_pts"] = pts[::5]
    # layout: column 1 = a (frontal) over d (posterior); column 2 = b (dorsal); column 3 = c (lateral); b, c share v
    for v in ("dorsal", "lateral"):
        bounds[v][[1, 3]] = [min(bounds["dorsal"][1], bounds["lateral"][1]),
                             max(bounds["dorsal"][3], bounds["lateral"][3])]
    w = {v: bounds[v][2] - bounds[v][0] for v in VIEWS}
    h = {v: bounds[v][3] - bounds[v][1] for v in VIEWS}
    col1 = max(w["frontal"], w["posterior"])
    top_mm = 3.5
    s = (FIG_W_MM - 2 * MARGIN_MM - 2 * GAP_MM) / (col1 + w["dorsal"] + w["lateral"])   # mm per um
    H_MM = top_mm + max(h["dorsal"] * s, (h["frontal"] + h["posterior"]) * s + GAP_MM + top_mm) + MARGIN_MM
    x0 = {"frontal": MARGIN_MM, "posterior": MARGIN_MM, "dorsal": MARGIN_MM + col1 * s + GAP_MM,
          "lateral": MARGIN_MM + (col1 + w["dorsal"]) * s + 2 * GAP_MM}
    ytop = {"frontal": H_MM - top_mm, "dorsal": H_MM - top_mm, "lateral": H_MM - top_mm,
            "posterior": H_MM - top_mm - h["frontal"] * s - GAP_MM - top_mm}

    fig = plt.figure(figsize=(FIG_W_MM * MM, H_MM * MM))
    fig.patch.set_facecolor("white")
    fill = to_rgba(OUTLINE_FILL, OUTLINE_ALPHA)
    keys = {}
    for letter, view in zip("abcd", VIEWS):
        wi, hi = w[view] * s, h[view] * s
        ax = fig.add_axes([x0[view] / FIG_W_MM, (ytop[view] - hi) / H_MM, wi / FIG_W_MM, hi / H_MM])
        ax.set_xlim(bounds[view][0], bounds[view][2])
        ax.set_ylim(bounds[view][1], bounds[view][3])
        ax.set_axis_off()
        for part in sorted(depth[view], key=depth[view].get):          # far part first: outline, then neurons
            cs = sil[view][part]
            verts = np.concatenate([np.r_[c, c[:1]] for c in cs])
            codes = np.concatenate([[MPath.MOVETO] + [MPath.LINETO] * (len(c) - 1) + [MPath.CLOSEPOLY] for c in cs])
            ax.add_artist(PathPatch(MPath(verts, codes), facecolor=fill, edgecolor=OUTLINE_EDGE, lw=OUTLINE_LW))
            for role in DRAW_ORDER:                                  # add_artist: no per-vertex limit update
                col, lw = STYLE[role][:2]
                for nr in uv[view][role]:
                    for band in PART_BANDS[part]:
                        p, st = nr[band]
                        if len(p) < 2:
                            continue
                        codes = np.where(st, MPath.MOVETO, MPath.LINETO).astype(np.uint8)
                        ax.add_artist(PathPatch(MPath(p, codes), fill=False, edgecolor=col, lw=lw,
                                                alpha=band_alpha(role, band), capstyle="round", joinstyle="round"))
        fig.text((x0[view] - 0.2) / FIG_W_MM, (ytop[view] + 1.2) / H_MM, letter, fontsize=LETTER_PT,
                 fontweight="bold", ha="left", va="bottom")
        ku, kv = key_centre(view, bounds[view], uv[view]["_pts"], [c for cs in sil[view].values() for c in cs], s)
        keys[view] = (ku, kv)
        orientation_key(fig, x0[view] + (ku - bounds[view][0]) * s, ytop[view] - hi + (kv - bounds[view][1]) * s,
                        view)
    # scale bar: below panel d
    sb_y = ytop["posterior"] - h["posterior"] * s - 4.0
    fig.add_artist(Line2D([MARGIN_MM / FIG_W_MM, (MARGIN_MM + SCALE_UM * s) / FIG_W_MM], [sb_y / H_MM] * 2,
                          transform=fig.transFigure, color="black", lw=1.0, solid_capstyle="butt"))
    fig.text((MARGIN_MM + SCALE_UM * s / 2) / FIG_W_MM, (sb_y - 1.0) / H_MM, f"{SCALE_UM} \u00b5m", ha="center",
             va="top", fontsize=FONT_PT)
    t0 = time.time()
    for ext, path in paths.items():
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return dict(size_mm=[FIG_W_MM, round(H_MM, 1)], scale_mm_per_100um=round(100 * s, 2),
                save_s=round(time.time() - t0, 1), bytes={ext: Path(p).stat().st_size for ext, p in paths.items()})


def outline_meshes(fetch) -> tuple:
    """{'brain', 'vnc'}: (vertices nm, faces) and the neck cut in um (the posterior end of the brain outline)."""
    meshes = {part: read_mesh(fetch["segment_ids"][lab]) for part, lab in zip(("brain", "vnc"), OUTLINES)}
    return meshes, float(meshes["brain"][0][:, 1].max() / 1e3)


def step_figure(a):
    fam = font_setup()
    fetch = json.loads((OUT / "fetch.json").read_text())
    meshes, y_cut = outline_meshes(fetch)
    lines, n_drawn, dropped = load_circuit_lines(y_cut)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    info = render(lines, meshes, {"pdf": FIG_DIR / f"{FIG_NAME}.pdf", "png": FIG_DIR / f"{FIG_NAME}.png"}, DPI)
    info.update(font=fam, neck_cut_y_um=round(y_cut, 1), neurons_drawn=n_drawn, long_edges_dropped=dropped,
                vertices={r: int(sum(len(p) for nr in v for p, _ in nr.values())) for r, v in lines.items()})
    (FIG_DIR / "caption.txt").write_text(caption(n_drawn, fetch, dropped), encoding="utf-8")
    (OUT / "figure.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def caption(n, fetch, dropped) -> str:
    sw = fetch["swc"]
    banc = {r: sw.get(f"{r}/banc", {}) for r in ("banc_dn", "interneuron", "mn")}
    sk = sum(v.get("skeleton", 0) for v in banc.values())
    l2 = sum(v.get("l2", 0) for v in banc.values())
    ids = json.loads((OUT / "ids.json").read_text())
    lines = [
        "Stage-A circuit of the brain-to-nerve-cord model in BANC space.",
        "",
        "a-d, Frontal (a), dorsal (b), lateral (c, from the fly's left) and posterior (d) orthographic views at the "
        f"same scale. Lines: {n['fw_dn']} FlyWire descending neurons (DNs) that drive the model's DN bridge (sky "
        "blue; FlyWire v783 skeletons transformed into BANC space by the BANC team, FAFB14 -> JRC2018F -> mirror "
        "(FlyWire convention) -> BANC, elastix and thin-plate spline; brain portion only); "
        f"{n['banc_dn']} BANC DNs that receive the bridge input (blue); {n['interneuron']} VNC interneurons on the "
        "strongest two-hop paths from these DNs to the steering motor neurons, ranked by the summed synapse count "
        f"of the path in the model's nerve-cord graph ({ids['interneurons']['eligible']} eligible, at most "
        f"{N_INTERNEURONS} shown; orange); and the {n['mn']} wing steering motor neurons read out by the model "
        "(b2, b3, i1, i2, hg1-hg4, left and right; BANC types iv1-iv4 for hg1-hg4; vermillion). BANC skeletons are "
        f"full-resolution skeletor skeletons where available ({sk}) and L2 skeletons otherwise ({l2}); for drawing, "
        f"terminal twigs shorter than {TWIG_UM:g} um are removed, nodes are kept >= {STEP_UM:g} um apart along "
        f"the cable and {dropped} straight edges longer than {MAX_EDGE_UM:g} um (skeleton-healing jumps) are "
        "omitted. Grey: BANC brain and VNC neuropil outlines (region_outlines, alpha shapes of synapse density), "
        "semi-opaque; each neuron is split at the posterior end of the brain outline and brain and VNC are drawn "
        "far to near, so in a and d the nearer region dims the farther one. BANC DNs are drawn faint inside the "
        f"brain, where the FlyWire DNs of the same types run (alpha ramp over {RAMP_UM:g} um behind the split). "
        "Fly's own axes: D, dorsal; V, ventral; L, left; R, "
        "right; A, anterior; P, posterior. Scale bar, 100 um. Colours: Okabe-Ito palette. Demonstration only; not "
        "an experiment and no claim.",
        "",
        "Data: FlyWire v783 connectome of the female adult fly brain (Dorkenwald et al. 2024; Schlegel et al. 2024; "
        "EM: Zheng et al. 2018), CC BY-NC 4.0. BANC brain-and-nerve-cord connectome, materialisation v888 (Bates, "
        f"Phelps, Kim, Yang et al. 2026, Nature 656:957-970), CC BY 4.0, Harvard Dataverse doi:{DOI}; skeletons, "
        "warped FlyWire skeletons and region outlines from gs://lee-lab_brain-and-nerve-cord-fly-connectome "
        "(compiled_data/, region_outlines/). This figure combines both sources and carries the more restrictive "
        "licence, CC BY-NC 4.0."]
    return "\n".join(lines).replace(" um", " \u00b5m") + "\n"


STEPS = {"ids": step_ids, "fetch": step_fetch, "check": step_check, "figure": step_figure}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    steps = list(STEPS) if a.step == "all" else [a.step]
    existing = [str(OUT / f) for st in steps for f in FILES.get(st, ()) if (OUT / f).exists()]
    if "figure" in steps:
        existing += [str(FIG_DIR / f) for f in FIG_FILES if (FIG_DIR / f).exists()]
    if existing and not a.overwrite:
        raise SystemExit(f"{existing} exist; pass --overwrite")
    for st in steps:
        STEPS[st](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
