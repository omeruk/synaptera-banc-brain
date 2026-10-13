"""BANC adapter (FlyWire Codex export of the Brain And Nerve Cord connectome): neurons -> the common schema.

New file (no v1 source). F1 version: neuron table, DN list, edges (one row per neuron pair, synapse counts of all neuropils summed),
regions from the neuropil of the synapses, and the sign rule of ``prereg/01_banc_sign_rule.md``: reference =
the Shiu rule (GABA and glutamate -1, everything else incl. HIST, TYR and a missing prediction +1), variants
``hist_inhibitory`` / ``nt_impute`` by flag only (``connectome.sign_rules.assign_signs``).

Column mapping (Codex ``neurons-banc.csv.gz``): Root ID -> id, Primary Cell Type -> cell_type, Alternative
Cell Type(s) -> cell_type_alt, Class -> cell_class, Super Class -> super_class, Soma side -> side,
Predicted NT type / confidence -> nt / nt_conf. ``region`` = region of the neuron's dominant neuropil
(synapse-weighted, in + out): see :func:`region_of_neuropil`.
Extra columns kept for later selections: sub_class, body_part, function, flow, verified_nt, nerve, hemilineage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from connectome import schema
from connectome.schema import NEURON_COLUMNS, normalise_side

DN_SUPER_CLASS = "descending"
_RENAME = {"Root ID": "id", "Primary Cell Type": "cell_type", "Alternative Cell Type(s)": "cell_type_alt",
           "Class": "cell_class", "Super Class": "super_class", "Predicted NT type": "nt",
           "Predicted NT confidence": "nt_conf"}
_EXTRA = {"Sub Class": "sub_class", "Body Part": "body_part", "Function": "function", "Flow": "flow",
          "Verified NT type": "verified_nt", "Nerve": "nerve", "Hemilineage": "hemilineage"}


def load_neurons(path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({"id": raw["Root ID"].to_numpy(np.int64)})
    for src, dst in _RENAME.items():
        if dst != "id":
            out[dst] = raw[src].to_numpy()
    out["side"] = normalise_side(raw["Soma side"].to_numpy()).to_numpy()
    out["region"] = np.nan
    for c in ("soma_x", "soma_y", "soma_z"):
        out[c] = np.nan            # not in the Codex export
    out["status"] = np.nan
    for src, dst in _EXTRA.items():
        out[dst] = raw[src].to_numpy()
    return out[list(NEURON_COLUMNS) + list(_EXTRA.values())]


def descending_neurons(neurons: pd.DataFrame) -> pd.DataFrame:
    """BANC descending neurons (Super Class "descending") in table order, with the same columns as the
    FlyWire view: root_id, cell_type, top_nt (here the predicted NT), side."""
    dn = schema.descending_neurons(neurons, DN_SUPER_CLASS)
    return pd.DataFrame({"root_id": dn["id"].to_numpy(), "cell_type": dn["cell_type"].to_numpy(),
                         "top_nt": dn["nt"].to_numpy(), "side": dn["side"].to_numpy()})


def load_edges_unsigned(path) -> pd.DataFrame:
    """Codex ``connections_princeton-banc.csv.gz`` aggregated to one row per neuron pair (synapse counts of all
    neuropils summed). No ``sign`` column: the transmitter -> sign rule is an F1 decision."""
    c = pd.read_csv(path, usecols=["pre_root_id", "post_root_id", "syn_count"])
    g = c.groupby(["pre_root_id", "post_root_id"], sort=True)["syn_count"].sum().reset_index()
    return pd.DataFrame({"pre": g["pre_root_id"].to_numpy(np.int64), "post": g["post_root_id"].to_numpy(np.int64),
                         "n_syn": g["syn_count"].to_numpy(np.int64)})


def load_connectome_unsigned(neurons_path, connections_path, hash_files: bool = True) -> schema.Connectome:
    """BANC as an *unsigned* Connectome (anatomy only: profiles, degrees, crosswalk evidence)."""
    from connectome.adapters import file_sha256
    meta = dict(dataset="banc", sign_rule=None, signed=False, neurons=str(neurons_path),
                connections=str(connections_path))
    if hash_files:
        meta["sha256"] = {"neurons": file_sha256(neurons_path), "connections": file_sha256(connections_path)}
    return schema.Connectome("banc", load_neurons(neurons_path), load_edges_unsigned(connections_path), meta)


# ── regions ──────────────────────────────────────────────────────────────────
_VNC_NAMES = ("IntTct", "LTct", "ANm", "Tct")
_SEZ = ("GNG", "PRW", "SAD")
OPTIC_NEUROPILS = ("ME", "LO", "LOP", "LA", "AME")       # optic lobe: a subset of "brain" (QA reports it separately)


def region_of_neuropil(name: str) -> str:
    """Region (schema REGIONS) of a BANC neuropil name.

    VNC: ``VNC_*``, ``ABDNM_*``, ``LegNp_*``, ``*_UTct_*``, IntTct, LTct, ANm, Tct. SEZ: GNG, PRW, SAD, FLA_*, CAN_*.
    neck: cervical_connective. brain: everything else (the optic lobes ME / LO / LOP / LA / AME included)."""
    if name.startswith(("VNC_", "ABDNM_", "LegNp_")) or "_UTct_" in name or name in _VNC_NAMES:
        return "VNC"
    if name in _SEZ or name.startswith(("FLA_", "CAN_")):
        return "SEZ"
    if name == "cervical_connective":
        return "neck"
    return "brain"


def is_optic(name: str) -> bool:
    return name.split("_")[0] in OPTIC_NEUROPILS


def load_connections(path) -> pd.DataFrame:
    """Per-neuropil connection rows (pre_root_id, post_root_id, neuropil, syn_count)."""
    return pd.read_csv(path, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])


def dominant_regions(conn: pd.DataFrame, ids: np.ndarray) -> pd.Series:
    """Region of the neuropil holding most of each neuron's synapses (in + out); NaN without connections."""
    reg = conn["neuropil"].map({n: region_of_neuropil(n) for n in conn["neuropil"].unique()})
    w = conn["syn_count"].to_numpy(np.int64)
    both = pd.concat([pd.DataFrame({"id": conn["pre_root_id"].to_numpy(), "r": reg.to_numpy(), "w": w}),
                      pd.DataFrame({"id": conn["post_root_id"].to_numpy(), "r": reg.to_numpy(), "w": w})])
    g = both.groupby(["id", "r"])["w"].sum().reset_index().sort_values(["id", "w"], ascending=[True, False])
    top = g.drop_duplicates("id").set_index("id")["r"]
    return top.reindex(ids)


def load_connectome(neurons_path, connections_path, variants=(), hash_files: bool = True,
                    min_syn: int | None = None) -> schema.Connectome:
    """Signed BANC Connectome: reference Shiu rule (+ named variants), regions filled from the synapses.

    ``min_syn``: keep only neuron pairs with >= min_syn synapses (neuropils summed). Default None = the graph as
    exported (the export keeps pairs with >= 3 synapses). The threshold and the pairs before / after go to the meta.
    The Shiu FlyWire graph is not thresholded (min 1 synapse per pair), so a threshold is a named sensitivity
    option, not part of the reference arm."""
    from connectome.adapters import file_sha256
    from connectome.sign_rules import assign_signs
    nrn = load_neurons(neurons_path)
    conn = load_connections(connections_path)
    nrn["region"] = dominant_regions(conn, nrn["id"].to_numpy()).to_numpy(object)
    g = conn.groupby(["pre_root_id", "post_root_id"], sort=True)["syn_count"].sum().reset_index()
    edges = pd.DataFrame({"pre": g["pre_root_id"].to_numpy(np.int64), "post": g["post_root_id"].to_numpy(np.int64),
                          "n_syn": g["syn_count"].to_numpy(np.int64)})
    n_pairs_all = len(edges)
    if min_syn is not None:
        edges = edges[edges["n_syn"] >= int(min_syn)].reset_index(drop=True)
    meta = dict(dataset="banc", neurons=str(neurons_path), connections=str(connections_path),
                min_syn=None if min_syn is None else int(min_syn), pairs_before_threshold=int(n_pairs_all),
                pairs_kept=int(len(edges)), synapses_kept=int(edges["n_syn"].sum()))
    if hash_files:
        meta["sha256"] = {"neurons": file_sha256(neurons_path), "connections": file_sha256(connections_path)}
    unsigned = schema.Connectome("banc", nrn, edges, dict(meta, signed=False))
    return assign_signs(unsigned, variants)


# ── full materialization-888 graph (Harvard Dataverse doi:10.7910/DVN/7WTH1N), data source "banc888_full" ──────
NEURON_SETS = ("codex", "all")
_META_NT = {"acetylcholine": "ACH", "glutamate": "GLUT", "gaba": "GABA", "dopamine": "DA", "histamine": "HIST",
            "octopamine": "OCT", "serotonin": "SER", "tyramine": "TYR"}


def load_edgelist_888(path, ids: np.ndarray | None = None) -> pd.DataFrame:
    """``banc_888_edgelist_simple_v2.feather`` as (pre, post, n_syn): one row per directed neuron pair, no count
    threshold, autapses excluded at build time. Ids are strings in the file (int64 range) and converted. ``ids``:
    keep only pairs whose both endpoints are in ``ids``."""
    e = pd.read_feather(path, columns=["pre", "post", "count"])
    out = pd.DataFrame({"pre": e["pre"].to_numpy().astype(np.int64), "post": e["post"].to_numpy().astype(np.int64),
                        "n_syn": e["count"].to_numpy(np.int64)})
    if ids is not None:
        k = np.isin(out["pre"].to_numpy(), ids) & np.isin(out["post"].to_numpy(), ids)
        out = out[k].reset_index(drop=True)
    return out.sort_values(["pre", "post"], kind="stable").reset_index(drop=True)


def _neurons_from_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Neuron rows of ``banc_888_meta.feather`` on the adapter's columns (used for neurons outside the Codex table)."""
    n = len(meta)
    nt = meta["neurotransmitter_predicted"].map(_META_NT)
    out = pd.DataFrame({"id": meta["banc_888_id"].to_numpy().astype(np.int64),
                        "cell_type": meta["cell_type"].to_numpy(object),
                        "cell_type_alt": np.full(n, np.nan, object),
                        "cell_class": meta["cell_class"].to_numpy(object),
                        "super_class": meta["super_class"].to_numpy(object),
                        "side": normalise_side(meta["side"].to_numpy()).to_numpy(),
                        "region": np.full(n, np.nan, object),
                        "nt": nt.to_numpy(object),
                        "nt_conf": pd.to_numeric(meta["neurotransmitter_score"], errors="coerce").to_numpy(float)})
    for c in ("soma_x", "soma_y", "soma_z"):
        out[c] = np.nan
    out["status"] = np.nan
    out["sub_class"] = meta["cell_sub_class"].to_numpy(object)
    out["body_part"] = meta["body_part_sensory"].to_numpy(object)
    out["function"] = meta["cell_function"].to_numpy(object)
    out["flow"] = meta["flow"].to_numpy(object)
    out["verified_nt"] = meta["neurotransmitter_verified"].to_numpy(object)
    out["nerve"] = meta["nerve"].to_numpy(object)
    out["hemilineage"] = meta["hemilineage"].to_numpy(object)
    return out[list(NEURON_COLUMNS) + list(_EXTRA.values())]


def load_connectome_full(neurons_path, edgelist_path, meta_path, connections_path=None, neuron_set: str = "codex",
                         variants=(), hash_files: bool = True,
                         data_source: str = "banc888_full") -> schema.Connectome:
    """Signed BANC Connectome from the full v888 pair-level edge list (no synapse threshold; data source
    ``banc888_full`` (v2 edge list) or ``banc888_full_v3`` (v3 edge list) in ``configs/data_sources.yaml``; the
    label is only recorded in the graph meta).

    ``neuron_set``: "codex" (primary; the 158,262 neurons of the Codex neuron table, edges among them) or "all" (every
    row of ``banc_888_meta.feather``; the Codex table annotates its neurons, the meta table the rest, NT names mapped
    by ``_META_NT``). The Codex Root IDs are the v888 ids (all 158,262 are in the meta table). The neuron table
    order of the Codex table is kept, so engine indices of the Codex neurons are identical in both sets.
    ``connections_path``: the Codex per-neuropil table, used only for the ``region`` annotation of the Codex neurons."""
    from connectome.adapters import file_sha256
    from connectome.sign_rules import assign_signs
    if neuron_set not in NEURON_SETS:
        raise ValueError(f"neuron_set must be one of {NEURON_SETS}")
    nrn = load_neurons(neurons_path)
    if connections_path is not None:
        conn = load_connections(connections_path)
        nrn["region"] = dominant_regions(conn, nrn["id"].to_numpy()).to_numpy(object)
        del conn
    meta = pd.read_feather(meta_path)
    n_meta = len(meta)
    if neuron_set == "all":
        extra = meta[~np.isin(meta["banc_888_id"].to_numpy().astype(np.int64), nrn["id"].to_numpy())]
        nrn = pd.concat([nrn, _neurons_from_meta(extra)], ignore_index=True)
    edges = load_edgelist_888(edgelist_path, None if neuron_set == "all" else nrn["id"].to_numpy())
    info = dict(dataset="banc", data_source=data_source, neuron_set=neuron_set, neurons=str(neurons_path),
                edgelist=str(edgelist_path), meta_table=str(meta_path), meta_rows=int(n_meta),
                min_syn=None, pairs_kept=int(len(edges)), synapses_kept=int(edges["n_syn"].sum()))
    if hash_files:
        info["sha256"] = {"neurons": file_sha256(neurons_path), "edgelist": file_sha256(edgelist_path),
                          "meta": file_sha256(meta_path)}
    unsigned = schema.Connectome("banc", nrn, edges, dict(info, signed=False))
    return assign_signs(unsigned, variants)
