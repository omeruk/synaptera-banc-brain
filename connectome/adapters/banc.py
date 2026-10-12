"""BANC adapter (FlyWire Codex export of the Brain And Nerve Cord connectome): neurons -> the common schema.

New file (no v1 source). Scope of this version: the neuron table and the descending-neuron (DN) list, the
equivalent of the FlyWire adapter's DN view. Edges and the sign rule (transmitter-to-sign mapping, including
HIST and neurons without a prediction) are decided in phase F1 and are deliberately not implemented here.

Column mapping (Codex ``neurons-banc.csv.gz``): Root ID -> id, Primary Cell Type -> cell_type, Alternative
Cell Type(s) -> cell_type_alt, Class -> cell_class, Super Class -> super_class, Soma side -> side,
Predicted NT type / confidence -> nt / nt_conf. ``region`` is left unassigned (NaN) until F1 defines it.
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
