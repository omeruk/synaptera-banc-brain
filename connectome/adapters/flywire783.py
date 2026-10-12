"""FlyWire v783 adapter: the Shiu et al. 2023 model files + FlyWire annotations -> the common schema.

Ported from synaptera@c94796d:flight/groups.py (paths, load_root_ids, SHIU_SUGAR) and the loading part of
scripts/make_descending_neurons.py. Row order of the neurons table is the row order of Completeness_783.csv,
which is the Brian2 neuron index of the Shiu model; the edge ``sign`` is the Shiu rule (the ``Excitatory``
column: every transmitter other than GABA / glutamate is +1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from connectome.adapters import file_sha256
from connectome.schema import NEURON_COLUMNS, Connectome, normalise_side

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DIR = REPO_ROOT / "brain_model"
FILES = dict(completeness="Completeness_783.csv", connectivity="Connectivity_783.parquet",
             annotations="flywire_annotations.tsv")

# Shiu et al. 2023 sugar GRNs (their figures.ipynb, v630 root ids updated to v783). 720575940620900446 is not
# in v783 (20 left-side LB3 remain).
SHIU_SUGAR_IDS = (720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                  720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                  720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                  720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                  720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                  720575940611875570)

_ANN_COLUMNS = ["root_id", "cell_type", "hemibrain_type", "cell_class", "super_class", "side", "soma_x", "soma_y",
                "soma_z", "status", "top_nt", "known_nt", "known_nt_source"]
_EXTRA_COLUMNS = ("top_nt", "known_nt", "known_nt_source")


def load_root_ids(brain_dir=BRAIN_DIR) -> np.ndarray:
    """root_id of each Completeness row; row position = Brian neuron index."""
    return pd.read_csv(Path(brain_dir) / FILES["completeness"], index_col=0).index.to_numpy(dtype=np.int64)


def load_codex_nt(path) -> pd.DataFrame:
    """Codex v783 ``neurons.csv.gz``: root_id -> (nt, nt_conf). nt is the Codex ``nt_type`` (ACH, GABA, GLUT,
    DA, SER, OCT; NaN where there is no prediction, then nt_conf is 0)."""
    t = pd.read_csv(path, usecols=["root_id", "nt_type", "nt_type_score"]).set_index("root_id")
    return t.rename(columns={"nt_type": "nt", "nt_type_score": "nt_conf"})


def load_neurons(brain_dir=BRAIN_DIR, codex_neurons=None) -> pd.DataFrame:
    """Neurons table in Completeness row order. ``codex_neurons``: optional path of Codex neurons.csv.gz
    (fills ``nt`` / ``nt_conf``; they stay NaN without it)."""
    brain_dir = Path(brain_dir)
    ids = load_root_ids(brain_dir)
    ann = pd.read_csv(brain_dir / FILES["annotations"], sep="\t", low_memory=False, usecols=_ANN_COLUMNS)
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    out = pd.DataFrame({"id": ids})
    out["cell_type"] = ann["cell_type"].to_numpy(object)
    out["cell_type_alt"] = ann["hemibrain_type"].to_numpy(object)
    out["cell_class"] = ann["cell_class"].to_numpy(object)
    out["super_class"] = ann["super_class"].to_numpy(object)
    out["side"] = normalise_side(ann["side"].to_numpy()).to_numpy()
    out["region"] = "brain"
    if codex_neurons is not None:
        nt = load_codex_nt(codex_neurons).reindex(ids)
        out["nt"], out["nt_conf"] = nt["nt"].to_numpy(object), nt["nt_conf"].to_numpy(float)
    else:
        out["nt"], out["nt_conf"] = np.nan, np.nan
    for c in ("soma_x", "soma_y", "soma_z"):
        out[c] = ann[c].to_numpy(float)
    out["status"] = ann["status"].to_numpy(object)
    for c in _EXTRA_COLUMNS:
        out[c] = ann[c].to_numpy(object)
    return out[list(NEURON_COLUMNS) + list(_EXTRA_COLUMNS)]


def load_edges(brain_dir=BRAIN_DIR) -> pd.DataFrame:
    """Edges on the common schema; sign = the Shiu rule (parquet ``Excitatory``)."""
    con = pd.read_parquet(Path(brain_dir) / FILES["connectivity"],
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    return pd.DataFrame({"pre": con["Presynaptic_ID"].to_numpy(np.int64),
                         "post": con["Postsynaptic_ID"].to_numpy(np.int64),
                         "n_syn": con["Connectivity"].to_numpy(np.int64),
                         "sign": con["Excitatory"].to_numpy(np.int64)})


def load_connectome(brain_dir=BRAIN_DIR, codex_neurons=None, hash_files: bool = True) -> Connectome:
    brain_dir = Path(brain_dir)
    meta = dict(dataset="flywire783", sign_rule="shiu2023", brain_dir=str(brain_dir))
    if hash_files:
        meta["sha256"] = {k: file_sha256(brain_dir / v) for k, v in FILES.items()}
    cx = Connectome("flywire783", load_neurons(brain_dir, codex_neurons), load_edges(brain_dir), meta)
    cx.meta["n_without_annotation"] = int(cx.neurons["super_class"].isna().sum())
    return cx
