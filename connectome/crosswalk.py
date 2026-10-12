"""Type crosswalk between datasets (FlyWire v783 <-> BANC): which neurons of each dataset stand for the same
named type, with the evidence source recorded per row.

Generalises synaptera@c94796d:scripts/make_sugar_grn.py (homolog selection for one input set) to a
crosswalk of every critical type of docs/05_ARCHITECTURE_PLAN.md F1. Evidence sources:
  name       the primary cell_type matches the dataset's pattern
  name_alt   only the alternative cell type matches
  filter     selected by annotation columns (e.g. motor neurons of body part "wing"); no name involved
Neuron-level cross-dataset matching (NBLAST, connectivity) is added as further ``source`` values later;
:func:`homologs.cross_dataset_cosine` gives connectivity evidence for a candidate pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CROSSWALK_COLUMNS = ("query", "dataset", "id", "cell_type", "side", "source", "note")


@dataclass(frozen=True)
class TypeQuery:
    """One crosswalk entry. ``patterns``: dataset -> regex on the type names ("*" = every dataset);
    ``where``: annotation column -> required value, or a tuple of accepted values (needs that column in the
    dataset's neurons table)."""
    name: str
    patterns: dict = field(default_factory=dict)
    where: dict = field(default_factory=dict)
    note: str = ""


# Critical types of F1 (docs/05_ARCHITECTURE_PLAN.md section 9, step 4). BANC subtypes of LB3 are LB3a..d.
CRITICAL_QUERIES = (
    TypeQuery("DNp01", {"*": r"^DNp01$"}, note="giant fibre"),
    TypeQuery("DNg02", {"*": r"^DNg02(?:_|$)"}, note="DNg02 family incl. DNg02_a..h"),
    TypeQuery("DNa01", {"*": r"^DNa01$"}),
    TypeQuery("DNa02", {"*": r"^DNa02$"}),
    TypeQuery("DNp15", {"*": r"^DNp15$"}),
    TypeQuery("DNp07", {"*": r"^DNp07$"}),
    TypeQuery("DNp10", {"*": r"^DNp10$"}),
    TypeQuery("MN9", {"flywire783": r"^CB0701$", "banc": r"^MN9$"},
              note="FlyWire names the proboscis MN9 CB0701 (Shiu et al. 2023); to be confirmed by connectivity"),
    TypeQuery("LB3", {"flywire783": r"^LB3$", "banc": r"^LB3[a-z]?$"}, note="labellar sugar GRN candidates"),
    TypeQuery("TTMn", {"*": r"^TTMn$"}),
    TypeQuery("wing_MN", where={"super_class": "motor", "body_part": "wing"}, note="BANC annotation filter"),
    TypeQuery("haltere_MN", where={"super_class": "motor", "body_part": "haltere"}, note="BANC annotation filter"),
    TypeQuery("haltere_afferent", where={"super_class": ("sensory", "sensory_ascending"), "body_part": "haltere"},
              note="BANC annotation filter"),
)


def _pattern_for(query: TypeQuery, dataset: str):
    return query.patterns.get(dataset, query.patterns.get("*"))


def crosswalk_by_name(datasets: dict, queries=CRITICAL_QUERIES) -> pd.DataFrame:
    """Neuron-level crosswalk table (CROSSWALK_COLUMNS) for ``datasets`` = {name: neurons DataFrame}.

    A query with no pattern for a dataset and no usable ``where`` columns yields no rows for it (the column
    is simply absent, e.g. ``body_part`` in the brain-only FlyWire table)."""
    rows = []
    for ds, nrn in datasets.items():
        ct = nrn["cell_type"].astype(str)
        alt = nrn["cell_type_alt"].astype(str)
        for q in queries:
            pat = _pattern_for(q, ds)
            hit = pd.Series("", index=nrn.index, dtype=object)
            if pat is not None:
                by_name = ct.str.contains(pat, regex=True)
                by_alt = alt.str.contains(pat, regex=True) & ~by_name
                hit[by_name] = "name"
                hit[by_alt] = "name_alt"
            if q.where and all(c in nrn.columns for c in q.where):
                m = np.ones(len(nrn), bool)
                for c, v in q.where.items():
                    m &= (nrn[c].isin(v) if isinstance(v, tuple) else nrn[c] == v).to_numpy()
                if pat is not None:
                    m &= hit.ne("").to_numpy()      # pattern and filter together narrow the selection
                    hit[~m] = ""
                else:
                    hit[m] = "filter"
            sel = nrn.loc[hit.ne("")]
            for (_, r), src in zip(sel.iterrows(), hit[hit.ne("")]):
                rows.append((q.name, ds, int(r["id"]), r["cell_type"], r["side"], src, q.note))
    return pd.DataFrame(rows, columns=list(CROSSWALK_COLUMNS))


def crosswalk_summary(cw: pd.DataFrame) -> pd.DataFrame:
    """Counts per (query, dataset): neurons, left, right, other side, and the evidence sources."""
    g = cw.groupby(["query", "dataset"])
    out = g.agg(n=("id", "size"),
                n_left=("side", lambda s: int((s == "left").sum())),
                n_right=("side", lambda s: int((s == "right").sum())),
                n_other=("side", lambda s: int((~s.isin(["left", "right"])).sum())),
                types=("cell_type", lambda s: ",".join(sorted(set(s.astype(str))))[:80]),
                sources=("source", lambda s: ",".join(sorted(set(s)))))
    return out.reset_index()
