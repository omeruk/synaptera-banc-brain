"""Small synthetic connectomes for unit tests (no data files)."""
import numpy as np
import pandas as pd

from connectome.schema import NEURON_COLUMNS, Connectome


def neurons_table(rows):
    """rows: list of dicts with at least id / cell_type / side; the rest of the schema is filled with NaN."""
    df = pd.DataFrame(rows)
    for c in NEURON_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df["side"] = df["side"].fillna("unknown")
    return df[list(NEURON_COLUMNS)]


def edges_table(triples, sign=1):
    pre, post, n = zip(*triples)
    return pd.DataFrame({"pre": pre, "post": post, "n_syn": n, "sign": sign})


def lb3_like(n_left=6, n_pos=3, n_right=3):
    """Candidates 'X': left ones with partner type A (positives) or B (negatives); right ones follow A or B.
    Partners: A and B neurons on both sides. Returns (Connectome, positive ids, right ids following A)."""
    rows, tr = [], []
    nid = 100
    partners = {}
    for t in ("A", "B"):
        for s in ("left", "right"):
            partners[(t, s)] = nid
            rows.append(dict(id=nid, cell_type=t, side=s))
            nid += 1
    pos, right_a = [], []
    for i in range(n_left):
        rows.append(dict(id=i, cell_type="X", side="left"))
        is_pos = i < n_pos
        (pos.append(i) if is_pos else None)
        tr.append((i, partners[("A" if is_pos else "B", "left")], 5 + i % 2))
    for j in range(n_right):
        rid = 50 + j
        rows.append(dict(id=rid, cell_type="X", side="right"))
        follows_a = j % 2 == 0
        (right_a.append(rid) if follows_a else None)
        tr.append((rid, partners[("A" if follows_a else "B", "right")], 4))
    return Connectome("toy", neurons_table(rows), edges_table(tr)), pos, right_a
