"""Graph controls for F5 (prereg/14 section 5, arm `shuffled`; docs/10 section 5).

`degree_preserving_shuffle`: directed double-edge swaps (Maslov-Sneppen). Two edges a->b and c->d become a->d and c->b.
A swap is rejected if it would create a self-loop or a pair that already exists. Each edge keeps its index, its
presynaptic neuron and its synapse count (so the sign stays with the presynaptic neuron), and only its postsynaptic
end moves. In- and out-degree of every neuron are therefore unchanged. The procedure runs until `factor` x E swaps
have been accepted.

Random numbers: numpy `default_rng(seed)` draws the candidate edge pairs in chunks of `CHUNK` pairs
(`rng.integers(0, E, size=(CHUNK, 2))`); the pairs are tried in order and the chunk's unused tail is discarded when the
target is reached. The swap loop itself is compiled with numba and uses no random numbers.
"""
from __future__ import annotations

import numba
import numpy as np

CHUNK = 1 << 22


@numba.njit(cache=True)
def _swap_chunk(pre, post, n, keys, pairs, need):
    """Try the candidate pairs in order until ``need`` swaps are accepted. Returns (accepted, tried, rejected by
    reason: same edge, self-loop, duplicate)."""
    acc = 0
    tried = 0
    r_same = 0
    r_self = 0
    r_dup = 0
    for k in range(pairs.shape[0]):
        if acc >= need:
            break
        tried += 1
        i = pairs[k, 0]
        j = pairs[k, 1]
        if i == j:
            r_same += 1
            continue
        a = pre[i]
        b = post[i]
        c = pre[j]
        d = post[j]
        if a == d or c == b:
            r_self += 1
            continue
        k1 = a * n + d
        k2 = c * n + b
        if k1 in keys or k2 in keys:
            r_dup += 1
            continue
        del keys[a * n + b]
        del keys[c * n + d]
        keys[k1] = np.uint8(1)
        keys[k2] = np.uint8(1)
        post[i] = d
        post[j] = b
        acc += 1
    return acc, tried, r_same, r_self, r_dup


@numba.njit(cache=True)
def _key_set(pre, post, n):
    """Existing pairs as a typed dict (key -> 1). A numba ``set`` is not used: after many add / discard pairs in one
    call its probing degrades until it never ends; the typed dict handles deletions."""
    s = numba.typed.Dict.empty(key_type=numba.types.int64, value_type=numba.types.uint8)
    for e in range(pre.shape[0]):
        s[pre[e] * n + post[e]] = np.uint8(1)
    return s


def degree_preserving_shuffle(pre, post, n, seed, factor=10):
    """Returns (new post array in the original edge order, info). ``pre`` is unchanged by construction."""
    pre = np.ascontiguousarray(pre, np.int64)
    post = np.array(post, np.int64, copy=True)
    E = len(pre)
    if len(np.unique(pre * n + post)) != E:
        raise ValueError("duplicate pairs in the input graph")
    if (pre == post).any():
        raise ValueError("self-loops in the input graph")
    keys = _key_set(pre, post, n)
    rng = np.random.default_rng(seed)
    target = int(factor) * E
    acc = tried = r_same = r_self = r_dup = 0
    while acc < target:
        pairs = rng.integers(0, E, size=(CHUNK, 2))
        a, t, s1, s2, s3 = _swap_chunk(pre, post, n, keys, pairs, target - acc)
        acc += a
        tried += t
        r_same += s1
        r_self += s2
        r_dup += s3
    info = dict(seed=int(seed), factor=int(factor), n_edges=E, accepted=acc, tried=tried,
                rejected=dict(same_edge=r_same, self_loop=r_self, duplicate=r_dup), chunk=CHUNK)
    return post, info


def shuffle_check(pre, post0, post1, n, sign=None) -> dict:
    """prereg/14 gate 4 (docs/10 check 5): identical in- and out-degree vectors and per-presynaptic sign, no self-loops,
    no duplicate pairs; the fraction of edges whose postsynaptic end is unchanged."""
    pre = np.asarray(pre, np.int64)
    post0, post1 = np.asarray(post0, np.int64), np.asarray(post1, np.int64)
    res = dict(
        out_degree_equal=True,                                     # pre is the same array
        in_degree_equal=bool(np.array_equal(np.bincount(post0, minlength=n), np.bincount(post1, minlength=n))),
        no_self_loops=bool(not (pre == post1).any()),
        no_duplicates=bool(len(np.unique(pre * n + post1)) == len(pre)),
        fraction_in_place=float(np.mean(post0 == post1)),
        fraction_pairs_kept=float(np.isin(pre * n + post1, pre * n + post0).mean()))
    if sign is not None:
        sign = np.asarray(sign)
        ps = {}
        ok = True
        for p, s in zip(pre, sign):            # the edge keeps its sign; check one sign per presynaptic neuron
            if ps.setdefault(int(p), int(s)) != int(s):
                ok = False
                break
        res["sign_per_pre_equal"] = ok
    res["passed"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res
