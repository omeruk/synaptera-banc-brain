"""train/graphs.py: degree-preserving shuffle (prereg/14 arm `shuffled`) and its check (gate 4)."""
import numpy as np
import pytest

from train.graphs import degree_preserving_shuffle, shuffle_check


def graph(seed=0, n=500, k=8):
    rng = np.random.default_rng(seed)
    pre, post = [], []
    for i in range(n):
        c = rng.choice(n - 1, size=k, replace=False)
        c[c >= i] += 1
        pre.append(c)
        post.append(np.full(k, i))
    pre, post = np.concatenate(pre), np.concatenate(post)
    sign = np.where(rng.random(n) < 0.4, -1, 1)[pre]
    return pre, post, sign, n


def test_shuffle_preserves_degrees_signs_and_has_no_loops_or_duplicates():
    pre, post, sign, n = graph()
    new, info = degree_preserving_shuffle(pre, post, n, seed=20261501)
    assert info["accepted"] == 10 * len(pre)
    chk = shuffle_check(pre, post, new, n, sign)
    assert chk["passed"], chk
    assert chk["fraction_in_place"] < 0.05


def test_shuffle_is_deterministic_and_seed_dependent():
    pre, post, _, n = graph(1)
    a, _ = degree_preserving_shuffle(pre, post, n, seed=5, factor=2)
    b, _ = degree_preserving_shuffle(pre, post, n, seed=5, factor=2)
    c, _ = degree_preserving_shuffle(pre, post, n, seed=6, factor=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_input_graph_with_duplicates_is_refused():
    with pytest.raises(ValueError):
        degree_preserving_shuffle(np.array([0, 0]), np.array([1, 1]), 3, seed=0)


def test_check_detects_a_broken_shuffle():
    pre, post, sign, n = graph(2, n=50, k=4)
    bad = post.copy()
    bad[0] = pre[0]                       # self-loop, and the in-degree changes
    chk = shuffle_check(pre, post, bad, n, sign)
    assert not chk["no_self_loops"] and not chk["in_degree_equal"] and not chk["passed"]
