"""coupling/dn_bridge.py on toy tables (no connectome data)."""
import numpy as np
import pandas as pd
import pytest

from coupling.dn_bridge import BridgeConfig, DNBridge, family

FW = pd.DataFrame(dict(cell_type=["DNa04", "DNa04", "DNg02_a", "DNg02_b", "DNg02_d", "DNa10", "DNa10", "DNx01"],
                       side=["left", "right", "left", "left", "left", "left", "right", "center"]))
BANC = pd.DataFrame(dict(cell_type=["DNa04", "DNa04", "DNg02", "DNg02", "DNg02_d", "DNa10", "DNa10", "DNa15", "DNx01"],
                         side=["left", "right", "left", "left", "left", "left", "right", "left", "left"],
                         index=[10, 11, 20, 21, 22, 30, 31, 40, 50]))


def bridge(**kw):
    kw.setdefault("families", ("DNa04", "DNg02", "DNa10", "DNa15", "DNx01"))
    return DNBridge(FW, BANC, BridgeConfig(**kw))


def test_family_helper():
    assert family("DNg02_d") == "DNg02" and family("DNa04") == "DNa04" and family("DNp_ab") == "DNp_ab"


def test_exact_type_first_then_family_then_silent():
    b = bridge()
    by = {tuple(u["key"]): u for u in b.units}
    assert by[("type", "DNg02_d", "left")]["fw_types"] == ["DNg02_d"]
    assert by[("type", "DNg02_d", "left")]["banc_index"].tolist() == [22]
    fam = by[("family", "DNg02", "left")]
    assert fam["fw_types"] == ["DNg02_a", "DNg02_b", "DNg02_d"] and fam["banc_index"].tolist() == [20, 21]
    assert by[("type", "DNa04", "right")]["banc_index"].tolist() == [11]
    # DNa15 has no FlyWire neuron; DNx01 left has only a FlyWire centre neuron: both silent
    silent = {(d["cell_type"], d["side"]) for d in b.unmatched_banc}
    assert silent == {("DNa15", "left"), ("DNx01", "left")}
    assert 40 not in b.target_index and 50 not in b.target_index


def test_side_is_kept():
    b = bridge(families=("DNa04",))
    for u in b.units:
        assert set(FW["side"].iloc[u["fw_pos"]]) == {u["side"]}


def test_per_neuron_mean_rate_latency_and_window():
    b = bridge(families=("DNg02",), window_intervals=2, latency_intervals=1, interval_ms=5.0)
    n = 6
    c = np.zeros((n, len(FW)), int)
    c[2, 2] = 1            # DNg02_a left, interval 2
    c[2, 3] = 1            # DNg02_b left, interval 2
    ur = b.unit_rates(c)
    k = [tuple(u["key"]) for u in b.units].index(("family", "DNg02", "left"))
    # 2 spikes over 3 FlyWire neurons over a 2-interval (10 ms) window = 66.7 Hz per neuron, applied in 3 and 4
    exp = np.zeros(n)
    exp[3] = exp[4] = 2 / (3 * 2 * 0.005)
    assert np.allclose(ur[:, k], exp)
    assert ur[2, k] == 0.0      # never in the interval the spikes happened in


def test_latency_zero_needs_acausal_flag():
    with pytest.raises(ValueError):
        BridgeConfig(families=("DNa04",), latency_intervals=0)
    b = bridge(families=("DNa04",), latency_intervals=0, acausal=True, window_intervals=1)
    c = np.zeros((3, len(FW)), int)
    c[1, 0] = 1
    assert b.unit_rates(c)[1].max() == pytest.approx(200.0)
    assert b.config.record()["variant"] == "ACAUSAL"


def test_gain_and_rates_columns():
    b = bridge(families=("DNa04", "DNa10"), gains={"DNa04": 2.0, "DNa10": 0.5}, window_intervals=1)
    c = np.zeros((3, len(FW)), int)
    c[0, 0] = 1                 # DNa04 left
    c[0, 6] = 1                 # DNa10 right
    r = b.rates(c)
    tgt = b.target_index.tolist()
    assert r.shape == (3, len(tgt))
    assert r[1, tgt.index(10)] == pytest.approx(400.0)
    assert r[1, tgt.index(31)] == pytest.approx(100.0)
    assert r[1, tgt.index(11)] == 0.0 and r[0].max() == 0.0


def test_shuffled_source_and_gain_by():
    cfg = dict(families=("DNa04", "DNa10"), gains={"DNa04": 2.0, "DNa10": 10.0}, window_intervals=1,
               source={"DNa04": "DNa10", "DNa10": "DNa04"})
    c = np.zeros((2, len(FW)), int)
    c[0, 0] = 1                 # FlyWire DNa04 left fires
    for by, g in (("target", 10.0), ("source", 2.0)):
        b = bridge(gain_by=by, **cfg)
        tgt = b.target_index.tolist()
        r = b.rates(c)
        assert r[1, tgt.index(30)] == pytest.approx(200.0 * g)    # BANC DNa10 left receives FlyWire DNa04 left
        assert r[1, tgt.index(10)] == 0.0                          # BANC DNa04 left receives FlyWire DNa10 left (silent)
        assert all(u["match"] == "shuffled" for u in b.units)


def test_transfer_accounting_and_record():
    b = bridge(families=("DNa04", "DNa10"))
    c = np.zeros((2, len(FW)), int)
    c[0, [0, 1, 5, 7]] = 1
    t = b.transfer(c)
    assert t["spikes_all_dn"] == 4 and t["spikes_used"] == 3 and t["fraction_lost"] == 0.0
    rec = b.record()
    assert rec["label"] == "BRAIN" and rec["config"]["latency_intervals"] == 1
    assert sum(u["n_banc"] for u in rec["units"]) == len(b.target_index)


def test_unmatched_fw_is_reported():
    fw = pd.concat([FW, pd.DataFrame(dict(cell_type=["DNa04"], side=["center"]))], ignore_index=True)
    b = DNBridge(fw, BANC, BridgeConfig(families=("DNa04",)))
    assert b.unmatched_fw == [dict(cell_type="DNa04", side="center")]
    c = np.zeros((1, len(fw)), int)
    c[0, -1] = 3
    c[0, 0] = 1
    assert b.transfer(c)["fraction_lost"] == pytest.approx(0.75)


def test_bad_config():
    with pytest.raises(ValueError):
        BridgeConfig(families=("DNa04",), source={"DNa10": "DNa04"})
    with pytest.raises(ValueError):
        BridgeConfig(families=("DNa04",), window_intervals=0)
    with pytest.raises(ValueError):
        DNBridge(FW, BANC, BridgeConfig(families=("DNa04",))).unit_rates(np.zeros((2, 3)))


def test_f3b1_arms():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import f3b1_common as C
    assert all(t != s for t, s in C.PERMUTATION.items()) and sorted(C.PERMUTATION.values()) == sorted(C.STEER_FAMILIES)
    assert C.bridge_config("bridge_off") is None
    assert C.bridge_config("ref").gains == {} and C.bridge_config("ref").latency_intervals == 1
    sg = C.bridge_config("shuffled_gain")
    assert sg.gain_by == "source" and sg.gains == C.RATE_GAIN and sg.source == C.PERMUTATION
    assert len({C.vnc_seed(st, s) for st in C.STIMULI for s in C.SEEDS}) == len(C.STIMULI) * len(C.SEEDS)


def test_f3b2_arms():
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    import f3b2_common as C
    cw = json.loads((root / "docs" / "results" / "f3_dn_crosswalk.json").read_text())["summary"]
    assert list(C.ENSEMBLE) == cw["shared_gate_both"] and len(C.ENSEMBLE) == cw["n_shared_gate_both"] == 112
    assert not any(f.startswith("DNg02") for f in C.ENSEMBLE)
    P = C.PERMUTATION
    assert set(P) == set(P.values()) == set(C.ENSEMBLE) and all(t != s for t, s in P.items())
    assert P == C.sattolo(list(C.ENSEMBLE), C.PERMUTATION_SEED)
    f, seen = C.ENSEMBLE[0], set()          # one cycle through all families
    while f not in seen:
        seen.add(f)
        f = P[f]
    assert len(seen) == len(C.ENSEMBLE)
    assert C.bridge_config("bridge_off") is None
    for arm in ("ensemble", "ensemble_shuffled"):
        c = C.bridge_config(arm)
        assert c.gains == {} and c.latency_intervals == 1 and c.window_intervals == 5 and c.families == C.ENSEMBLE
    assert C.bridge_config("ensemble_shuffled").source == P and C.bridge_config("ensemble").source == {}
    assert set(C.CONTROLS["ensemble"]) == {"bridge_off", "ensemble_shuffled"}


def test_sattolo_is_single_cycle():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import f3b2_common as C
    items = [f"f{i}" for i in range(9)]
    for seed in range(20):
        p = C.sattolo(items, seed)
        f, seen = items[0], set()
        while f not in seen:
            seen.add(f)
            f = p[f]
        assert len(seen) == len(items)


def test_f3b3_arms():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import f3b2_common as C2
    import f3b3_common as C
    assert C.ENSEMBLE == C2.ENSEMBLE and len(C.RANDOM_ARMS) == len(C.PERM_SEEDS) == 39
    assert set(C.SEEDS) == set(range(10, 30)) and not set(C.SEEDS) & set(range(10))
    assert C.permutations_sha256() == C.PERMUTATIONS_SHA256
    P = C.permutations()
    assert list(P) == list(C.RANDOM_ARMS)
    for m in P.values():
        assert set(m) == set(m.values()) == set(C.ENSEMBLE) and all(t != s for t, s in m.items())
    assert len({tuple(sorted(m.items())) for m in P.values()}) == 39 and C2.PERMUTATION not in P.values()
    assert C.bridge_config("bridge_off") is None and C.bridge_config("ensemble").source == {}
    for arm in C.RANDOM_ARMS:
        c = C.bridge_config(arm, P)
        assert c.source == P[arm] and c.gains == {} and c.latency_intervals == 1 and c.window_intervals == 5
    vs = {C.vnc_seed(st, s) for st in C.STIMULI for s in C.SEEDS}
    assert len(vs) == len(C.STIMULI) * len(C.SEEDS) and not vs & (set(C.GATE1_SEEDS) | set(C.GATE2_SEEDS))


def test_derangement_has_no_fixed_point():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import f3b3_common as C
    items = [f"f{i}" for i in range(5)]
    for seed in range(50):
        p = C.derangement(items, seed)
        assert sorted(p.values()) == items and all(t != s for t, s in p.items())
    assert C.derangement(items, 7) == C.derangement(items, 7)
