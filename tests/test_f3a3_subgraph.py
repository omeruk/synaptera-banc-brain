"""BANC VNC subgraph of prereg/08 on a toy graph (no data files)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from f3a3_subgraph import vnc_subgraph  # noqa: E402
from tests.toy import edges_table, neurons_table  # noqa: E402
from connectome.schema import Connectome  # noqa: E402


def _toy():
    rows = [dict(id=1, cell_type="Bx", side="left", region="brain", super_class="central_brain_intrinsic"),
            dict(id=2, cell_type="DNx", side="left", region="brain", super_class="descending"),
            dict(id=3, cell_type="Vx", side="left", region="VNC", super_class="ventral_nerve_cord_intrinsic"),
            dict(id=4, cell_type="MNx", side="left", region="VNC", super_class="motor"),
            dict(id=5, cell_type="ANx", side="left", region="brain", super_class="ascending"),
            dict(id=6, cell_type="MNh", side="left", region="SEZ", super_class="motor"),
            dict(id=7, cell_type="Sx", side="left", region="SEZ", super_class="central_brain_intrinsic")]
    tr = [(1, 2, 5), (2, 3, 4), (3, 4, 3), (3, 2, 6), (3, 5, 7), (1, 5, 8), (5, 1, 9), (5, 3, 2), (7, 6, 3), (2, 2, 1)]
    return Connectome("toy", neurons_table(rows), edges_table(tr))


def test_vnc_subgraph_keeps_vnc_motor_dn_ascending_and_cuts_dn_inputs():
    sub, info = vnc_subgraph(_toy())
    assert sorted(sub.neurons["id"]) == [2, 3, 4, 5, 6]
    assert np.array_equal(sub.meta["global_index"], [1, 2, 3, 4, 5])
    got = sorted(map(tuple, sub.edges[["pre", "post"]].to_numpy()))
    assert got == [(2, 3), (3, 4), (3, 5), (5, 3)]          # onto DN (3->2, 2->2) removed; brain/SEZ edges gone
    assert info["kept"] == dict(neurons=5, pairs=4, synapses=16)
    assert info["removed"]["edges_onto_dn_from_kept"] == dict(pairs=2, synapses=7)
    assert info["ascending"]["inputs_removed"] == dict(pairs=1, synapses=8)
    assert info["ascending"]["outputs_removed"] == dict(pairs=1, synapses=9)
    assert info["kept_by_reason"] == dict(region=1, motor=2, descending=1, ascending=1)
