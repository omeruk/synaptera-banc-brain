"""F0 bisection step c: log what the engine is actually driven with, interval by interval, through the perch and the
input cut of the F0 protocol (same code path as tasks/f0_flight.py)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from coupling.scheduler import CoupledLoop, LoopConfig  # noqa: E402
from tasks import f0_flight as F  # noqa: E402

out_dir = Path(sys.argv[1])
args = F.parse_args(["--name", "bisect_c", "--out-dir", str(out_dir), "--latency", "1"] + sys.argv[2:])
terms = F.RecordedTerms(F.V1_RECORD)
p = F.build(args, print)
plant, dec = F.F0Plant(p, terms, args), F.F0Decider(p, args.interval_ms)
cfg = LoopConfig(interval_ms=args.interval_ms, latency_intervals=1, mode="single_run")
eng = p["engine"]
eng.reset(args.seed)
log = []
saved = {'L': [], 'R': [], 'sugar': []}
orig_sense = plant.sense


def sense(k):
    d = orig_sense(k)
    saved['L'].append(np.asarray(d['vbnd_L'], float).copy()); saved['R'].append(np.asarray(d['vbnd_R'], float).copy())
    saved['sugar'].append(float(d['sugar']))
    log.append(dict(stage=plant.stage, k=k, kind="sense",
                    applied={n: float(np.mean(v)) for n, v in d.items()},
                    engine_rates={n: float(np.mean(eng._poisson[n].rates_[:])) for n in eng._poisson},
                    t_ms=eng.t_ms))
    return d


plant.sense = sense
dec.collect, dec.acc = True, []
plant.stage = "perch"
CoupledLoop(eng, plant, dec, cfg).run(args.calib_steps)
dec.collect = False
p["bridge"].set_baseline(np.mean(dec.acc, axis=0))
plant.stage = "cut"
CoupledLoop(eng, plant, dec, cfg).run(args.persist_steps)
# what the poisson groups hold right after the cut loop, and one more interval with nothing changed
after_cut = {n: float(np.mean(eng._poisson[n].rates_[:])) for n in eng._poisson}
t_end = eng.t_ms
rates_each = []
n_all = eng.n_neurons
for k in range(3):
    before = eng.spike_counts()
    eng.advance(25.0)
    rates_each.append(float((eng.spike_counts() - before).sum() / (n_all * 0.025)))
res = dict(sense_log=log, rates_after_cut_loop=after_cut, t_end_cut_ms=t_end, extra_idle_intervals_net_hz=rates_each)
np.savez(out_dir / 'drive.npz', L=np.array(saved['L']), R=np.array(saved['R']), sugar=np.array(saved['sugar']),
         idx_L=p['groups_in']['vbnd_L'], idx_R=p['groups_in']['vbnd_R'], idx_sugar=p['groups_in']['sugar'])
(out_dir / "inputs_log.json").write_text(json.dumps(res, indent=1))
for r in log:
    print(r["stage"], r["k"], "t", r["t_ms"], "applied", {k: round(v, 2) for k, v in r["applied"].items()},
          "engine", {k: round(v, 2) for k, v in r["engine_rates"].items()})
print("after cut loop:", after_cut, "idle 3 intervals net Hz:", np.round(rates_each, 3))
