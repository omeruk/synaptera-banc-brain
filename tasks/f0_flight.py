"""F0 acceptance flight: the v1 n1 configuration on the new stack (docs/06_NEW_PROJECT_SETUP.md section 3).

n1 = hybrid, boundary-layer vision (sB), no brain steering, head reflex, postures, no olfaction input, seed 3,
300 decision steps of 25 ms after a 10-step perch calibration and an 8-step input cut.

What is and is not reproduced. v1 steered with a HAND odour->turn map (``turn_hand``) and a FlyVis loom bias
(``turn_flyvis``): both NeuroFly adaptations, not carried over (CLAUDE.md rule 12, docs/06 section 2). Their
*recorded* per-step values, and the recorded collective / tilt targets (which also depend on the odour field),
are replayed here as an open-loop command series from the v1 HDF5 (data, not code). Everything else runs on the new
code: arena, wing model, FlyGym body, REFLEX arm (haltere, head), HAND phases and leg logic, FlyVis boundary-layer
vision, the full-brain LIF engine, readouts + symmetry gate + bridge, the coupling scheduler and the recorder.

The brain does not enter the n1 command (turn_brain = 0, ``--no-brain-steer``), so the trajectory is a test of the
body / scaffold / coupling reproduction; the brain's contribution is the recorded feeding decision (MN9 -> S2).

    nohup env -u PYTHONPATH python scripts/run_f0.py --latency 0 --allow-acausal --name f0_lat0 &
    nohup env -u PYTHONPATH python scripts/run_f0.py --latency 1 --name f0_lat1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V1_RECORD = Path("~/Documents/NeuroFly/simulations/flight_v44_hybrid_sB_noBrSteer_head_pose_noOlf_n1_data.h5").expanduser()
MIN_FREE_GB = 6.0


def free_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


class RecordedTerms:
    """Per-step command terms of a v1 record, replayed open-loop (see module doc)."""

    FIELDS = ("turn_hand", "turn_flyvis", "thrust_hand", "pitch_hand", "roll_hand")

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            b = f["behavior"]
            self.a = {k: b[k][:] for k in self.FIELDS + ("turn_total", "wings_on", "phase", "pos", "leg_pose",
                                                         "is_feeding", "mn9_rate", "net_rate", "tower_contact",
                                                         "tower_penetration", "platform_contact", "turn_brain",
                                                         "proboscis_extended")}
            self.meta = {k: f["meta"].attrs[k] for k in ("seed", "n_steps", "calib_steps", "persist_steps")}
        self.n = len(self.a["turn_hand"])

    def at(self, k):
        return {name: float(self.a[name][k]) for name in self.FIELDS}


def criteria(path) -> dict:
    """S1: no tower contact (step-end flag and intra-step penetration) up to touchdown (phase code 4 = touchdown,
    8 = landed); S2: ``is_feeding`` = 1 in at least one step after touchdown. Landing step = first touchdown /
    landed row."""
    with h5py.File(path, "r") as f:
        b = f["behavior"]
        ph = b["phase"][:]
        n = len(ph)
        landed = np.flatnonzero((ph == 4) | (ph == 8))
        k0 = int(landed[0]) if len(landed) else None
        upto = slice(0, (k0 + 1) if k0 is not None else n)
        tc = b["tower_contact"][upto]
        pen = b["tower_penetration"][upto]
        feeding = b["is_feeding"][:]
    s1 = bool(k0 is not None and tc.sum() == 0 and float(pen.max(initial=0.0)) == 0.0)
    s2 = bool(k0 is not None and feeding[k0:].max(initial=0) > 0)
    return dict(landed_step=k0, S1=s1, S2=s2, tower_contact_steps=int(tc.sum()),
                max_penetration_mm=float(pen.max(initial=0.0)),
                feeding_steps=int(feeding.sum()), first_feeding_step=(int(np.flatnonzero(feeding)[0])
                                                                      if feeding.any() else None))


def compare(new_path, v1_path=V1_RECORD, stride: int = 1) -> dict:
    """Position / phase comparison of a new run with the v1 record (same 300 closed-loop steps)."""
    with h5py.File(new_path, "r") as f, h5py.File(v1_path, "r") as g:
        a, b = f["behavior"], g["behavior"]
        sl = slice(stride - 1, None, stride)           # the state at the end of every 25 ms block
        A = {k: a[k][:][sl] for k in ("pos", "phase", "wings_on", "leg_pose", "heading", "speed")}
        n = min(len(A["pos"]), len(b["pos"]))
        pn, pv = A["pos"][:n], b["pos"][:n]
        d = np.linalg.norm(pn - pv, axis=1)
        out = dict(n_steps=int(n), pos_diff_mm=dict(max=float(d.max()), final=float(d[-1]), mean=float(d.mean()),
                                                    rms=float(np.sqrt((d ** 2).mean())),
                                                    first_step_above_1mm=(int(np.flatnonzero(d > 1.0)[0])
                                                                          if (d > 1.0).any() else None)),
                   phase_identical=bool(np.array_equal(A["phase"][:n], b["phase"][:n])),
                   phase_steps_differing=int((A["phase"][:n] != b["phase"][:n]).sum()),
                   wings_on_identical=bool(np.array_equal(A["wings_on"][:n], b["wings_on"][:n])),
                   leg_pose_steps_differing=int((A["leg_pose"][:n] != b["leg_pose"][:n]).sum()),
                   heading_diff_deg_max=float(np.degrees(np.abs(np.angle(np.exp(
                       1j * (A["heading"][:n] - b["heading"][:n]))))).max()),
                   speed_diff_max_mm_s=float(np.abs(A["speed"][:n] - b["speed"][:n]).max()))
    return out


# ── the experiment ───────────────────────────────────────────────────────────
def build(args, log=print):
    """Engine, body, vision, readouts, bridge, scaffold. Returns a dict of the parts."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    from body import quasi_steady as qs  # noqa: F401
    from body.arena import load_arena
    from body.flygym_body import FlightBody
    from configs import load
    from connectome.adapters import flywire783 as fw
    from coupling.recorder import FLIGHT_FIELDS, Recorder
    from engine.lif_brian2 import Brian2LIFEngine
    from motor import hand_scaffold as HS
    from motor import legacy_bridge as LB
    from motor import legacy_reflexes as RF
    from motor.readouts import Readouts
    from sense import sensors
    from sense import vision_boundary as VB
    from sense import vision_flyvis as V

    arena = load_arena()
    t0 = time.time()
    log("connectome ...")
    cx = fw.load_connectome(hash_files=True)
    readouts = Readouts(cx, "flywire783", waive=tuple(args.waive))
    sugar_ids = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    table = VB.load_table(ROOT / "data" / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    vgroups = VB.boundary_groups(cx, table)
    groups_in = {"sugar": cx.index.group(sugar_ids), "vbnd_L": vgroups["vbnd_L"], "vbnd_R": vgroups["vbnd_R"]}
    out = Path(args.out_dir) / f"{args.name}_data.h5"
    n_pre = args.calib_steps + args.persist_steps
    import dataclasses
    fields = tuple(dataclasses.replace(f, label="REPLAY") if f.name in RecordedTerms.FIELDS else f
                   for f in FLIGHT_FIELDS)
    rec = Recorder(out, fields,
                   meta=dict(experiment="F0 acceptance (v1 n1 configuration)", name=args.name, seed=args.seed,
                             interval_ms=args.interval_ms, latency_intervals=args.latency,
                             n_steps=args.n_steps, calib_steps=args.calib_steps, persist_steps=args.persist_steps,
                             dataset="flywire783", sign_rule="shiu_v1", sha256=cx.meta.get("sha256"),
                             readouts=readouts.record(), arena=arena.geometry_record(),
                             replayed_terms="REPLAY: turn_hand, turn_flyvis, thrust_hand, pitch_hand, roll_hand from the v1 "
                                            "record (NeuroFly-adapted terms are not carried as code); a port "
                                            "regression test, not a result",
                             v1_record=str(V1_RECORD)),
                   n_intervals=args.n_steps, overwrite=False,
                   input_groups={k: len(v) for k, v in groups_in.items()})
    ro_groups = readouts.group_names()
    engine = Brian2LIFEngine(cx, input_groups=groups_in, groups=ro_groups, seed=args.seed,
                             spike_sink=rec.spike_sink, poisson_delay_v1=getattr(args, 'poisson_delay_v1', False))
    log(f"engine: {engine.n_neurons:,} neurons, {len(engine.syn):,} synapses (setup {time.time() - t0:.0f} s, "
        f"{free_gb():.1f} GB free)")
    body = FlightBody(arena=arena, enable_vision=True, textured=True, platform="dark", neck_actuators=True,
                      postures=True)
    columns = V.load_columns(ROOT / "data" / "column_assignment.csv.gz")
    tr = VB.load_transduction(args.boundary_transduction)
    assert abs(tr["dt"] - args.interval_ms * 1e-3 / args.subframes) < 1e-9, \
        f"transduction calibrated at dt {tr['dt']}, run uses {args.interval_ms * 1e-3 / args.subframes}"
    interval_s = args.interval_ms * 1e-3
    eyes = VB.BoundaryEyes(table, {"L": cx.index.ids_of(groups_in["vbnd_L"]), "R": cx.index.ids_of(groups_in["vbnd_R"])},
                           tr, columns, cx.index, interval_s=interval_s, subframes=args.subframes, verbose=False)
    bridge = LB.Bridge(readouts, interval_s, reference=LB.load_reference(ROOT / "data" / "dn_lr_reference_sB.json"))
    scaffold = HS.HandScaffold(arena)
    return dict(arena=arena, cx=cx, readouts=readouts, engine=engine, body=body, eyes=eyes, bridge=bridge,
                scaffold=scaffold, rec=rec, groups_in=groups_in, out=out, HS=HS, RF=RF, sensors=sensors, qs=qs,
                load=load, n_pre=n_pre)


class F0Plant:
    """Body + vision + HAND scaffold; replays the recorded terms. One instance serves the perch and the flight."""

    def __init__(self, p, terms: RecordedTerms, args):
        self.p, self.terms, self.args = p, terms, args
        self.body, self.eyes, self.arena = p["body"], p["eyes"], p["arena"]
        self.RF, self.HS, self.qs = p["RF"], p["HS"], p["qs"]
        self.interval_s = args.interval_ms * 1e-3
        self.stride = int(round(25.0 / args.interval_ms))
        n = self.arena.physics_steps(self.interval_s)
        self.render_at = tuple(int(round(n * (j + 1) / args.subframes)) for j in range(args.subframes))
        self.halt = self.RF.HaltereReflex(self.body.params)
        self.head = self.RF.HeadReflex()
        self.pm = self.HS.HandPhases(p["scaffold"], self.interval_s)
        self.expansion = p["sensors"].ExpansionRate(self.arena, self.interval_s)
        self.frames = [self.body.update_vision().copy()] * args.subframes     # static scene before step 0
        self.stage = "perch"                  # perch | cut | flight
        self.tuck_done = self.body.legs == "tuck"
        self.touched = False
        self.t_run0 = None
        self.last_sense = {}
        self.off = self.qs.WingCommand(0, 0, 0, on=False)

    # engine input for interval k ------------------------------------------------------------------
    def sense(self, k):
        a, loom = self.eyes.step(self.frames)
        r = self.eyes.rates(a)
        self.last_sense = dict(r=r, loom=loom)
        if self.stage == "cut":
            return {"sugar": 0.0, "vbnd_L": np.zeros(len(r["L"])), "vbnd_R": np.zeros(len(r["R"]))}
        con = self.body.contacts()
        sugar = 100.0 if (self.stage == "flight" and con["platform_legs"].any()) else 0.0
        return {"sugar": sugar, "vbnd_L": r["L"], "vbnd_R": r["R"]}

    # body for interval k ----------------------------------------------------------------------------
    def act(self, k, command, dt_s):
        body, arena, a = self.body, self.arena, self.args
        t_step = time.time()
        if self.stage != "flight":
            r = body.step(self.off, dt_s, inner=[self.head], render_at=self.render_at)
            self.frames = r["frames"]
            return dict(stage=self.stage)
        st = body.state()
        con = body.contacts()
        pm = self.pm
        if self.t_run0 is None:
            self.t_run0 = st["time"]
        phase = pm.update(st["pos"], st["vel"], int(con["platform_legs"].sum()))
        if pm.legs_extended and not pm.landed:
            body.set_legs("stand")                                    # HAND: legs extended in approach / descend
        extending = pm.legs_extended
        if (not self.tuck_done and pm.wings_on and not extending and not pm.landed
                and not (con["pedestal_legs"].any() or con["ground_legs"].any() or con["platform_legs"].any())):
            body.set_legs("tuck")                                     # REFLEX: tarsal contact loss -> tuck, once
            self.tuck_done = True
        if pm.landed and not self.touched:
            self.touched = True
            body.set_legs("stand", adhesion=1.0)
        feed = command["feed"]
        if pm.landed:
            body.set_legs("feed" if feed["proboscis_extended"] else "stand")
        terms = self.terms.at(k // self.stride)     # recorded terms are constant per 25 ms block (v1 step)
        t_brain = 0.0 if a.no_brain_steer else command["steer"]["turn_dn"]
        t_tot = self.HS.combine_turn(t_brain, terms["turn_hand"], terms["turn_flyvis"])
        pitch_down, roll_right = terms["pitch_hand"], terms["roll_hand"]
        cmd, lift = self.p["scaffold"].wing_command(t_tot, terms["thrust_hand"], pitch_down, roll_right,
                                                    body.params, wings_on=pm.wings_on)
        self.halt.set_target(pitch_down, roll_right)
        inner = [self.halt, self.head] if cmd.on else [self.head]
        r = body.step(cmd, dt_s, inner=inner, render_at=self.render_at)
        self.frames = r["frames"]
        s1 = body.state()
        c1 = body.contacts()
        t_reflex = self.qs.yaw_torque_to_turn(self.halt.pop_yaw_torque(), body.params, body.wing["turn_dphi_deg"])
        h = r["head"] or {}
        sense = self.last_sense
        t45 = self.eyes.t45_view(sense["r"])[0]
        e = self.expansion(st["pos"])
        return dict(
            stage="flight", t=s1["time"] - self.t_run0, pos=s1["pos"], vel=s1["vel"],
            speed=float(np.linalg.norm(s1["vel"])), quat=s1["quat"], omega=s1["omega"], heading=s1["heading"],
            dist_to_food=float(np.linalg.norm(s1["pos"] - np.asarray(arena.food_pos))),
            phase=self.HS.PHASE_CODE[phase], turn_total=t_tot, turn_brain=t_brain, turn_hand=terms["turn_hand"],
            turn_flyvis=terms["turn_flyvis"], turn_reflex=t_reflex, thrust_hand=terms["thrust_hand"],
            pitch_hand=pitch_down, roll_hand=roll_right, lift_frac=lift, wings_on=int(pm.wings_on),
            stroke_amp_L=cmd.amp_L, stroke_amp_R=cmd.amp_R, stroke_freq=cmd.freq, force=r["force"],
            torque=r["torque"], amp_clipped_substeps=r["clipped_substeps"],
            mn9_rate=feed["mn9_rate"], proboscis_extended=feed["proboscis_extended"],
            steer_raw=command["steer"]["steer_raw"], steer_norm_t=command["steer"]["steer_norm_t"],
            steer_norm=command["steer"]["steer_norm"], dn_readout_rate=command["used"],
            t45_rate_L=float(t45["L"].mean()), t45_rate_R=float(t45["R"].mean()),
            loom_L=sense["loom"][0], loom_R=sense["loom"][1], expansion_rate=e,
            height_above=float(st["pos"][2] - arena.surface_below(st["pos"][0], st["pos"][1])),
            platform_azimuth=self.p["sensors"].platform_azimuth(arena, s1["pos"], s1["heading"]),
            platform_contact=int(c1["platform_legs"].sum()), tower_contact=int(c1["tower_contact"]),
            tower_penetration=r["tower_penetration"], min_tower_clearance=c1["min_tower_clearance"],
            is_feeding=int(pm.landed and feed["proboscis_extended"]), leg_pose={"stand": 0, "tuck": 1, "feed": 2}[body.legs],
            step_time=time.time() - t_step)


class F0Decider:
    """Engine output of interval k -> (feeding decision, steering decision, filtered readouts)."""

    def __init__(self, p, interval_ms):
        self.p, self.interval_ms = p, interval_ms
        self.ro, self.bridge = p["readouts"], p["bridge"]
        self.neutral = dict(feed=dict(mn9_rate=0.0, proboscis_extended=0),
                            steer=dict(turn_dn=0.0, steer_raw=np.zeros(2), steer_norm_t=np.zeros(2), steer_norm=0.0),
                            used=np.zeros((len(self.ro.types), 2)))
        self.record = None
        self.acc = None            # perch calibration accumulator
        self.collect = False
        self.active = False        # the bridge filter runs in the flight loop only (as in v1)

    def decide(self, k, engine):
        hz = self.ro.hz(engine, self.interval_ms)
        if self.collect:
            self.acc.append(hz)
        if not self.active:
            self.record = dict(net_rate=float(engine.rates(["all"], self.interval_ms)["all"].mean()))
            return self.neutral
        _, used = self.bridge.update_hz(hz)
        out = dict(feed=self.bridge.proboscis(used), steer=self.bridge.steer(used), used=used)
        self.record = dict(net_rate=float(engine.rates(["all"], self.interval_ms)["all"].mean()))
        return out


def run(args, log=print) -> dict:
    from coupling.scheduler import CoupledLoop, LoopConfig
    stride = int(round(25.0 / args.interval_ms))
    if abs(stride * args.interval_ms - 25.0) > 1e-9:
        raise SystemExit("the interval must divide 25 ms (the unit of the replayed terms)")
    # n_steps / calib_steps / persist_steps are given in v1 steps of 25 ms and converted to intervals
    args.n_steps, args.calib_steps, args.persist_steps = (args.n_steps * stride, args.calib_steps * stride,
                                                          args.persist_steps * stride)
    terms = RecordedTerms(V1_RECORD)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb():.1f} GB RAM available (< {MIN_FREE_GB}); not starting a full-brain run")
    p = build(args, log)
    plant, dec = F0Plant(p, terms, args), F0Decider(p, args.interval_ms)
    cfg = LoopConfig(interval_ms=args.interval_ms, latency_intervals=args.latency, mode="single_run",
                     allow_acausal=args.allow_acausal)
    rec, engine = p["rec"], p["engine"]
    p["engine"].reset(args.seed)
    t0 = time.time()

    # perch calibration (wings off) and the input cut, same coupling code; no recording of behaviour
    dec.collect, dec.acc = True, []
    plant.stage = "perch"
    perch = CoupledLoop(engine, plant, dec, cfg)
    perch.run(args.calib_steps)
    dec.collect = False
    base = np.mean(dec.acc, axis=0)
    p["bridge"].set_baseline(base)
    plant.stage = "cut"
    CoupledLoop(engine, plant, dec, cfg).run(args.persist_steps)
    persist_rate = float(engine.rates(["all"], 25.0)["all"].mean())      # last 25 ms of the cut (the v1 step)
    p["bridge"].set_baseline(base)          # the filter starts the flight at the perch baseline (as in v1)
    log(f"calibration done ({time.time() - t0:.0f} s); net rate after the input cut {persist_rate:.3f} Hz")

    # closed loop
    plant.stage = "flight"
    dec.active = True
    state = {"k": 0}

    def on_interval(k, r):
        fields = {n: v for n, v in r.items() if n in rec.fields and n != "cmd_source"}
        if r.get("decided"):
            fields["net_rate"] = r["decided"]["net_rate"]
        fields["cmd_source"] = r["cmd_source"]
        fields["rss_gb"] = _rss_gb()
        rec.put(k, **fields)
        if k < 3 or k % 25 == 0:
            ph = r.get("phase")
            log(f"  step {k:03d} phase {ph} pos {np.round(r['pos'], 1).tolist()} v {r['speed']:.0f} "
                f"mn9 {r['mn9_rate']:.1f} Hz feed {r['is_feeding']} rss {fields['rss_gb']:.2f} GB")
        state["k"] = k + 1

    loop = CoupledLoop(engine, plant, dec, cfg, on_interval=on_interval)
    # the engine inputs of the flight intervals are stored for the replay audit
    orig_sense = plant.sense

    def sense_and_store(k):
        d = orig_sense(k)
        rec.put_inputs(k, d)
        return d
    plant.sense = sense_and_store
    loop.run(args.n_steps)
    wall = time.time() - t0
    extra = dict(coupling=cfg.record(), perch_baseline_hz=base.tolist(), persist_net_rate_hz=persist_rate,
                 wall_s=wall, peak_rss_gb=_peak_rss_gb(), clip_events=p["body"].clip_events,
                 badqacc=p["body"].badqacc_count() - p["body"].badqacc0,
                 engine_state=_jsonable(engine.state_dict()), n_spikes=int(engine.spike_counts().sum()),
                 seed=args.seed)
    path = rec.close(extra)
    log(f"done in {wall:.0f} s -> {path}")
    res = dict(path=str(path), criteria=criteria(path), vs_v1=compare(path, stride=stride), wall_s=wall)
    Path(str(path).replace("_data.h5", "_report.json")).write_text(json.dumps(res, indent=1, default=str))
    return res


def _rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def _peak_rss_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def _jsonable(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if k not in ("v", "g")}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "f0"))
    ap.add_argument("--latency", type=int, default=1, help="coupling latency in intervals (0 = v1 same-interval)")
    ap.add_argument("--allow-acausal", action="store_true")
    ap.add_argument("--interval-ms", type=float, default=25.0, help="25 ms reproduces v1 (5 ms is the new default)")
    ap.add_argument("--subframes", type=int, default=2)
    ap.add_argument("--boundary-transduction", default=str(ROOT / "data" / "visual_transduction.json"),
                    help="must be calibrated at dt = interval / subframes (scripts/make_transduction.py)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=300, help="in v1 steps of 25 ms")
    ap.add_argument("--calib-steps", type=int, default=10)
    ap.add_argument("--persist-steps", type=int, default=8)
    ap.add_argument("--no-brain-steer", dest="no_brain_steer", action="store_true", default=True)
    ap.add_argument("--poisson-delay-v1", dest="poisson_delay_v1", action="store_true",
                    help="named engine variant poisson_delay_v1 (v1 1.8 ms delay on the Poisson input)")
    ap.add_argument("--waive", nargs="*", default=[], help="readout types allowed to fail the symmetry gate")
    return ap.parse_args(argv)


def main(argv=None):
    res = run(parse_args(argv))
    print(json.dumps(res, indent=1, default=str))
    return 0
