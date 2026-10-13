"""v1 reproduction: hover + yaw-perturbation flights with the brain closing the yaw loop (v1 flight_v30 ... v41).

Brain-only path of v1 (no HAND odour / FlyVis terms): the yaw command is ``turn_dn`` of the hand-made bridge, the
collective is the vertical-speed damping -v_z / VZ_DAMP_REF (odour off), the body tilt is 0, the REFLEX arm is the
haltere attitude PD. A 0.1 s body-z torque pulse of ``--yaw-perturb DEG`` turns the passive fly by DEG at t = 0.5 s.
``--ablate-steer`` replaces the steer readouts by their perch baseline (the v31 control).

The v1 drive is DNp15, which FAILS the symmetry gate (R/L 1.66x): it is run only as a labelled exception
(``--waive DNp15``) to reproduce v1. Nothing here is a new result; the DNa02 version is a new experiment (F2).

    nohup env -u PYTHONPATH python scripts/run_hover_yaw.py --name hy_v30_lat1 --latency 1 &
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

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = Path("~/Documents/NeuroFly/simulations").expanduser()
V1_RECORDS = {
    "v30": "flight_v30_sB_ablOdor_noOlf_air440_-170_160_90_hover_yawP+30_v2val_data.h5",
    "v31": "flight_v31_sB_ablDN-steer_ablOdor_noOlf_air440_-170_160_90_hover_yawP+30_v2val_data.h5",
}
YAW_PERTURB_S = 0.1
VZ_DAMP_REF = 400.0
VENTRAL_H, VENTRAL_K = 8.0, 0.5        # v1 HAND ventral reflex (inactive at 160 mm above the ground)
SPAWN, SPAWN_YAW_DEG = (440.0, -170.0, 160.0), 90.0


def build(args, log=print):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from body import quasi_steady as qs
    from body.arena import load_arena
    from body.flygym_body import FlightBody
    from connectome.adapters import flywire783 as fw
    from coupling.recorder import FLIGHT_FIELDS, Recorder
    from engine.lif_brian2 import Brian2LIFEngine
    from motor import hand_scaffold as HS
    from motor import legacy_bridge as LB
    from motor import legacy_reflexes as RF
    from motor.readouts import Readouts
    from sense import vision_boundary as VB
    from sense import vision_flyvis as V

    arena = load_arena()
    cx = fw.load_connectome(hash_files=True)
    readouts = Readouts(cx, "flywire783", waive=tuple(args.waive))
    sugar_ids = pd.read_csv(ROOT / "data" / "sugar_grn_783.csv")["root_id"].to_numpy()
    table = VB.load_table(ROOT / "data" / "vision_boundary_783.csv").rename(columns={"root_id": "id"})
    vg = VB.boundary_groups(cx, table)
    groups_in = {"sugar": cx.index.group(sugar_ids), "vbnd_L": vg["vbnd_L"], "vbnd_R": vg["vbnd_R"]}
    out = Path(args.out_dir) / f"{args.name}_data.h5"
    rec = Recorder(out, FLIGHT_FIELDS,
                   meta=dict(experiment="v1 reproduction: hover + yaw perturbation (brain closes the yaw loop)",
                             name=args.name, seed=args.seed, interval_ms=args.interval_ms,
                             latency_intervals=args.latency, n_steps=args.n_steps, v1_reference=args.v1,
                             ablate_steer=args.ablate_steer, waived=list(args.waive),
                             readouts=readouts.record(), sha256=cx.meta.get("sha256"),
                             yaw_perturb=[args.perturb_t, args.perturb_deg]),
                   n_intervals=args.n_steps, input_groups={k: len(v) for k, v in groups_in.items()})
    engine = Brian2LIFEngine(cx, input_groups=groups_in, groups=readouts.group_names(), seed=args.seed,
                             spike_sink=rec.spike_sink)
    body = FlightBody(arena=arena, spawn_pos=SPAWN, spawn_yaw=np.radians(SPAWN_YAW_DEG), legs="tuck",
                      enable_vision=True, textured=True, platform="dark")
    columns = V.load_columns(ROOT / "data" / "column_assignment.csv.gz")
    tr = VB.load_transduction(ROOT / "data" / "visual_transduction.json")
    dt = args.interval_ms * 1e-3
    eyes = VB.BoundaryEyes(table, {"L": cx.index.ids_of(groups_in["vbnd_L"]), "R": cx.index.ids_of(groups_in["vbnd_R"])},
                           tr, columns, cx.index, interval_s=dt, subframes=2, verbose=False)
    bridge = LB.Bridge(readouts, dt, ablate=("steer",) if args.ablate_steer else (), zero_turn=args.zero_turn,
                       reference=LB.load_reference(ROOT / "data" / "dn_lr_reference_sB.json"),
                       drive=("DNp15",))
    return dict(arena=arena, cx=cx, readouts=readouts, engine=engine, body=body, eyes=eyes, bridge=bridge,
                scaffold=HS.HandScaffold(arena), rec=rec, RF=RF, qs=qs, HS=HS)


class HoverPlant:
    def __init__(self, p, args):
        self.p, self.args = p, args
        self.body, self.eyes, self.arena = p["body"], p["eyes"], p["arena"]
        self.dt = args.interval_ms * 1e-3
        n = self.arena.physics_steps(self.dt)
        self.render_at = tuple(int(round(n * (j + 1) / 2)) for j in range(2))
        self.halt = p["RF"].HaltereReflex(self.body.params)
        self.frames = [self.body.update_vision().copy()] * 2
        self.stage = "perch"
        self.pert_T = self.body.yaw_pulse_torque(args.perturb_deg, YAW_PERTURB_S)
        self.t_run0 = None

    def sense(self, k):
        a, loom = self.eyes.step(self.frames)
        r = self.eyes.rates(a)
        if self.stage == "cut":
            return {"sugar": 0.0, "vbnd_L": np.zeros(len(r["L"])), "vbnd_R": np.zeros(len(r["R"]))}
        return {"sugar": 0.0, "vbnd_L": r["L"], "vbnd_R": r["R"]}

    def _hover_cmd(self, turn):
        st = self.body.state()
        h = float(st["pos"][2] - self.arena.surface_below(st["pos"][0], st["pos"][1]))
        ventral = VENTRAL_K * (1.0 - max(h, 0.0) / VENTRAL_H) if h < VENTRAL_H else 0.0
        thrust = -float(st["vel"][2]) / VZ_DAMP_REF + ventral
        cmd, lift = self.p["scaffold"].wing_command(turn, thrust, 0.0, 0.0, self.body.params, wings_on=True)
        return cmd, lift, st, h

    def act(self, k, command, dt_s):
        body = self.body
        if self.stage != "flight":
            cmd, *_ = self._hover_cmd(0.0)
            r = body.step(cmd, dt_s, inner=[self.halt], render_at=self.render_at)
            self.frames = r["frames"]
            return {}
        turn = float(np.clip(command["steer"]["turn_dn"], -2.5, 2.5))
        cmd, lift, st, h = self._hover_cmd(turn)
        if self.t_run0 is None:
            self.t_run0 = st["time"]
        t_rel = k * dt_s
        pert = self.pert_T if self.args.perturb_t - 1e-9 <= t_rel < self.args.perturb_t + YAW_PERTURB_S - 1e-9 else 0.0
        self.halt.set_target(0.0, 0.0)
        r = body.step(cmd, dt_s, inner=[self.halt], render_at=self.render_at,
                      ext_torque=st["R"][:, 2] * pert if pert else None)
        self.frames = r["frames"]
        s1 = body.state()
        c1 = body.contacts()
        sb = command["steer"]
        return dict(t=s1["time"] - self.t_run0, pos=s1["pos"], vel=s1["vel"], speed=float(np.linalg.norm(s1["vel"])),
                    quat=s1["quat"], omega=s1["omega"], heading=s1["heading"], phase=2, turn_total=turn,
                    turn_brain=float(sb["turn_dn"]), lift_frac=lift, wings_on=1, stroke_amp_L=cmd.amp_L,
                    stroke_amp_R=cmd.amp_R, stroke_freq=cmd.freq, force=r["force"], torque=r["torque"],
                    amp_clipped_substeps=r["clipped_substeps"], steer_raw=sb["steer_raw"],
                    steer_norm_t=sb["steer_norm_t"], steer_norm=sb["steer_norm"], dn_readout_rate=command["used"],
                    mn9_rate=command["feed"]["mn9_rate"], height_above=h, yaw_perturb_torque=pert,
                    tower_contact=int(c1["tower_contact"]), tower_penetration=r["tower_penetration"],
                    min_tower_clearance=c1["min_tower_clearance"],
                    turn_reflex=self.p["qs"].yaw_torque_to_turn(self.halt.pop_yaw_torque(), body.params,
                                                                body.wing["turn_dphi_deg"]))


def run(args, log=print):
    from coupling.scheduler import CoupledLoop, LoopConfig
    from tasks.f0_flight import F0Decider, _jsonable, _peak_rss_gb, free_gb
    if free_gb() < 6.0:
        raise SystemExit(f"only {free_gb():.1f} GB RAM available (< 6); not starting a full-brain run")
    p = build(args, log)
    plant, dec = HoverPlant(p, args), F0Decider(p, args.interval_ms)
    cfg = LoopConfig(interval_ms=args.interval_ms, latency_intervals=args.latency, mode="single_run",
                     allow_acausal=args.latency == 0)
    eng, rec = p["engine"], p["rec"]
    eng.reset(args.seed)
    t0 = time.time()
    dec.collect, dec.acc = True, []
    CoupledLoop(eng, plant, dec, cfg).run(args.calib_steps)
    dec.collect = False
    base = np.mean(dec.acc, axis=0)
    p["bridge"].set_baseline(base)
    plant.stage = "cut"
    CoupledLoop(eng, plant, dec, cfg).run(args.persist_steps)
    persist = float(eng.rates(["all"], args.interval_ms)["all"].mean())
    p["bridge"].set_baseline(base)
    plant.stage = "flight"
    dec.active = True

    def on_interval(k, r):
        f = {n: v for n, v in r.items() if n in rec.fields and n != "cmd_source"}
        if r.get("decided"):
            f["net_rate"] = r["decided"]["net_rate"]
        f["cmd_source"] = r["cmd_source"]
        rec.put(k, **f)

    loop = CoupledLoop(eng, plant, dec, cfg, on_interval=on_interval)
    loop.run(args.n_steps)
    extra = dict(coupling=cfg.record(), perch_baseline_hz=base.tolist(), persist_net_rate_hz=persist,
                 wall_s=time.time() - t0, peak_rss_gb=_peak_rss_gb(), seed=args.seed,
                 engine_state=_jsonable(eng.state_dict()))
    out = rec.close(extra)
    res = dict(path=str(out), vs_v1=compare(out, args.v1), persist_net_rate_hz=persist, wall_s=time.time() - t0)
    Path(str(out).replace("_data.h5", "_report.json")).write_text(json.dumps(res, indent=1))
    log(json.dumps(res["vs_v1"], indent=1))
    return res


def compare(new_path, v1_key: str) -> dict:
    v1 = V1_DIR / V1_RECORDS[v1_key]
    with h5py.File(new_path, "r") as f, h5py.File(v1, "r") as g:
        a, b = f["behavior"], g["behavior"]
        n = min(len(a["heading"]), len(b["heading"]))
        dh = np.degrees(np.abs(np.angle(np.exp(1j * (a["heading"][:n] - b["heading"][:n])))))
        wz = a["omega"][:n, 2] - b["omega"][:n, 2]
        return dict(n_steps=int(n), heading_diff_deg=dict(max=float(dh.max()), final=float(dh[-1]),
                                                          rms=float(np.sqrt((dh ** 2).mean()))),
                    yaw_rate_diff_rad_s_max=float(np.abs(wz).max()),
                    turn_dn_diff_max=float(np.abs(a["turn_brain"][:n] - b["turn_dn"][:n]).max()),
                    turn_dn_new_absmax=float(np.abs(a["turn_brain"][:n]).max()),
                    turn_dn_v1_absmax=float(np.abs(b["turn_dn"][:n]).max()),
                    heading_total_change_deg=dict(
                        new=float(np.degrees(np.unwrap(a["heading"][:n])[-1] - a["heading"][0])),
                        v1=float(np.degrees(np.unwrap(b["heading"][:n])[-1] - b["heading"][0]))),
                    pos_diff_mm_max=float(np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1).max()))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--v1", choices=tuple(V1_RECORDS), required=True, help="v1 record to reproduce / compare")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "hover_yaw"))
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--interval-ms", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--calib-steps", type=int, default=10)
    ap.add_argument("--persist-steps", type=int, default=8)
    ap.add_argument("--perturb-t", type=float, default=0.5)
    ap.add_argument("--perturb-deg", type=float, default=30.0)
    ap.add_argument("--ablate-steer", action="store_true", help="v1 control: readout replaced by the perch baseline")
    ap.add_argument("--zero-turn", action="store_true", help="control arm zero_turn: yaw command exactly 0")
    ap.add_argument("--waive", nargs="*", default=["DNp15"])
    return ap.parse_args(argv)


def main(argv=None):
    res = run(parse_args(argv))
    return 0
