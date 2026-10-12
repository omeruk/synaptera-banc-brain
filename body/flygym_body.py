"""FlyGym body + arena for flight; stroke-averaged forces applied through the thorax xfrc.

Ported from synaptera@c94796d:flight/body.py. Changes with respect to v1:
  * ``step(cmd, dt, ...)`` takes a duration; the number of physics steps comes from the arena spec (no fixed
    250 steps per 25 ms decision).
  * FlyGym's private API is used only through body/flygym_private.py (review M6).
  * Hand-made inner loops (haltere attitude PD, head reflex) are :class:`InnerLoop` objects supplied by the
    caller (REFLEX arm, motor/legacy_reflexes.py); the body no longer knows them by name.
  * Seeded physical noise (body/noise.py, review M5) and an amplitude-clip record (review H5).
  * The arena comes from an :class:`ArenaSpec` instead of module globals.
Arena: FlatTerrain + two towers (boxes), take-off pedestal and food platform (cylinders). Obstacle geoms carry
the softening solimp/solref; arena geoms have contype = conaffinity = 0 so only the explicit pairs collide.
Textures are visual-only meshes ("viz_tex_*"); platform="dark" makes the food platform and droplet uniformly
dark. The fly collides with the arena with body, head, wings and legs (antennae, aristae, halteres excluded).
"""
from __future__ import annotations

from typing import Protocol, Sequence

import mujoco
import numpy as np
from flygym import Fly, SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.state import KinematicPose

from body import flygym_private as private
from body import quasi_steady as qs
from body.arena import ArenaSpec, load_arena, point_box_distance
from configs import load

OBSTACLES = ("tower1", "tower2", "pedestal", "food_platform")
TOWERS = ("tower1", "tower2")
EXCLUDED_COLLISION_PARTS = ("Pedicel", "Funiculus", "Arista", "Haltere")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
NECK_JOINTS = {"yaw": "joint_Head_roll", "pitch": "joint_Head", "roll": "joint_Head_yaw"}
NECK_AXES = ("yaw", "pitch", "roll")


class InnerLoop(Protocol):
    """A hand-made loop run every physics sub-step.

    update(R, v, omega, dt) -> object with optional ``torque`` / ``force`` (world frame, uN mm / uN) and
    ``neck_target`` ((yaw, pitch, roll) rad); R: body->world (3x3), v: COM velocity, omega: angular velocity."""

    label: str

    def update(self, R, v, omega, dt): ...


def _box_mesh(size, period):
    """Box (half-extents `size`) as 6 separate quads with texcoords in units of `period` (mm), so a 2D checker
    has the same period on every face (MuJoCo does not tile 2D textures on box / cylinder primitives)."""
    hx, hy, hz = size
    verts, uvs, faces = [], [], []
    for ax, sg, a, b in ((0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 1, 0)):
        h = np.array([hx, hy, hz])
        n0 = len(verts)
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[ax] = sg * h[ax]
            p[a] = du * h[a]
            p[b] = dv * h[b]
            verts.append(p)
            uvs.append((p[a] / period, p[b] / period))
        faces += [(n0, n0 + 1, n0 + 2), (n0, n0 + 2, n0 + 3)]
    return np.array(verts), np.array(uvs), np.array(faces)


def _cylinder_mesh(r, hz, period, n=48):
    """Closed cylinder (radius r, half-height hz): side texcoords = (arc length, z) / period, caps = (x, y)."""
    verts, uvs, faces = [], [], []
    ang = np.linspace(0, 2 * np.pi, n + 1)
    for z in (-hz, hz):
        for a in ang:
            verts.append((r * np.cos(a), r * np.sin(a), z))
            uvs.append((r * a / period, z / period))
    for i in range(n):
        b0, t0 = i, n + 1 + i
        faces += [(b0, b0 + 1, t0 + 1), (b0, t0 + 1, t0)]
    for z, sg in ((-hz, -1), (hz, 1)):
        c = len(verts)
        verts.append((0.0, 0.0, z))
        uvs.append((0.0, 0.0))
        ring = []
        for a in ang[:-1]:
            ring.append(len(verts))
            verts.append((r * np.cos(a), r * np.sin(a), z))
            uvs.append((r * np.cos(a) / period, r * np.sin(a) / period))
        for i in range(n):
            j, k = ring[i], ring[(i + 1) % n]
            faces.append((c, j, k) if sg > 0 else (c, k, j))
    return np.array(verts), np.array(uvs), np.array(faces)


class FlightArena(FlatTerrain):
    """textured=True: obstacles are drawn by visual-only checker meshes over the collision primitives, which are
    made invisible (alpha 0) but keep their names, solimp / solref and contacts."""

    def __init__(self, spec: ArenaSpec, textured=True, platform="neutral"):
        if platform not in ("dark", "neutral"):
            raise ValueError(platform)
        super().__init__(size=spec.arena_half_size)
        dark = platform == "dark"
        wb = self.root_element.worldbody
        asset = self.root_element.asset
        soft = dict(solimp=spec.obstacle_solimp, solref=spec.obstacle_solref, contype=0, conaffinity=0)
        hidden = dict(rgba=(1, 1, 1, 0))
        if textured:
            rgb1, rgb2 = spec.texture_rgb
            ground = asset.find("material", "grid")
            ground.texture.rgb1, ground.texture.rgb2 = rgb1, rgb2
            ground.texrepeat = tuple(2 * h / spec.ground_tex_period for h in spec.arena_half_size)
            tex = asset.add("texture", name="obstacle_checker", type="2d", builtin="checker",
                            width=64, height=64, rgb1=rgb1, rgb2=rgb2)
            mat = asset.add("material", name="obstacle_mat", texture=tex)

            def viz_mesh(name, pos, mesh, rgba=None):
                v, uv, f = mesh
                asset.add("mesh", name=f"mesh_{name}", vertex=v.ravel(), texcoord=uv.ravel(), face=f.ravel())
                look = dict(rgba=rgba) if rgba is not None else dict(material=mat, rgba=(1, 1, 1, 1))
                wb.add("geom", type="mesh", name=f"viz_tex_{name}", mesh=f"mesh_{name}", pos=pos,
                       **look, contype=0, conaffinity=0, mass=0)
        for name, box in (("tower1", spec.tower1), ("tower2", spec.tower2)):
            (x0, x1), (y0, y1), (z0, z1) = box
            pos = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
            size = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
            wb.add("geom", type="box", name=name, pos=pos, size=size,
                   **(hidden if textured else dict(rgba=(0.55, 0.55, 0.6, 1.0))), **soft)
            if textured:
                viz_mesh(name, pos, _box_mesh(size, spec.obstacle_tex_period))
        for name, (cx, cy, r, zt), rgba in (("pedestal", spec.takeoff_pedestal, (0.4, 0.3, 0.2, 1)),
                                            ("food_platform", spec.food_platform, (0.3, 0.5, 0.3, 1))):
            is_dark = dark and name == "food_platform"
            if is_dark:
                rgba = spec.platform_dark_rgba
            wb.add("geom", type="cylinder", name=name, pos=(cx, cy, zt / 2), size=(r, zt / 2),
                   **(hidden if textured else dict(rgba=rgba)), **soft)
            if textured:
                viz_mesh(name, (cx, cy, zt / 2), _cylinder_mesh(r, zt / 2, spec.obstacle_tex_period),
                         rgba=spec.platform_dark_rgba if is_dark else None)
        wb.add("geom", type="sphere", name="viz_food_drop", pos=tuple(spec.food_pos),
               size=(spec.food_drop_radius,), rgba=spec.platform_dark_rgba if dark else (0.9, 0.7, 0.1, 1),
               contype=0, conaffinity=0)


class FlightFly(Fly):
    """Fly whose arena collision set is body + legs, without pairs to viz_* geoms."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.floor_collisions = [
            g.name for g in self.model.find_all("geom")
            if g.name and not any(p in g.name for p in EXCLUDED_COLLISION_PARTS)
        ]

    def init_floor_contacts(self, arena):
        super().init_floor_contacts(arena)
        for key, _ in private.floor_contact_items(self):
            if key.startswith("viz_"):
                private.drop_floor_contact(self, key)


def pose_from_offsets(joint_names, stand, offsets):
    """stand + offsets (deg, per leg part; R-side roll / yaw mirrored), actuated-joint order."""
    pose = stand.copy()
    for part, dv in offsets.items():
        for side in "LR":
            j = joint_names.index(f"joint_{side}{part}")
            mirror = -1 if (side == "R" and ("roll" in part or "yaw" in part)) else 1
            pose[j] += np.radians(dv) * mirror
    return pose


class FlightBody:
    def __init__(self, arena: ArenaSpec | None = None, wing: dict | None = None, spawn_pos=None, spawn_yaw=0.0,
                 legs="stand", enable_adhesion=True, enable_vision=False, vision_refresh_rate=40, textured=True,
                 platform="neutral", neck_actuators=False, postures=False):
        self.arena = load_arena() if arena is None else arena
        self.wing = load("body_flight") if wing is None else wing
        a = self.arena
        if spawn_pos is None:  # standing on the take-off pedestal
            spawn_pos = (a.takeoff_pedestal[0], a.takeoff_pedestal[1], a.takeoff_pedestal[3] + 0.5)
        joint_names = list(Fly().actuated_joints)
        self.stand_pose = np.array(PreprogrammedSteps().default_pose, dtype=float)
        self.tuck_pose = pose_from_offsets(joint_names, self.stand_pose, self.wing["tuck_pose_offsets"])
        self.feed_pose = None
        if postures:          # flight pose replaces the tuck pose, plus a feeding pose (HAND)
            self.tuck_pose = pose_from_offsets(joint_names, self.stand_pose, self.wing["flight_pose_offsets"])
            self.feed_pose = pose_from_offsets(joint_names, self.stand_pose, self.wing["feed_pose_offsets"])
        init = self.stand_pose if legs == "stand" else self.tuck_pose
        self.fly = FlightFly(
            spawn_pos=tuple(spawn_pos), spawn_orientation=(0.0, 0.0, spawn_yaw),
            init_pose=KinematicPose(dict(zip(joint_names, init))),
            enable_adhesion=enable_adhesion, enable_vision=enable_vision,
            vision_refresh_rate=vision_refresh_rate,
        )
        neck_acts = None
        if neck_actuators:    # position actuators on the three neck joints (driven by a head InnerLoop)
            from motor.legacy_reflexes import HeadReflex
            kp = HeadReflex().kp
            neck_acts = [self.fly.model.actuator.add("position", name=f"actuator_neck_{ax}",
                                                     joint=self.fly.model.find("joint", NECK_JOINTS[ax]),
                                                     kp=kp, ctrllimited=False, forcelimited=False)
                         for ax in NECK_AXES]
        self.sim = SingleFlySimulation(fly=self.fly, arena=FlightArena(a, textured=textured, platform=platform),
                                       cameras=[], timestep=a.physics_dt)
        self.sim.reset()
        self.enable_adhesion = enable_adhesion
        self.physics = self.sim.physics
        self.m = self.physics.model.ptr
        self.d = self.physics.data.ptr
        m = self.m
        name = self.fly.name
        self.root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}/")
        self.thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}/Thorax")
        # The free root body carries a 1e-6 g dummy inertial whose ipos equals the spawn position (xipos =
        # 2*spawn): it adds 1e-6*|spawn|^2 g mm^2 to the real dynamics. Put it at the body origin.
        m.body_ipos[self.root] = 0.0
        mujoco.mj_forward(m, self.d)
        self.obstacle_geom = {g: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in OBSTACLES}
        self.ground_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.tower_ids = np.array([self.obstacle_geom[t] for t in TOWERS])
        self.tarsus_leg = np.full(m.ngeom, -1, dtype=int)     # fly geom id -> leg index (tarsus), -1 otherwise
        self.fly_geom = np.zeros(m.ngeom, dtype=bool)
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname.startswith(f"{name}/"):
                self.fly_geom[gid] = True
                part = gname[len(name) + 1:]
                if "Tarsus" in part:
                    self.tarsus_leg[gid] = LEGS.index(part[:2])
        self.mass = float(m.body_subtreemass[self.root])
        self.params = qs.make_params(self.mass, float(-m.opt.gravity[2]), self.inertia_body(), self.wing)
        self.legs = legs
        self.leg_target = init.copy()          # PD joint target applied at the last physics step
        self._ramp = None                      # (start, end, n_done, n_total) physics steps
        self._act_ids = np.asarray(self.physics.bind(self.fly.actuators).element_id)
        self.adhesion = np.zeros(6)
        self.badqacc0 = self.badqacc_count()
        self.max_tower_penetration = 0.0
        self.clip_events = 0                   # physics sub-steps with a clipped stroke amplitude (H5)
        self._neck_ids = self._neck_qadr = None
        if neck_acts is not None:
            self._neck_ids = np.asarray(self.physics.bind(neck_acts).element_id)
            jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/{NECK_JOINTS[ax]}") for ax in NECK_AXES]
            self._neck_qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.head_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}/Head")

    # ── model quantities ────────────────────────────────────────────────────
    def inertia_body(self):
        """Composite inertia of the whole fly about its COM, in thorax axes."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        com = d.subtree_com[self.root].copy()
        I = np.zeros((3, 3))
        for b in range(m.nbody):
            if m.body_rootid[b] != self.root or m.body_mass[b] == 0:
                continue
            Rb = d.ximat[b].reshape(3, 3)
            r = d.xipos[b] - com
            I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        Rt = d.xmat[self.thorax].reshape(3, 3)
        return Rt.T @ I @ Rt

    def yaw_pulse_torque(self, deg, duration):
        """Body-z torque (uN mm) that, against the total yaw damping alone (turn command 0), holds the yaw rate
        deg/duration: a first-order system then turns by ``deg`` in total (yaw perturbation; + = left / CCW)."""
        return self.params.d_yaw_total * np.radians(deg) / duration

    def badqacc_count(self):
        return int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    # ── state ───────────────────────────────────────────────────────────────
    def state(self):
        m, d = self.m, self.d
        mujoco.mj_subtreeVel(m, d)
        R = d.xmat[self.thorax].reshape(3, 3).copy()
        return dict(pos=d.subtree_com[self.root].copy(), vel=d.subtree_linvel[self.root].copy(),
                    R=R, quat=d.xquat[self.thorax].copy(), omega=d.cvel[self.thorax][:3].copy(),
                    heading=qs.heading_of(R), time=self.sim.curr_time)

    def contacts(self):
        """Current contacts: tarsus-platform legs, tower contact, max tower penetration."""
        d = self.d
        n = d.ncon
        legs_on = {g: np.zeros(6, dtype=bool) for g in ("food_platform", "pedestal", "ground")}
        tower, pen = False, 0.0
        if n:
            for a, b, dd in zip(d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]):
                other, flyg = (b, a) if self.fly_geom[a] else (a, b)
                if not self.fly_geom[flyg]:
                    continue
                if other in self.tower_ids:
                    tower = True
                    pen = max(pen, -dd)
                for gname in legs_on:
                    gid = self.ground_geom if gname == "ground" else self.obstacle_geom[gname]
                    if other == gid and self.tarsus_leg[flyg] >= 0:
                        legs_on[gname][self.tarsus_leg[flyg]] = True
        clearance = self.arena.tower_clearance(self.d.subtree_com[self.root])
        return dict(platform_legs=legs_on["food_platform"], pedestal_legs=legs_on["pedestal"],
                    ground_legs=legs_on["ground"], tower_contact=tower, tower_penetration=pen,
                    min_tower_clearance=clearance)

    # ── stepping ────────────────────────────────────────────────────────────
    def set_legs(self, legs, adhesion=None, ramp_s=None):
        """Leg pose "stand" / "tuck" / "feed" (feed only with postures=True). A pose change starts a cosine ramp of
        the PD joint target from the current target; repeating the current pose does not restart a running ramp."""
        if legs not in ("stand", "tuck", "feed") or (legs == "feed" and self.feed_pose is None):
            raise ValueError(legs)
        if legs != self.legs:
            if ramp_s is None:
                w = self.wing
                ramp_s = (w["leg_feed_ramp_s"] if "feed" in (legs, self.legs) else
                          w["leg_tuck_ramp_s"] if legs == "tuck" else w["leg_extend_ramp_s"])
            end = {"stand": self.stand_pose, "tuck": self.tuck_pose, "feed": self.feed_pose}[legs]
            n = max(1, int(round(ramp_s / self.arena.physics_dt)))
            self._ramp = (self.leg_target.copy(), end, 0, n)
        self.legs = legs
        if adhesion is not None:
            self.adhesion = np.full(6, float(adhesion)) if np.isscalar(adhesion) else np.asarray(adhesion, float)

    def step(self, cmd: qs.WingCommand, dt: float, inner: Sequence[InnerLoop] = (), ext_force=None,
             ext_torque=None, noise=None, snap_at=(), render_at=()):
        """Advance ``dt`` seconds (a whole number of physics steps) with a constant wing command.

        inner: hand-made inner loops run every sub-step (REFLEX arm). ext_force / ext_torque: constant extra
        world-frame force (uN) / torque (uN mm) on the thorax (gusts in tests); noise: a PhysicalNoise advanced
        once per call. snap_at: sub-step indices before which qpos is copied (replay); render_at: sub-step
        counts after which both eyes are rendered.
        Returns mean applied force / torque (world), max tower penetration, qpos / qvel snapshots, rendered
        frames, the number of sub-steps with a clipped amplitude and, when an inner loop drives the neck, the
        head statistics."""
        m, d = self.m, self.d
        n_steps = self.arena.physics_steps(dt)
        pdt = self.arena.physics_dt
        action = {"joints": self.leg_target}
        if self.enable_adhesion:
            action["adhesion"] = self.adhesion
        self.fly.pre_step(action, self.sim)
        F_acc, T_acc, pen_max, clipped = np.zeros(3), np.zeros(3), 0.0, 0
        p = self.params
        snap_at, render_at = set(snap_at), set(render_at)
        snaps, vsnaps, frames = [], [], []
        n_F, n_T = (np.zeros(3), np.zeros(3)) if noise is None else noise.step(dt)
        ext_F = n_F + (0.0 if ext_force is None else np.asarray(ext_force, float))
        ext_T = n_T + (0.0 if ext_torque is None else np.asarray(ext_torque, float))
        head = None
        h_gaze = h_body = 0.0
        h_turn = h_turn_lt = 0
        h_qmax = np.zeros(3)
        h_target = np.zeros(3)
        turn_rate = max([getattr(l, "turn_rate", 0.0) for l in inner] + [0.0])
        for k in range(n_steps):
            if k in snap_at:
                snaps.append(d.qpos.copy())
                vsnaps.append(d.qvel.copy())
            mujoco.mj_subtreeVel(m, d)
            R = d.xmat[self.thorax].reshape(3, 3)
            com = d.subtree_com[self.root]
            v = d.subtree_linvel[self.root]
            w = d.cvel[self.thorax][:3]
            F = np.zeros(3)
            T = np.zeros(3)
            if cmd.on:
                F, T, info = qs.aero_wrench(R, v, w, cmd, p)
                clipped += int(info["clipped_L"] or info["clipped_R"])
            neck = None
            for loop in inner:
                out = loop.update(R, v, w, pdt)
                if getattr(out, "torque", None) is not None:
                    T = T + out.torque
                if getattr(out, "force", None) is not None:
                    F = F + out.force
                if getattr(out, "neck_target", None) is not None:
                    neck = out.neck_target
            if cmd.on or inner:
                # xfrc acts at the thorax COM: shift the force to the whole-fly COM
                d.xfrc_applied[self.thorax, :3] = F
                d.xfrc_applied[self.thorax, 3:] = T + np.cross(com - d.xipos[self.thorax], F)
                F_acc += F
                T_acc += T
            else:
                d.xfrc_applied[self.thorax] = 0.0
            d.xfrc_applied[self.thorax, :3] += ext_F
            d.xfrc_applied[self.thorax, 3:] += ext_T
            d.ctrl[self._act_ids] = self._advance_leg_target()
            if neck is not None:
                if self._neck_ids is None:
                    raise RuntimeError("an inner loop drives the neck but the body has no neck actuators")
                d.ctrl[self._neck_ids] = neck
                h_target = neck
            self.physics.step()
            if neck is not None:
                z_t = d.xmat[self.thorax].reshape(3, 3)[:, 2]
                wb = float(d.cvel[self.thorax][:3] @ z_t)
                wg = float(d.cvel[self.head_body][:3] @ z_t)
                h_gaze += wg
                h_body += wb
                if abs(wb) > turn_rate:
                    h_turn += 1
                    h_turn_lt += int(abs(wg) < abs(wb))
                np.maximum(h_qmax, np.abs(d.qpos[self._neck_qadr]), out=h_qmax)
                head = True
            nc = d.ncon
            if nc:
                g1, g2 = d.contact.geom1[:nc], d.contact.geom2[:nc]
                hit = np.isin(g1, self.tower_ids) | np.isin(g2, self.tower_ids)
                if hit.any():
                    pen_max = max(pen_max, float(-d.contact.dist[:nc][hit].min()))
            if k + 1 in render_at:
                frames.append(self.update_vision().copy())
        self.sim.curr_time += n_steps * pdt
        self.max_tower_penetration = max(self.max_tower_penetration, pen_max)
        self.clip_events += clipped
        return dict(force=F_acc / n_steps, torque=T_acc / n_steps, tower_penetration=pen_max,
                    badqacc=self.badqacc_count() - self.badqacc0, qpos=snaps, qvel=vsnaps, frames=frames,
                    clipped_substeps=clipped, n_substeps=n_steps,
                    head=dict(q=d.qpos[self._neck_qadr].copy(), target=np.asarray(h_target).copy(), q_absmax=h_qmax,
                              gaze_yaw_rate=h_gaze / n_steps, body_yaw_rate=h_body / n_steps,
                              turn_sub=h_turn, turn_sub_lt=h_turn_lt) if head else None)

    def _advance_leg_target(self):
        """PD joint target for the next physics step (cosine ramp while a transition runs)."""
        if self._ramp is not None:
            start, end, i, n = self._ramp
            i += 1
            s = 0.5 * (1.0 - np.cos(np.pi * i / n))
            self.leg_target = start + s * (end - start)
            self._ramp = None if i >= n else (start, end, i, n)
        return self.leg_target

    @property
    def leg_ramp_active(self):
        return self._ramp is not None

    def update_vision(self):
        """Render both compound eyes now (FlyGym ommatidia readout, (2, 721, 2)). Needs enable_vision=True."""
        return private.render_eyes(self.fly, self.sim)

    def set_velocity(self, v_world):
        """Set the free-joint linear velocity (tests / initial conditions)."""
        adr = self.m.jnt_dofadr[self.m.body_jntadr[self.root]]
        self.d.qvel[adr:adr + 3] = v_world
        mujoco.mj_forward(self.m, self.d)
