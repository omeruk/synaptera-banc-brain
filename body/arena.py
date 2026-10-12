"""Frozen arena specification (replaces the module globals of synaptera@c94796d:flight/config.py).

``ArenaSpec`` holds geometry, physics step and texture settings; ``with_start_offset`` replaces
``move_takeoff_pedestal`` (which mutated module globals that every user then read at call time). Values come
from ``configs/arena_x20.yaml`` (each with an origin class).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from configs import load


def _box(b) -> tuple:
    return tuple(tuple(float(v) for v in ax) for ax in b)


def point_box_distance(p, box) -> float:
    lo = np.array([b[0] for b in box])
    hi = np.array([b[1] for b in box])
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - np.asarray(p, float), np.asarray(p, float) - hi))))


@dataclass(frozen=True)
class ArenaSpec:
    scale: float
    towers: tuple               # (box, box); box = ((x0, x1), (y0, y1), (z0, z1))
    takeoff_pedestal: tuple     # (cx, cy, r, z_top)
    food_platform: tuple
    food_pos: tuple
    food_drop_radius: float
    start_pos: tuple
    arena_half_size: tuple
    odor_grid_bounds: tuple
    odor_grid_res: float
    odor_d0: float
    physics_dt: float
    obstacle_solimp: str
    obstacle_solref: str
    texture_rgb: tuple
    obstacle_tex_period: float
    ground_tex_period: float
    platform_dark_rgba: tuple

    @property
    def tower1(self):
        return self.towers[0]

    @property
    def tower2(self):
        return self.towers[1]

    def physics_steps(self, interval_s: float) -> int:
        """Physics steps in one coupling interval (must be a whole number)."""
        n = interval_s / self.physics_dt
        if abs(n - round(n)) > 1e-6 or round(n) < 1:
            raise ValueError(f"interval {interval_s} s is not a multiple of physics_dt {self.physics_dt} s")
        return int(round(n))

    def with_start_offset(self, dx: float, dy: float) -> "ArenaSpec":
        """New spec with the take-off pedestal, and the start with it, shifted by (dx, dy) mm (experiment
        design, v1 --start-offset). The pedestal must stay inside the odour grid and clear of the towers."""
        x, y, r, zt = self.takeoff_pedestal
        ped = (x + float(dx), y + float(dy), r, zt)
        start = (self.start_pos[0] + float(dx), self.start_pos[1] + float(dy), self.start_pos[2])
        cx, cy = ped[:2]
        (gx0, gx1), (gy0, gy1), _ = self.odor_grid_bounds
        if not (gx0 < cx - r and cx + r < gx1 and gy0 < cy - r and cy + r < gy1):
            raise ValueError(f"pedestal at ({cx:g}, {cy:g}) leaves the odor grid {self.odor_grid_bounds[:2]}")
        for (x0, x1), (y0, y1), _ in self.towers:
            if x0 - r <= cx <= x1 + r and y0 - r <= cy <= y1 + r:
                raise ValueError(f"pedestal at ({cx:g}, {cy:g}) overlaps a tower")
        return replace(self, takeoff_pedestal=ped, start_pos=start)

    def surface_below(self, x: float, y: float) -> float:
        """Height of the highest surface under (x, y): ground, tower tops, cylinder tops."""
        h = 0.0
        for (x0, x1), (y0, y1), (_, z1) in self.towers:
            if x0 <= x <= x1 and y0 <= y <= y1:
                h = max(h, z1)
        for cx, cy, r, zt in (self.takeoff_pedestal, self.food_platform):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                h = max(h, zt)
        return h

    def tower_clearance(self, p) -> float:
        return min(point_box_distance(p, b) for b in self.towers)

    def geometry_record(self) -> dict:
        """The geometry as written to the run record."""
        return dict(tower1=self.tower1, tower2=self.tower2, pedestal=self.takeoff_pedestal,
                    food_platform=self.food_platform, food=self.food_pos)


def load_arena(name: str = "arena_x20", path=None) -> ArenaSpec:
    c = load(name, path)
    return ArenaSpec(
        scale=c["scale"], towers=(_box(c["tower1"]), _box(c["tower2"])),
        takeoff_pedestal=tuple(c["takeoff_pedestal"]), food_platform=tuple(c["food_platform"]),
        food_pos=tuple(c["food_pos"]), food_drop_radius=c["food_drop_radius"], start_pos=tuple(c["start_pos"]),
        arena_half_size=tuple(c["arena_half_size"]), odor_grid_bounds=_box(c["odor_grid_bounds"]),
        odor_grid_res=c["odor_grid_res"], odor_d0=c["odor_d0"], physics_dt=c["physics_dt"],
        obstacle_solimp=c["obstacle_solimp"], obstacle_solref=c["obstacle_solref"],
        texture_rgb=tuple(tuple(v) for v in c["texture_rgb"]),
        obstacle_tex_period=c["obstacle_tex_period"], ground_tex_period=c["ground_tex_period"],
        platform_dark_rgba=tuple(c["platform_dark_rgba"]))
