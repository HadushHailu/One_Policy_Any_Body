"""Peg insertion task: pick peg, insert into hole."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class PegInsertionTask(BaseTask):
    """Pick up a cylindrical peg and insert it into a matching hole."""

    name = "peg_insertion"

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        p = placement
        peg_pos = p.cube_pos  # peg starts at cube position
        hole_pos = p.target_pos  # hole at target position

        peg_radius = p.peg_radius
        peg_half_length = p.peg_half_length
        hole_inner_r = peg_radius + p.hole_clearance
        hole_outer_r = hole_inner_r + 0.010
        hole_depth = p.hole_depth

        # Generate 32-segment hollow cylinder for the hole
        import math
        n_segments = 32
        segments = ""
        for i in range(n_segments):
            angle_deg = i * 360.0 / n_segments
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            r_mid = (hole_inner_r + hole_outer_r) / 2
            seg_half_width = r_mid * math.sin(math.pi / n_segments)
            wall_half = (hole_outer_r - hole_inner_r) / 2
            segments += f"""      <geom name="hole_seg_{i}" type="box"
            pos="{r_mid * cos_a} {r_mid * sin_a} 0"
            euler="0 0 {angle_deg}"
            size="{seg_half_width} {wall_half} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />\n"""

        return f"""
    <body name="peg" pos="{peg_pos[0]} {peg_pos[1]} {peg_pos[2]}">
      <freejoint name="peg_joint" />
      <geom name="peg_geom" type="cylinder" size="{peg_radius} {peg_half_length}"
            mass="0.02" rgba="0.2 0.6 0.9 1" condim="6"
            friction="2.0 0.1 0.01" contype="1" conaffinity="1" />
    </body>

    <body name="hole_body" pos="{hole_pos[0]} {hole_pos[1]} {hole_pos[2]}">
{segments}
      <!-- Visual inner cylinder (semi-transparent) -->
      <geom name="hole_void" type="cylinder" size="{hole_inner_r} {hole_depth + 0.001}"
            rgba="0.1 0.1 0.1 0.2" contype="0" conaffinity="0" />
      <!-- Success site at bottom -->
      <site name="hole_bottom" pos="0 0 -{hole_depth}" size="0.005" rgba="0 1 0 0.5" />
      <!-- Floor -->
      <geom name="hole_floor" type="cylinder" size="{hole_inner_r} 0.002"
            pos="0 0 -{hole_depth}"
            rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1" />
    </body>
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "peg_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg"),
            "peg_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_joint"),
            "hole_bottom_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_bottom"),
            "hole_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole_body"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        peg_body_id = ids["peg_body"]
        hole_body_id = ids["hole_body"]
        if peg_body_id < 0 or hole_body_id < 0:
            return False

        peg_pos = data.xpos[peg_body_id]
        hole_body_pos = data.xpos[hole_body_id]
        hole_inner_r = placement.peg_radius + placement.hole_clearance

        # XY: peg center must be within hole opening
        dist_xy = np.linalg.norm(peg_pos[:2] - hole_body_pos[:2])
        inside_xy = dist_xy < hole_inner_r

        # Z: peg center must be at or below hole top
        inserted_z = peg_pos[2] < hole_body_pos[2] + 0.002

        return inside_xy and inserted_z

    def randomize_reset(self, model: mujoco.MjModel, data: mujoco.MjData,
                        ids: dict[str, int], placement: TaskPlacement,
                        rng: np.random.Generator) -> None:
        peg_joint_id = ids["peg_joint"]
        if peg_joint_id >= 0:
            qpos_adr = model.jnt_qposadr[peg_joint_id]
            r = placement.cube_randomize_range * 0.5
            data.qpos[qpos_adr] += rng.uniform(-r, r)
            data.qpos[qpos_adr + 1] += rng.uniform(-r, r)
