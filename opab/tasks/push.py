"""Push task: push cube to target zone (no lifting)."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class PushTask(BaseTask):
    """Push a cube along the table surface to a target location."""

    name = "push"

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        p = placement
        cube_pos = p.cube_pos
        target_pos = p.target_pos
        return f"""
    <body name="cube" pos="{cube_pos[0]} {cube_pos[1]} {cube_pos[2]}">
      <freejoint name="cube_joint" />
      <geom name="cube_geom" type="box" size="{p.cube_size} {p.cube_size} {p.cube_size}"
            mass="{p.cube_mass}" rgba="0.9 0.1 0.1 1" condim="4"
            friction="1.0 0.005 0.0001" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" />
    </body>

    <site name="target_zone" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}"
          size="{p.success_threshold}" rgba="0 1 0 0.3" type="sphere" />
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "cube_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
            "cube_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"),
            "target_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_zone"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        cube_body_id = ids["cube_body"]
        target_site_id = ids["target_site"]
        if cube_body_id < 0 or target_site_id < 0:
            return False

        cube_pos = data.xpos[cube_body_id]
        target_pos = data.site_xpos[target_site_id]
        dist_xy = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        return dist_xy < placement.success_threshold

    def randomize_reset(self, model: mujoco.MjModel, data: mujoco.MjData,
                        ids: dict[str, int], placement: TaskPlacement,
                        rng: np.random.Generator) -> None:
        cube_joint_id = ids["cube_joint"]
        if cube_joint_id >= 0:
            qpos_adr = model.jnt_qposadr[cube_joint_id]
            r = placement.cube_randomize_range
            data.qpos[qpos_adr] += rng.uniform(-r, r)
            data.qpos[qpos_adr + 1] += rng.uniform(-r, r)
