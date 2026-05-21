"""Stack task: pick cube_A, stack on top of cube_B."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class StackTask(BaseTask):
    """Pick cube_A and stack it on top of cube_B."""

    name = "stack"

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        p = placement
        cube_pos = p.cube_pos
        target_pos = p.target_pos
        # Cube B sits at the target position, at same height as cube A
        cube_b_pos = target_pos.copy()
        cube_b_pos[2] = cube_pos[2]

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

    <body name="cube_b" pos="{cube_b_pos[0]} {cube_b_pos[1]} {cube_b_pos[2]}">
      <freejoint name="cube_b_joint" />
      <geom name="cube_b_geom" type="box" size="{p.cube_size} {p.cube_size} {p.cube_size}"
            mass="{p.cube_mass}" rgba="0.1 0.1 0.9 1" condim="4"
            friction="1.0 0.005 0.0001" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" />
    </body>
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "cube_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
            "cube_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"),
            "cube_b_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b"),
            "cube_b_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_joint"),
            "target_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_zone"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        cube_body_id = ids["cube_body"]
        cube_b_body_id = ids["cube_b_body"]
        if cube_body_id < 0 or cube_b_body_id < 0:
            return False

        cube_a_pos = data.xpos[cube_body_id]
        cube_b_pos = data.xpos[cube_b_body_id]

        dist_xy = np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2])
        expected_z = cube_b_pos[2] + 2 * placement.cube_size
        height_ok = abs(cube_a_pos[2] - expected_z) < 0.02

        return dist_xy < placement.success_threshold and height_ok

    def randomize_reset(self, model: mujoco.MjModel, data: mujoco.MjData,
                        ids: dict[str, int], placement: TaskPlacement,
                        rng: np.random.Generator) -> None:
        # Randomize cube A
        cube_joint_id = ids["cube_joint"]
        if cube_joint_id >= 0:
            qpos_adr = model.jnt_qposadr[cube_joint_id]
            r = placement.cube_randomize_range
            data.qpos[qpos_adr] += rng.uniform(-r, r)
            data.qpos[qpos_adr + 1] += rng.uniform(-r, r)
        # Randomize cube B
        cube_b_joint_id = ids["cube_b_joint"]
        if cube_b_joint_id >= 0:
            qpos_adr = model.jnt_qposadr[cube_b_joint_id]
            r = placement.cube_randomize_range
            data.qpos[qpos_adr] += rng.uniform(-r, r)
            data.qpos[qpos_adr + 1] += rng.uniform(-r, r)
