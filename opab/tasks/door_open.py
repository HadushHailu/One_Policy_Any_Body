"""Door open task: grip handle, pull door open."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class DoorOpenTask(BaseTask):
    """Grip the door handle bar and pull to open the door."""

    name = "door_open"

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        dpos = placement.door_pos
        ds = placement.door_scale

        return f"""
    <body name="door_frame" pos="{dpos[0]} {dpos[1]} {dpos[2]}">
      <!-- Frame post (hinge side) -->
      <geom name="door_frame_post" type="box" size="{0.010*ds} {0.010*ds} {0.080*ds}"
            pos="{-0.080*ds} 0 {0.080*ds}" rgba="0.40 0.35 0.30 1"
            contype="1" conaffinity="1"/>
      <!-- Door panel (rotates around Z hinge) -->
      <body name="door_panel" pos="{-0.080*ds} 0 {0.080*ds}">
        <joint name="door_hinge" type="hinge" axis="0 0 1"
               range="0 1.57" damping="2" stiffness="0"/>
        <geom name="door_panel_geom" type="box"
              size="{0.075*ds} {0.005*ds} {0.075*ds}"
              pos="{0.075*ds} 0 0" rgba="0.85 0.80 0.70 1"
              contype="1" conaffinity="1"/>
        <!-- Door handle (L-shaped lever, horizontal) -->
        <body name="door_handle_body" pos="{0.120*ds} {-0.025*ds} 0">
          <!-- Handle stem -->
          <geom name="door_handle_stem" type="capsule" size="{0.006*ds}"
                fromto="0 0 0 0 {-0.045*ds} 0"
                rgba="0.85 0.65 0.10 1"
                contype="1" conaffinity="1"/>
          <!-- Handle bar (L shape) -->
          <geom name="door_handle_geom" type="capsule" size="{0.007*ds}"
                fromto="0 {-0.045*ds} 0 {-0.045*ds} {-0.045*ds} 0"
                rgba="0.85 0.65 0.10 1"
                contype="1" conaffinity="1"
                solref="0.005 1" solimp="0.99 0.99 0.001" condim="4" friction="1 0.005 0.0001"/>
          <site name="door_handle_site" pos="{-0.020*ds} {-0.045*ds} 0" size="0.002"/>
        </body>
      </body>
    </body>
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "door_hinge": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge"),
            "door_handle_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "door_handle_site"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        joint_id = ids["door_hinge"]
        if joint_id < 0:
            return False
        qpos_addr = model.jnt_qposadr[joint_id]
        angle = float(data.qpos[qpos_addr])
        return angle >= 1.047  # >= 60 degrees
