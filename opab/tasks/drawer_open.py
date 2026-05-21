"""Drawer open task: pull drawer open past threshold."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class DrawerOpenTask(BaseTask):
    """Grasp drawer handle and pull it open."""

    name = "drawer_open"

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        p = placement
        table_top_z = p.table_pos[2] + p.table_half_size[2]

        # Cabinet dimensions
        cab_w = p.drawer_size[0]
        cab_d = p.drawer_size[1]
        cab_h = p.drawer_size[2]
        wall = 0.008
        slide_range = p.drawer_slide_range

        cab_pos = p.drawer_pos.copy()
        cab_pos[2] = table_top_z + cab_h

        # Inner drawer dimensions
        inner_w = cab_w - wall - 0.001
        inner_d = cab_d - wall - 0.001
        inner_h = cab_h - wall - 0.001

        # Handle
        handle_offset = 0.035
        handle_half_len = cab_w * 0.6

        return f"""
    <!-- Cabinet housing (static) -->
    <body name="cabinet" pos="{cab_pos[0]} {cab_pos[1]} {cab_pos[2]}">
      <!-- Top -->
      <geom name="cab_top" type="box" size="{cab_w} {cab_d} {wall/2}"
            pos="0 0 {cab_h}" rgba="0.55 0.35 0.2 1" contype="1" conaffinity="1" />
      <!-- Bottom -->
      <geom name="cab_bottom" type="box" size="{cab_w} {cab_d} {wall/2}"
            pos="0 0 -{cab_h}" rgba="0.55 0.35 0.2 1" contype="1" conaffinity="1" />
      <!-- Back wall -->
      <geom name="cab_back" type="box" size="{cab_w} {wall/2} {cab_h - wall}"
            pos="0 {cab_d} 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />
      <!-- Left wall -->
      <geom name="cab_left" type="box" size="{wall/2} {cab_d} {cab_h - wall}"
            pos="-{cab_w} 0 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />
      <!-- Right wall -->
      <geom name="cab_right" type="box" size="{wall/2} {cab_d} {cab_h - wall}"
            pos="{cab_w} 0 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />

      <!-- Sliding drawer body -->
      <body name="drawer_body" pos="0 0 0">
        <joint name="drawer_joint" type="slide" axis="0 -1 0"
               range="0 {slide_range}" damping="0.5" frictionloss="0.1" />
        <!-- Inner bottom -->
        <geom name="drawer_bottom" type="box" size="{inner_w} {inner_d} {wall/2}"
              pos="0 0 -{cab_h - wall}" mass="0.1" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner back -->
        <geom name="drawer_inner_back" type="box" size="{inner_w} {wall/2} {inner_h}"
              pos="0 {inner_d} 0" mass="0.05" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner left -->
        <geom name="drawer_inner_left" type="box" size="{wall/2} {inner_d} {inner_h}"
              pos="-{inner_w} 0 0" mass="0.03" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner right -->
        <geom name="drawer_inner_right" type="box" size="{wall/2} {inner_d} {inner_h}"
              pos="{inner_w} 0 0" mass="0.03" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Front panel (face of drawer) -->
        <geom name="drawer_front" type="box" size="{cab_w - 0.001} {wall/2} {cab_h - 0.001}"
              pos="0 -{cab_d} 0" mass="0.08" rgba="0.6 0.4 0.25 1"
              contype="1" conaffinity="1" />
        <!-- Bar handle (horizontal, centered on front panel) -->
        <geom name="handle_bar" type="capsule" size="0.006"
              fromto="-{handle_half_len} -{cab_d + handle_offset} 0 {handle_half_len} -{cab_d + handle_offset} 0"
              mass="0.02" rgba="0.75 0.75 0.75 1" condim="6"
              friction="2.0 0.1 0.01" contype="1" conaffinity="1" />
        <!-- Handle connectors -->
        <geom name="handle_conn_l" type="cylinder" size="0.005 {handle_offset/2}"
              pos="-{handle_half_len} -{cab_d + handle_offset/2} 0" euler="1.5708 0 0"
              rgba="0.75 0.75 0.75 1" contype="1" conaffinity="1" />
        <geom name="handle_conn_r" type="cylinder" size="0.005 {handle_offset/2}"
              pos="{handle_half_len} -{cab_d + handle_offset/2} 0" euler="1.5708 0 0"
              rgba="0.75 0.75 0.75 1" contype="1" conaffinity="1" />
        <site name="drawer_handle_site" pos="0 -{cab_d + handle_offset} 0" size="0.005" rgba="1 1 0 0.5" />
      </body>
    </body>
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "drawer_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint"),
            "drawer_handle_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        joint_id = ids["drawer_joint"]
        if joint_id < 0:
            return False
        qpos_idx = model.jnt_qposadr[joint_id]
        drawer_pos = data.qpos[qpos_idx]
        target = placement.drawer_slide_range * 0.8
        return drawer_pos >= target
