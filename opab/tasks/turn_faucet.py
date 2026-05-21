"""Turn faucet task: rotate faucet handle past threshold."""
from __future__ import annotations

import mujoco
import numpy as np

from opab.tasks.base_task import BaseTask, TaskPlacement


class TurnFaucetTask(BaseTask):
    """Grip and rotate a faucet lever handle."""

    name = "turn_faucet"

    def generate_asset_xml(self, placement: TaskPlacement) -> str:
        """Faucet materials to inject into <asset> section."""
        return """
    <material name="faucet_chrome" rgba="0.75 0.78 0.82 1"
              specular="0.9" shininess="0.95" reflectance="0.5" />
    <material name="faucet_dark" rgba="0.30 0.32 0.35 1"
              specular="0.8" shininess="0.9" reflectance="0.4" />
    <material name="faucet_highlight" rgba="0.85 0.87 0.90 1"
              specular="0.95" shininess="0.98" reflectance="0.6" />
"""

    def generate_object_xml(self, placement: TaskPlacement) -> str:
        fpos = placement.faucet_pos
        s = placement.faucet_scale

        return f"""
    <body name="faucet_base" pos="{fpos[0]} {fpos[1]} {fpos[2]}">
      <!-- Base mounting ring -->
      <geom name="faucet_base_ring" type="cylinder" size="{0.025*s} {0.005*s}"
            pos="0 0 0" material="faucet_dark" mass="0.5"
            contype="1" conaffinity="1" />
      <!-- Lower column (wider) -->
      <geom name="faucet_col_lo" type="cylinder" size="{0.018*s} {0.025*s}"
            pos="0 0 {0.030*s}" material="faucet_chrome" mass="0.3"
            contype="1" conaffinity="1" />
      <!-- Upper column (narrower, tapered look) -->
      <geom name="faucet_col_hi" type="cylinder" size="{0.014*s} {0.030*s}"
            pos="0 0 {0.085*s}" material="faucet_chrome" mass="0.3"
            contype="1" conaffinity="1" />
      <!-- Neck dome (transition sphere) -->
      <geom name="faucet_neck" type="sphere" size="{0.016*s}"
            pos="0 0 {0.115*s}" material="faucet_chrome" mass="0.1"
            contype="1" conaffinity="1" />
      <!-- Gooseneck spout -->
      <geom name="faucet_spout1" type="capsule" size="{0.008*s}"
            fromto="0 0 {0.115*s}  {0.020*s} 0 {0.125*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout2" type="capsule" size="{0.008*s}"
            fromto="{0.020*s} 0 {0.125*s}  {0.045*s} 0 {0.128*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout3" type="capsule" size="{0.007*s}"
            fromto="{0.045*s} 0 {0.128*s}  {0.065*s} 0 {0.120*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout4" type="capsule" size="{0.006*s}"
            fromto="{0.065*s} 0 {0.120*s}  {0.072*s} 0 {0.105*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <!-- Aerator tip -->
      <geom name="faucet_tip" type="cylinder" size="{0.007*s} {0.003*s}"
            pos="{0.072*s} 0 {0.102*s}" material="faucet_dark"
            contype="0" conaffinity="0" />
      <!-- Handle lever (articulated) -->
      <body name="faucet_switch" pos="0 0 {0.120*s}">
        <joint name="faucet_joint" type="hinge" axis="0 0 1"
               range="-1.5708 1.5708" damping="0.15" frictionloss="0.03" />
        <!-- Handle base hub -->
        <geom name="faucet_hub" type="cylinder" size="{0.012*s} {0.008*s}"
              material="faucet_dark" mass="0.05"
              contype="1" conaffinity="1" />
        <!-- Lever arm -->
        <geom name="faucet_lever" type="capsule" size="{0.006*s}"
              fromto="0 0 {0.005*s}  0 {-0.060*s} {0.015*s}"
              material="faucet_highlight" mass="0.03"
              contype="1" conaffinity="1" friction="2.0 0.1 0.01" />
        <!-- Grip ball at lever tip -->
        <geom name="faucet_grip" type="sphere" size="{0.009*s}"
              pos="0 {-0.065*s} {0.016*s}" material="faucet_highlight" mass="0.02"
              contype="1" conaffinity="1" friction="2.0 0.1 0.01" />
        <!-- Target site at lever tip -->
        <site name="faucet_handle_site" pos="0 {-0.065*s} {0.016*s}"
              size="0.004" rgba="0 1 0 0.3" />
      </body>
    </body>
"""

    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        return {
            "faucet_joint": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "faucet_joint"),
            "faucet_handle_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "faucet_handle_site"),
            "faucet_switch_body": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "faucet_switch"),
        }

    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        joint_id = ids["faucet_joint"]
        if joint_id < 0:
            return False
        qpos_addr = model.jnt_qposadr[joint_id]
        current_angle = float(data.qpos[qpos_addr])
        target = placement.faucet_target_angle
        # Target is negative, success when angle <= 80% of target
        return current_angle <= target * 0.8
