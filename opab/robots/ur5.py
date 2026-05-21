"""UR5e robot with Robotiq 2F-85 gripper."""
from __future__ import annotations

import os
import re
from pathlib import Path

import mujoco
import numpy as np

from opab.robots.base_robot import BaseRobot


class UR5Robot(BaseRobot):
    """Universal Robots UR5e 6-DOF with Robotiq 2F-85 parallel gripper."""

    def __init__(self, **kwargs):
        self.name = "ur5"
        self.scene_path = kwargs.get(
            "scene_path", "assets/mujoco_menagerie/universal_robots_ur5e/scene.xml"
        )
        self.ee_body_name = "robotiq_base"
        self.ee_site_name = "robotiq_pinch"
        self.n_arm_joints = 6
        self.n_gripper_joints = 2
        self.arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.gripper_joint_names = [
            "robotiq_right_driver_joint", "robotiq_left_driver_joint",
        ]
        self.gripper_actuator_idx = 6
        self.n_arm_actuators = None  # same as n_arm_joints
        self.action_scale = 0.01
        self.ik_damping = 0.01
        self.ik_max_iter = 50
        self.gripper_open = 0.0
        self.gripper_closed = 255.0
        self.home_qpos = {
            "shoulder_pan_joint": -3.14,
            "shoulder_lift_joint": -1.45,
            "elbow_joint": 0.9,
            "wrist_1_joint": -1.0208,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0,
        }
        self.robot_z_offset = 0.4

    def modify_xml(self, xml: str, scene_dir: Path) -> str:
        """Inject Robotiq 2F-85 gripper into UR5e wrist_3_link."""
        return self._inject_robotiq_gripper(xml, scene_dir)

    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """UR5 Robotiq: ctrl 0=open, 255=closed."""
        grip_val = cmd * self.gripper_closed
        data.ctrl[self.gripper_actuator_idx] = grip_val

    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Read right_driver_joint [0=open, 0.8=closed] normalized to [0,1]."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robotiq_right_driver_joint")
        if jid >= 0:
            raw = data.qpos[model.jnt_qposadr[jid]]
            return raw / 0.8
        return 0.0

    def _inject_robotiq_gripper(self, xml: str, scene_dir: Path) -> str:
        """Inject Robotiq 2F-85 gripper body into UR5e wrist_3_link."""
        project_root = Path(__file__).resolve().parents[2]
        robotiq_mesh_dir = project_root / "assets" / "mujoco_menagerie" / "robotiq_2f85" / "assets"

        ur5_mesh_base = scene_dir / "assets"
        rel_mesh = os.path.relpath(robotiq_mesh_dir, ur5_mesh_base)

        # 1. Inject Robotiq assets
        robotiq_assets = f"""
    <!-- Robotiq 2F-85 assets -->
    <material name="robotiq_metal" rgba="0.58 0.58 0.58 1"/>
    <material name="robotiq_silicone" rgba="0.1882 0.1882 0.1882 1"/>
    <material name="robotiq_gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="robotiq_black" rgba="0.149 0.149 0.149 1"/>
    <mesh name="robotiq_base_mount" file="{rel_mesh}/base_mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_base" file="{rel_mesh}/base.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_driver" file="{rel_mesh}/driver.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_coupler" file="{rel_mesh}/coupler.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_follower" file="{rel_mesh}/follower.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_pad" file="{rel_mesh}/pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_silicone_pad" file="{rel_mesh}/silicone_pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_spring_link" file="{rel_mesh}/spring_link.stl" scale="0.001 0.001 0.001"/>
"""
        asset_close_idx = xml.rfind("</asset>")
        if asset_close_idx >= 0:
            xml = xml[:asset_close_idx] + robotiq_assets + "  </asset>" + xml[asset_close_idx + len("</asset>"):]
        else:
            worldbody_idx = xml.find("<worldbody")
            xml = xml[:worldbody_idx] + f"<asset>{robotiq_assets}</asset>\n" + xml[worldbody_idx:]

        # 2. Inject gripper body under wrist_3_link
        robotiq_body = """
                  <!-- Robotiq 2F-85 gripper mounted at attachment_site -->
                  <body name="robotiq_base_mount" pos="0 0.107 0" quat="-0.707107 0.707107 0 0">
                    <geom type="mesh" mesh="robotiq_base_mount" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                    <geom type="mesh" mesh="robotiq_base_mount" group="3"/>
                    <body name="robotiq_base" pos="0 0 0.0038" quat="1 0 0 -1">
                      <inertial mass="0.777441" pos="0 -2.70394e-05 0.0354675" quat="1 -0.00152849 0 0"
                          diaginertia="0.000260285 0.000225381 0.000152708"/>
                      <geom type="mesh" mesh="robotiq_base" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                      <geom type="mesh" mesh="robotiq_base" group="3"/>
                      <site name="robotiq_pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>
                      <!-- Right-hand side 4-bar linkage -->
                      <body name="robotiq_right_driver" pos="0 0.0306011 0.054904">
                        <inertial mass="0.00899563" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                            diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>
                        <joint name="robotiq_right_driver_joint" axis="1 0 0" range="0 0.8" armature="0.005" damping="0.1"
                            solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                        <geom type="mesh" mesh="robotiq_driver" material="robotiq_gray" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_driver" group="3"/>
                        <body name="robotiq_right_coupler" pos="0 0.0315 -0.0041">
                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                              diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>
                          <joint name="robotiq_right_coupler_joint" axis="1 0 0" range="-1.57 0" armature="0.001"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_coupler" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_coupler" group="3"/>
                        </body>
                      </body>
                      <body name="robotiq_right_spring_link" pos="0 0.0132 0.0609">
                        <inertial mass="0.0221642" pos="0 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                            diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>
                        <joint name="robotiq_right_spring_link_joint" axis="1 0 0" range="-0.29670597283 0.8"
                            armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
                        <geom type="mesh" mesh="robotiq_spring_link" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_spring_link" group="3"/>
                        <body name="robotiq_right_follower" pos="0 0.055 0.0375">
                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                              diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>
                          <joint name="robotiq_right_follower_joint" axis="1 0 0" range="-0.872664 0.872664"
                              armature="0.001" pos="0 -0.018 0.0065"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_follower" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_follower" group="3"/>
                          <body name="robotiq_right_pad" pos="0 -0.0189 0.01352">
                            <geom name="robotiq_right_pad1" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375"
                                mass="0" friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1" group="3"/>
                            <geom name="robotiq_right_pad2" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375"
                                mass="0" friction="0.6" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1" group="3"/>
                            <inertial mass="0.0035" pos="0 -0.0025 0.0185" quat="0.707107 0 0 0.707107"
                                diaginertia="4.73958e-07 3.64583e-07 1.23958e-07"/>
                            <geom type="mesh" mesh="robotiq_pad" contype="0" conaffinity="0" group="2"/>
                            <geom type="mesh" mesh="robotiq_silicone_pad" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          </body>
                        </body>
                      </body>
                      <!-- Left-hand side 4-bar linkage -->
                      <body name="robotiq_left_driver" pos="0 -0.0306011 0.054904" quat="0 0 0 1">
                        <inertial mass="0.00899563" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                            diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>
                        <joint name="robotiq_left_driver_joint" axis="1 0 0" range="0 0.8" armature="0.005" damping="0.1"
                            solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                        <geom type="mesh" mesh="robotiq_driver" material="robotiq_gray" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_driver" group="3"/>
                        <body name="robotiq_left_coupler" pos="0 0.0315 -0.0041">
                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                              diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>
                          <joint name="robotiq_left_coupler_joint" axis="1 0 0" range="-1.57 0" armature="0.001"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_coupler" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_coupler" group="3"/>
                        </body>
                      </body>
                      <body name="robotiq_left_spring_link" pos="0 -0.0132 0.0609" quat="0 0 0 1">
                        <inertial mass="0.0221642" pos="0 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                            diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>
                        <joint name="robotiq_left_spring_link_joint" axis="1 0 0" range="-0.29670597283 0.8"
                            armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
                        <geom type="mesh" mesh="robotiq_spring_link" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_spring_link" group="3"/>
                        <body name="robotiq_left_follower" pos="0 0.055 0.0375">
                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                              diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>
                          <joint name="robotiq_left_follower_joint" axis="1 0 0" range="-0.872664 0.872664"
                              armature="0.001" pos="0 -0.018 0.0065"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_follower" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_follower" group="3"/>
                          <body name="robotiq_left_pad" pos="0 -0.0189 0.01352">
                            <geom name="robotiq_left_pad1" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375"
                                mass="0" friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1" group="3"/>
                            <geom name="robotiq_left_pad2" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375"
                                mass="0" friction="0.6" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1" group="3"/>
                            <inertial mass="0.0035" pos="0 -0.0025 0.0185" quat="1 0 0 1"
                                diaginertia="4.73958e-07 3.64583e-07 1.23958e-07"/>
                            <geom type="mesh" mesh="robotiq_pad" contype="0" conaffinity="0" group="2"/>
                            <geom type="mesh" mesh="robotiq_silicone_pad" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
"""
        # Insert after attachment_site
        xml = xml.replace(
            '<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>',
            '<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>'
            + robotiq_body
        )

        # 3. Contact exclusions
        robotiq_contacts = """
  <contact>
    <exclude body1="robotiq_base" body2="robotiq_left_driver"/>
    <exclude body1="robotiq_base" body2="robotiq_right_driver"/>
    <exclude body1="robotiq_base" body2="robotiq_left_spring_link"/>
    <exclude body1="robotiq_base" body2="robotiq_right_spring_link"/>
    <exclude body1="robotiq_right_coupler" body2="robotiq_right_follower"/>
    <exclude body1="robotiq_left_coupler" body2="robotiq_left_follower"/>
  </contact>
"""
        # 4. Tendon
        robotiq_tendon = """
  <tendon>
    <fixed name="robotiq_split">
      <joint joint="robotiq_right_driver_joint" coef="0.5"/>
      <joint joint="robotiq_left_driver_joint" coef="0.5"/>
    </fixed>
  </tendon>
"""
        # 5. Equality constraints
        robotiq_equality = """
  <equality>
    <connect anchor="0 0 0" body1="robotiq_right_follower" body2="robotiq_right_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <connect anchor="0 0 0" body1="robotiq_left_follower" body2="robotiq_left_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="robotiq_right_driver_joint" joint2="robotiq_left_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="robotiq_right_spring_link_joint" joint2="robotiq_right_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="robotiq_left_spring_link_joint" joint2="robotiq_left_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>
"""
        # 6. Actuator
        robotiq_actuator = """
  <actuator>
    <general name="robotiq_fingers_actuator" tendon="robotiq_split" forcerange="-5 5" ctrlrange="0 255"
        gaintype="fixed" gainprm="0.3137255 0 0" biastype="affine" biasprm="0 -100 -10"/>
  </actuator>
"""
        mujoco_close_idx = xml.rfind("</mujoco>")
        xml = (xml[:mujoco_close_idx]
               + robotiq_contacts + robotiq_tendon + robotiq_equality + robotiq_actuator
               + "</mujoco>")

        return xml
