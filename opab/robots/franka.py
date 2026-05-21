"""Franka Emika Panda robot configuration."""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from opab.robots.base_robot import BaseRobot


class FrankaRobot(BaseRobot):
    """Franka Emika Panda 7-DOF robot with parallel-jaw gripper."""

    def __init__(self, **kwargs):
        self.name = "franka"
        self.scene_path = kwargs.get(
            "scene_path", "assets/mujoco_menagerie/franka_emika_panda/scene.xml"
        )
        self.ee_body_name = "hand"
        self.ee_site_name = "grip_site"
        self.n_arm_joints = 7
        self.n_gripper_joints = 2
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.gripper_joint_names = ["finger_joint1", "finger_joint2"]
        self.gripper_actuator_idx = [8, 9]  # injected finger actuators
        self.n_arm_actuators = 8  # ctrl[0..6] = arm position, ctrl[7] = arm velocity
        self.action_scale = 0.01
        self.ik_damping = 0.01
        self.ik_max_iter = 50
        self.gripper_open = 0.04
        self.gripper_closed = 0.0
        self.home_qpos = {
            "joint1": 0.0, "joint2": -0.785, "joint3": 0.0,
            "joint4": -2.356, "joint5": 0.0, "joint6": 1.571,
            "joint7": 0.785,
            "finger_joint1": 0.04, "finger_joint2": 0.04,
        }
        self.robot_z_offset = 0.4

    def modify_xml(self, xml: str, scene_dir: Path) -> str:
        """Inject grip_site at fingertip center and finger position actuators."""
        # 1. Inject grip_site inside the hand body
        grip_site_xml = (
            '\n        <site name="grip_site" pos="0 0 0.105" size="0.005" '
            'rgba="1 0 0 0.5" type="sphere"/>'
        )
        hand_body_tag = '<body name="hand"'
        idx = xml.find(hand_body_tag)
        if idx >= 0:
            # Find first geom inside hand body
            geom_idx = xml.find("<geom", idx)
            if geom_idx > idx:
                xml = xml[:geom_idx] + grip_site_xml + "\n        " + xml[geom_idx:]

        # 2. Inject finger position actuators
        finger_actuators = """
  <actuator>
    <position name="finger_joint1_act" joint="finger_joint1" kp="800"
              ctrlrange="0 0.04" forcerange="-20 20"/>
    <position name="finger_joint2_act" joint="finger_joint2" kp="800"
              ctrlrange="0 0.04" forcerange="-20 20"/>
  </actuator>
"""
        mujoco_close = xml.rfind("</mujoco>")
        if mujoco_close >= 0:
            xml = xml[:mujoco_close] + finger_actuators + "</mujoco>"

        return xml

    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """Franka: position-controlled parallel fingers."""
        grip_pos = self.gripper_open * (1.0 - cmd)
        for idx in self.gripper_actuator_idx:
            if idx < model.nu:
                data.ctrl[idx] = grip_pos

    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Read finger_joint1 and normalize to [0=open, 1=closed]."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        if jid >= 0:
            raw = data.qpos[model.jnt_qposadr[jid]]
            return 1.0 - (raw / self.gripper_open)
        return 0.0
