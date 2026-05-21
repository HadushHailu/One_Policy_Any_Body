"""UFatory Lite6 robot configuration."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from opab.robots.base_robot import BaseRobot


class Lite6Robot(BaseRobot):
    """UFatory Lite6 6-DOF robot with narrow parallel gripper."""

    def __init__(self, **kwargs):
        self.name = "lite6"
        self.scene_path = kwargs.get(
            "scene_path", "assets/mujoco_menagerie/ufactory_lite6/scene.xml"
        )
        self.ee_site_name = "end_effector"
        self.ee_body_name = "gripper_body"
        self.n_arm_joints = 6
        self.n_gripper_joints = 1
        self.arm_joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6",
        ]
        self.gripper_joint_names = ["gripper_left_finger"]
        self.gripper_actuator_idx = 6
        self.n_arm_actuators = None  # same as n_arm_joints
        self.action_scale = 0.01
        self.ik_damping = 0.01
        self.ik_max_iter = 50
        self.gripper_open = -0.025
        self.gripper_closed = -0.00001
        self.home_qpos = {
            "joint1": 0.0, "joint2": 0.0, "joint3": 1.57,
            "joint4": 0.0, "joint5": 1.57, "joint6": 0.0,
            "gripper_left_finger": 0.0, "gripper_right_finger": 0.0,
        }
        self.robot_z_offset = 0.4

    def pre_inline_xml(self, xml: str, scene_dir: Path) -> str:
        """Swap lite6.xml include for the gripper-equipped version (before inlining)."""
        xml = xml.replace(
            '<include file="lite6.xml"/>',
            '<include file="lite6_gripper_narrow.xml"/>'
        )
        return xml

    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """Lite6: motor actuator, ctrl=gripper_open -> open, ctrl=gripper_closed -> closed."""
        grip_val = self.gripper_open + cmd * (
            self.gripper_closed - self.gripper_open
        )
        data.ctrl[self.gripper_actuator_idx] = grip_val

    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Read gripper_left_finger slide joint, normalize to [0=open, 1=closed]."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_finger")
        if jid >= 0:
            raw = data.qpos[model.jnt_qposadr[jid]]
            # Range: -0.025 (open/apart) to ~0 (closed/together)
            return (raw + 0.025) / 0.025
        return 0.0
