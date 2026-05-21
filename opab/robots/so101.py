"""RobotStudio SO-101 robot configuration."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from opab.robots.base_robot import BaseRobot


class SO101Robot(BaseRobot):
    """RobotStudio SO-101 5-DOF robot with hinge gripper."""

    def __init__(self, **kwargs):
        self.name = "so101"
        self.scene_path = kwargs.get(
            "scene_path", "assets/mujoco_menagerie/robotstudio_so101/scene.xml"
        )
        self.ee_site_name = "gripperframe"
        self.ee_body_name = "gripper"
        self.n_arm_joints = 5
        self.n_gripper_joints = 1
        self.arm_joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll",
        ]
        self.gripper_joint_names = ["gripper"]
        self.gripper_actuator_idx = 5
        self.n_arm_actuators = None
        self.action_scale = 0.01
        self.ik_damping = 0.05
        self.ik_max_iter = 30
        self.gripper_open = 1.5
        self.gripper_closed = -0.17
        self.home_qpos = {
            "shoulder_pan": 0.0, "shoulder_lift": 0.0,
            "elbow_flex": 0.0, "wrist_flex": 1.5708,
            "wrist_roll": 0.0, "gripper": 1.5,
        }
        self.robot_z_offset = 0.2

    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """SO-101: hinge gripper (open=positive, closed=negative)."""
        grip_pos = self.gripper_open + cmd * (
            self.gripper_closed - self.gripper_open
        )
        data.ctrl[self.gripper_actuator_idx] = grip_pos

    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Read gripper hinge joint, normalize to [0=open, 1=closed]."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        if jid >= 0:
            raw = data.qpos[model.jnt_qposadr[jid]]
            span = self.gripper_closed - self.gripper_open
            if abs(span) > 1e-6:
                return (raw - self.gripper_open) / span
        return 0.0

    def post_load_tuning(self, model: mujoco.MjModel) -> None:
        """Boost arm force limits (default ±2.94 Nm too low for kp=998)."""
        for i in range(self.gripper_actuator_idx):
            model.actuator_forcerange[i] = [-50.0, 50.0]
        model.actuator_forcerange[self.gripper_actuator_idx] = [-5.0, 5.0]
