"""Trossen WidowX 250s robot configuration."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from opab.robots.base_robot import BaseRobot


class WidowXRobot(BaseRobot):
    """Trossen WidowX 250s 6-DOF robot with slide-joint gripper."""

    def __init__(self, **kwargs):
        self.name = "widowx"
        self.scene_path = kwargs.get(
            "scene_path", "assets/mujoco_menagerie/trossen_wx250s/scene.xml"
        )
        self.ee_site_name = "grip_site"
        self.ee_body_name = "wx250s/gripper_link"
        self.n_arm_joints = 6
        self.n_gripper_joints = 1
        self.arm_joint_names = [
            "waist", "shoulder", "elbow",
            "forearm_roll", "wrist_angle", "wrist_rotate",
        ]
        self.gripper_joint_names = ["left_finger"]
        self.gripper_actuator_idx = 6
        self.n_arm_actuators = None
        self.action_scale = 0.01
        self.ik_damping = 0.02
        self.ik_max_iter = 50
        self.gripper_open = 0.037
        self.gripper_closed = 0.015
        self.home_qpos = {
            "waist": 0.0, "shoulder": -0.96, "elbow": 1.16,
            "forearm_roll": 0.0, "wrist_angle": -0.3, "wrist_rotate": 0.0,
            "left_finger": 0.037, "right_finger": -0.037,
        }
        self.robot_z_offset = 0.4

    def modify_xml(self, xml: str, scene_dir: Path) -> str:
        """Inject grip_site at fingertip center for WidowX."""
        import re
        # Inject grip_site inside the gripper_link body
        grip_site = (
            '\n            <site name="grip_site" pos="0 0 0.042" '
            'size="0.005" rgba="1 0 0 0.5" type="sphere"/>'
        )
        # Find the gripper_link body and inject site before its first child body
        pattern = r'(<body name="[^"]*gripper_link"[^>]*>)'
        match = re.search(pattern, xml)
        if match:
            insert_pos = match.end()
            xml = xml[:insert_pos] + grip_site + xml[insert_pos:]
        return xml

    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """WidowX: position actuator on left_finger slide joint."""
        grip_pos = self.gripper_open + cmd * (
            self.gripper_closed - self.gripper_open
        )
        data.ctrl[self.gripper_actuator_idx] = grip_pos

    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Read left_finger slide joint, normalize to [0=open, 1=closed]."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
        if jid >= 0:
            raw = data.qpos[model.jnt_qposadr[jid]]
            span = self.gripper_closed - self.gripper_open
            if abs(span) > 1e-6:
                return (raw - self.gripper_open) / span
        return 0.0

    def post_load_tuning(self, model: mujoco.MjModel) -> None:
        """Boost arm force limits for position tracking."""
        for i in range(self.gripper_actuator_idx):
            model.actuator_forcerange[i] = [-50.0, 50.0]
        model.actuator_forcerange[self.gripper_actuator_idx] = [-5.0, 5.0]
