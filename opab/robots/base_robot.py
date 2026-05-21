"""Abstract base class for robot embodiments."""
from __future__ import annotations

import abc
import re
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


class BaseRobot(abc.ABC):
    """
    Abstract base class encapsulating robot-specific configuration and behavior.

    Each robot subclass provides:
      - Scene path and joint configuration
      - Gripper control mapping (set/get)
      - XML modifications needed for the robot
      - IK parameters
      - Home pose
    """

    # --- Required class-level attributes (set in subclass __init__) ---
    name: str
    scene_path: str
    ee_body_name: str
    ee_site_name: str
    n_arm_joints: int
    n_gripper_joints: int
    arm_joint_names: list[str]
    gripper_joint_names: list[str]
    gripper_actuator_idx: int | list[int]
    action_scale: float
    ik_damping: float
    ik_max_iter: int
    gripper_open: float
    gripper_closed: float
    home_qpos: dict[str, float]
    robot_z_offset: float

    # Optional
    n_arm_actuators: Optional[int] = None  # defaults to n_arm_joints

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Initialize robot configuration."""
        ...

    def modify_xml(self, xml: str, scene_dir: Path) -> str:
        """
        Apply robot-specific XML modifications before model compilation.

        Override in subclass for robot-specific patches (e.g., injecting grippers,
        grip sites, swapping includes).

        Args:
            xml: The inlined scene XML string
            scene_dir: Directory containing the scene file (for resolving paths)

        Returns:
            Modified XML string
        """
        return xml

    def pre_inline_xml(self, xml: str, scene_dir: Path) -> str:
        """
        Apply XML modifications BEFORE includes are inlined.

        Use for include file swaps (e.g., Lite6 gripper include).
        Override in subclass if needed.

        Args:
            xml: Raw scene XML with <include> tags still present
            scene_dir: Scene directory

        Returns:
            Modified XML string
        """
        return xml

    @abc.abstractmethod
    def set_gripper(self, model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
        """
        Set gripper position command.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            cmd: Normalized command [0=open, 1=closed]
        """
        ...

    @abc.abstractmethod
    def get_gripper_pos(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """
        Get normalized gripper position [0=open, 1=closed].

        Args:
            model: MuJoCo model
            data: MuJoCo data

        Returns:
            Normalized position in [0, 1]
        """
        ...

    def post_load_tuning(self, model: mujoco.MjModel) -> None:
        """
        Apply post-load actuator/physics tuning (e.g., force limits).

        Override in subclass if needed.
        """
        pass

    @property
    def effective_n_arm_actuators(self) -> int:
        """Number of arm actuator ctrl slots (may differ from n_arm_joints)."""
        return self.n_arm_actuators if self.n_arm_actuators is not None else self.n_arm_joints
