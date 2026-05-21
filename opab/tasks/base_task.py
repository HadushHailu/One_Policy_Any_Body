"""Abstract base class for manipulation tasks."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


@dataclass
class TaskPlacement:
    """Per-(robot, task) placement configuration.

    Contains all positions, sizes, and thresholds that vary
    between robot-task combinations.
    """
    # Table
    table_pos: np.ndarray
    table_half_size: np.ndarray
    # Camera
    cam_pos: np.ndarray
    # Success
    success_threshold: float = 0.025
    # Cube (shared by pick_place, push, stack)
    cube_pos: Optional[np.ndarray] = None
    cube_size: float = 0.02
    cube_mass: float = 0.05
    target_pos: Optional[np.ndarray] = None
    cube_randomize_range: float = 0.0
    # Peg insertion
    peg_radius: float = 0.010
    peg_half_length: float = 0.050
    hole_clearance: float = 0.002
    hole_depth: float = 0.045
    # Drawer
    drawer_pos: Optional[np.ndarray] = None
    drawer_size: Optional[np.ndarray] = None
    drawer_slide_range: float = 0.08
    # Faucet
    faucet_pos: Optional[np.ndarray] = None
    faucet_scale: float = 1.0
    faucet_target_angle: float = -1.2
    # Door
    door_pos: Optional[np.ndarray] = None
    door_scale: float = 1.0


class BaseTask(abc.ABC):
    """
    Abstract base class for manipulation tasks.

    Each task provides:
      - Object XML to inject into the scene
      - Success condition checking
      - ID caching for fast runtime access
      - Reset randomization for task objects
    """

    name: str  # Set in subclass

    @abc.abstractmethod
    def generate_object_xml(self, placement: TaskPlacement) -> str:
        """
        Generate MuJoCo XML string for task-specific objects.

        Args:
            placement: TaskPlacement with positions/sizes for this robot-task pair

        Returns:
            XML string to inject before </worldbody>
        """
        ...

    def generate_asset_xml(self, placement: TaskPlacement) -> str:
        """
        Generate additional asset XML (materials, meshes) if needed.

        Override in subclass if task needs custom assets.

        Returns:
            XML string to inject before </asset>, or empty string
        """
        return ""

    @abc.abstractmethod
    def cache_ids(self, model: mujoco.MjModel) -> dict[str, int]:
        """
        Cache MuJoCo object IDs for fast runtime access.

        Args:
            model: Compiled MuJoCo model

        Returns:
            Dict of name -> ID mappings for this task's objects
        """
        ...

    @abc.abstractmethod
    def check_success(self, model: mujoco.MjModel, data: mujoco.MjData,
                      ids: dict[str, int], placement: TaskPlacement) -> bool:
        """
        Check task-specific success condition.

        Args:
            model: MuJoCo model
            data: MuJoCo data (current state)
            ids: Cached IDs from cache_ids()
            placement: TaskPlacement config

        Returns:
            True if task is complete
        """
        ...

    def randomize_reset(self, model: mujoco.MjModel, data: mujoco.MjData,
                        ids: dict[str, int], placement: TaskPlacement,
                        rng: np.random.Generator) -> None:
        """
        Randomize task object positions on reset.

        Override in subclass for task-specific randomization.
        Default: no randomization.
        """
        pass

    def get_helper_methods(self) -> dict[str, callable]:
        """
        Return dict of helper methods exposed to scripted policies.

        These are methods like get_cube_pos(), get_door_angle() etc.
        The env will bind them as instance methods.

        Returns:
            Dict of method_name -> function(model, data, ids) -> value
        """
        return {}
