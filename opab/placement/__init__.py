"""
Robot x Task placement configuration matrix.

Contains all position/size/threshold values that vary between
robot-task combinations (35 entries: 5 robots x 7 tasks).
"""
from __future__ import annotations

import numpy as np

from opab.tasks.base_task import TaskPlacement


def get_placement(robot_name: str, task_name: str) -> TaskPlacement:
    """
    Get the TaskPlacement for a (robot, task) pair.

    This centralizes all the per-robot, per-task numeric tuning
    (positions, sizes, scales, thresholds) that was previously
    scattered inside RobotConfig.
    """
    _bases = _ROBOT_BASES[robot_name]

    placement = TaskPlacement(
        table_pos=_bases["table_pos"].copy(),
        table_half_size=_bases["table_half_size"].copy(),
        cam_pos=_bases["cam_pos"].copy(),
        success_threshold=_bases["success_threshold"],
        cube_pos=_bases["cube_pos"].copy(),
        cube_size=_bases["cube_size"],
        cube_mass=_bases["cube_mass"],
        target_pos=_bases["target_pos"].copy(),
        cube_randomize_range=_bases["cube_randomize_range"],
        peg_radius=_bases["peg_radius"],
        peg_half_length=_bases["peg_half_length"],
        hole_clearance=_bases["hole_clearance"],
        hole_depth=_bases["hole_depth"],
        drawer_pos=_bases["drawer_pos"].copy(),
        drawer_size=_bases["drawer_size"].copy(),
        drawer_slide_range=_bases["drawer_slide_range"],
        faucet_pos=_bases["faucet_pos"].copy(),
        faucet_scale=_bases["faucet_scale"],
        faucet_target_angle=_bases["faucet_target_angle"],
        door_pos=_bases["door_pos"].copy(),
        door_scale=_bases["door_scale"],
    )

    return placement


# ============================================================
# Per-robot base configurations
# ============================================================

_ROBOT_BASES: dict[str, dict] = {
    "franka": {
        "table_pos": np.array([0.35, 0.0, 0.2]),
        "table_half_size": np.array([0.40, 0.60, 0.2]),
        "cam_pos": np.array([0.25, 0.0, 1.3]),
        "success_threshold": 0.025,
        "cube_pos": np.array([0.45, 0.05, 0.42]),
        "cube_size": 0.02,
        "cube_mass": 0.05,
        "target_pos": np.array([0.45, -0.05, 0.405]),
        "cube_randomize_range": 0.0,
        "peg_radius": 0.010,
        "peg_half_length": 0.050,
        "hole_clearance": 0.002,
        "hole_depth": 0.045,
        "drawer_pos": np.array([0.45, 0.10, 0.42]),
        "drawer_size": np.array([0.05, 0.04, 0.03]),
        "drawer_slide_range": 0.08,
        "faucet_pos": np.array([0.45, -0.10, 0.42]),
        "faucet_scale": 1.0,
        "faucet_target_angle": -1.2,
        "door_pos": np.array([0.50, 0.15, 0.42]),
        "door_scale": 1.0,
    },
    "ur5": {
        "table_pos": np.array([0.35, 0.0, 0.2]),
        "table_half_size": np.array([0.40, 0.60, 0.2]),
        "cam_pos": np.array([0.30, 0.0, 1.3]),
        "success_threshold": 0.025,
        "cube_pos": np.array([0.45, 0.15, 0.42]),
        "cube_size": 0.02,
        "cube_mass": 0.05,
        "target_pos": np.array([0.45, -0.15, 0.405]),
        "cube_randomize_range": 0.0,
        "peg_radius": 0.010,
        "peg_half_length": 0.050,
        "hole_clearance": 0.002,
        "hole_depth": 0.045,
        "drawer_pos": np.array([0.40, 0.10, 0.42]),
        "drawer_size": np.array([0.05, 0.04, 0.03]),
        "drawer_slide_range": 0.08,
        "faucet_pos": np.array([0.40, -0.15, 0.42]),
        "faucet_scale": 1.0,
        "faucet_target_angle": -1.2,
        "door_pos": np.array([0.45, 0.15, 0.42]),
        "door_scale": 1.0,
    },
    "widowx": {
        "table_pos": np.array([0.18, 0.0, 0.2]),
        "table_half_size": np.array([0.38, 0.61, 0.2]),
        "cam_pos": np.array([0.15, 0.0, 1.2]),
        "success_threshold": 0.0225,
        "cube_pos": np.array([0.22, 0.12, 0.418]),
        "cube_size": 0.018,
        "cube_mass": 0.03,
        "target_pos": np.array([0.22, -0.12, 0.405]),
        "cube_randomize_range": 0.0,
        "peg_radius": 0.008,
        "peg_half_length": 0.030,
        "hole_clearance": 0.002,
        "hole_depth": 0.040,
        "drawer_pos": np.array([0.28, 0.06, 0.42]),
        "drawer_size": np.array([0.04, 0.035, 0.025]),
        "drawer_slide_range": 0.06,
        "faucet_pos": np.array([0.22, -0.06, 0.42]),
        "faucet_scale": 0.9,
        "faucet_target_angle": -1.2,
        "door_pos": np.array([0.28, 0.08, 0.065]),
        "door_scale": 0.75,
    },
    "lite6": {
        "table_pos": np.array([0.25, 0.0, 0.2]),
        "table_half_size": np.array([0.30, 0.60, 0.2]),
        "cam_pos": np.array([0.20, 0.0, 1.2]),
        "success_threshold": 0.0225,
        "cube_pos": np.array([0.30, 0.10, 0.418]),
        "cube_size": 0.018,
        "cube_mass": 0.04,
        "target_pos": np.array([0.30, -0.10, 0.405]),
        "cube_randomize_range": 0.0,
        "peg_radius": 0.008,
        "peg_half_length": 0.030,
        "hole_clearance": 0.002,
        "hole_depth": 0.040,
        "drawer_pos": np.array([0.20, 0.06, 0.42]),
        "drawer_size": np.array([0.035, 0.03, 0.02]),
        "drawer_slide_range": 0.05,
        "faucet_pos": np.array([0.30, -0.08, 0.42]),
        "faucet_scale": 1.0,
        "faucet_target_angle": -1.2,
        "door_pos": np.array([0.35, 0.12, 0.41]),
        "door_scale": 0.85,
    },
    "so101": {
        "table_pos": np.array([0.25, 0.0, 0.1]),
        "table_half_size": np.array([0.30, 0.50, 0.1]),
        "cam_pos": np.array([0.10, 0.0, 0.7]),
        "success_threshold": 0.01875,
        "cube_pos": np.array([0.15, 0.06, 0.213]),
        "cube_size": 0.012,
        "cube_mass": 0.03,
        "target_pos": np.array([0.15, -0.06, 0.205]),
        "cube_randomize_range": 0.0,
        "peg_radius": 0.006,
        "peg_half_length": 0.050,
        "hole_clearance": 0.002,
        "hole_depth": 0.020,
        "drawer_pos": np.array([0.23, 0.06, 0.42]),
        "drawer_size": np.array([0.035, 0.03, 0.02]),
        "drawer_slide_range": 0.05,
        "faucet_pos": np.array([0.18, -0.06, 0.22]),
        "faucet_scale": 0.7,
        "faucet_target_angle": -1.2,
        "door_pos": np.array([0.22, 0.08, 0.021]),
        "door_scale": 0.60,
    },
}
