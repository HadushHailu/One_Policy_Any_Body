"""
OPAB environment package.

Provides unified manipulation environments for multiple robot embodiments.

Architecture (v2 — modular):
  - opab.robots/  — per-robot config + gripper control + XML patches
  - opab.tasks/   — per-task object XML + success checks
  - opab.placement/ — robot×task placement matrix
  - opab.env.ik_solver  — standalone DLS IK
  - opab.env.modular_env — thin orchestrator (ManipulationEnv)

The original monolithic base_env.py is preserved for reference.
All public imports route through the modular implementation.
"""
from opab.env.modular_env import ManipulationEnv, ManipulationEnv as PickPlaceEnv
from opab.env.base_env import RobotConfig  # backwards compat for scripts reading RobotConfig
from opab.env.scripted_policies import (
    ScriptedPickPlace, ScriptedStack, ScriptedReach, ScriptedPush, ScriptedPegInsertion
)
from opab.env.domain_randomization import DomainRandomizer, DRConfig


SUPPORTED_ROBOTS = ("franka", "ur5", "widowx", "lite6", "so101")
SUPPORTED_TASKS = (
    "pick_place", "push", "stack", "peg_insertion",
    "drawer_open", "turn_faucet", "door_open",
)

# Task name → integer ID mapping (used in HDF5 and by TaskEncoder)
TASK_ID_MAP = {
    "pick_place": 0,
    "push": 1,
    "stack": 2,
    "peg_insertion": 3,
    "drawer_open": 4,
    "turn_faucet": 5,
    "door_open": 6,
}


def make_env(
    robot: str = "franka",
    image_size: tuple[int, int] = (128, 128),
    control_freq: float = 20.0,
    max_episode_steps: int = 300,
    seed: int | None = None,
    domain_randomization: bool = False,
    task: str = "pick_place",
) -> ManipulationEnv:
    """
    Factory function — creates a ManipulationEnv for the specified robot and task.

    Args:
        robot: One of 'franka', 'ur5', 'widowx', 'lite6', 'so101'
        image_size: Camera resolution (H, W)
        control_freq: Control loop frequency (Hz)
        max_episode_steps: Episode length limit
        seed: RNG seed for reproducibility
        domain_randomization: Enable DR
        task: One of the SUPPORTED_TASKS

    Returns:
        ManipulationEnv instance ready for reset()/step()
    """
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(
            f"Unknown robot '{robot}'. Supported: {SUPPORTED_ROBOTS}"
        )
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Supported: {SUPPORTED_TASKS}"
        )

    env = ManipulationEnv(
        robot=robot,
        image_size=image_size,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        seed=seed,
        task=task,
    )

    # Attach domain randomizer (inactive unless enabled)
    dr_config = DRConfig(enabled=domain_randomization)
    env.domain_randomizer = DomainRandomizer(env.model, dr_config)

    return env


__all__ = [
    "make_env",
    "ManipulationEnv",
    "PickPlaceEnv",
    "RobotConfig",
    "ScriptedPickPlace",
    "ScriptedStack",
    "ScriptedReach",
    "ScriptedPush",
    "ScriptedPegInsertion",
    "DomainRandomizer",
    "DRConfig",
    "SUPPORTED_ROBOTS",
    "SUPPORTED_TASKS",
    "TASK_ID_MAP",
]
