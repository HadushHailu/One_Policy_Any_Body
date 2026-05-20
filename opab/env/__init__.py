"""
OPAB environment package.

Provides unified manipulation environments for multiple robot embodiments.
"""
from opab.env.base_env import PickPlaceEnv, RobotConfig
from opab.env.scripted_policies import (
    ScriptedPickPlace, ScriptedStack, ScriptedReach, ScriptedPush, ScriptedPegInsertion
)
from opab.env.domain_randomization import DomainRandomizer, DRConfig


SUPPORTED_ROBOTS = ("franka", "ur5", "widowx", "lite6", "so101")
SUPPORTED_TASKS = ("reach", "pick_place", "push", "stack", "peg_insertion")

# Task name → integer ID mapping (used in HDF5 and by TaskEncoder)
TASK_ID_MAP = {
    "reach": 0,
    "pick_place": 1,
    "push": 2,
    "stack": 3,
    "peg_insertion": 4,
}


def make_env(
    robot: str = "franka",
    image_size: tuple[int, int] = (128, 128),
    control_freq: float = 20.0,
    max_episode_steps: int = 300,
    seed: int | None = None,
    domain_randomization: bool = False,
    task: str = "pick_place",
) -> PickPlaceEnv:
    """
    Factory function — creates a PickPlaceEnv for the specified robot.

    Args:
        robot: One of 'franka', 'ur5', 'widowx', 'lite6', 'so101'
        image_size: Camera resolution (H, W)
        control_freq: Control loop frequency (Hz)
        max_episode_steps: Episode length limit
        seed: RNG seed for reproducibility
        domain_randomization: Enable DR (default off for Week 1)
        task: One of 'pick_place', 'stack'

    Returns:
        PickPlaceEnv instance ready for reset()/step()
    """
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(
            f"Unknown robot '{robot}'. Supported: {SUPPORTED_ROBOTS}"
        )
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Supported: {SUPPORTED_TASKS}"
        )

    env = PickPlaceEnv(
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
