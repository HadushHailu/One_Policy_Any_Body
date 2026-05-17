"""
OPAB environment package.

Provides unified manipulation environments for multiple robot embodiments.
"""
from opab.env.base_env import PickPlaceEnv, RobotConfig
from opab.env.scripted_policies import ScriptedPickPlace
from opab.env.domain_randomization import DomainRandomizer, DRConfig


SUPPORTED_ROBOTS = ("franka", "ur5", "so101")


def make_env(
    robot: str = "franka",
    image_size: tuple[int, int] = (84, 84),
    control_freq: float = 20.0,
    max_episode_steps: int = 300,
    seed: int | None = None,
    domain_randomization: bool = False,
    kinematic_mode: bool = True,
) -> PickPlaceEnv:
    """
    Factory function — creates a PickPlaceEnv for the specified robot.

    Args:
        robot: One of 'franka', 'ur5', 'so101'
        image_size: Camera resolution (H, W)
        control_freq: Control loop frequency (Hz)
        max_episode_steps: Episode length limit
        seed: RNG seed for reproducibility
        domain_randomization: Enable DR (default off for Week 1)
        kinematic_mode: If True, set arm joints directly (instant IK tracking).
                        If False, use position actuators (realistic but slower).

    Returns:
        PickPlaceEnv instance ready for reset()/step()
    """
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(
            f"Unknown robot '{robot}'. Supported: {SUPPORTED_ROBOTS}"
        )

    env = PickPlaceEnv(
        robot=robot,
        image_size=image_size,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        seed=seed,
        kinematic_mode=kinematic_mode,
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
    "DomainRandomizer",
    "DRConfig",
    "SUPPORTED_ROBOTS",
]
