"""Robot modules for OPAB - one class per robot embodiment."""
from opab.robots.base_robot import BaseRobot
from opab.robots.franka import FrankaRobot
from opab.robots.ur5 import UR5Robot
from opab.robots.lite6 import Lite6Robot
from opab.robots.widowx import WidowXRobot
from opab.robots.so101 import SO101Robot

ROBOT_REGISTRY: dict[str, type[BaseRobot]] = {
    "franka": FrankaRobot,
    "ur5": UR5Robot,
    "lite6": Lite6Robot,
    "widowx": WidowXRobot,
    "so101": SO101Robot,
}


def get_robot(name: str, **kwargs) -> BaseRobot:
    """Factory function to get a robot by name."""
    if name not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown robot: {name}. Available: {list(ROBOT_REGISTRY.keys())}")
    return ROBOT_REGISTRY[name](**kwargs)
