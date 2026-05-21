"""Task modules for OPAB - one class per manipulation task."""
from opab.tasks.base_task import BaseTask
from opab.tasks.pick_place import PickPlaceTask
from opab.tasks.push import PushTask
from opab.tasks.stack import StackTask
from opab.tasks.peg_insertion import PegInsertionTask
from opab.tasks.drawer_open import DrawerOpenTask
from opab.tasks.turn_faucet import TurnFaucetTask
from opab.tasks.door_open import DoorOpenTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "pick_place": PickPlaceTask,
    "push": PushTask,
    "stack": StackTask,
    "peg_insertion": PegInsertionTask,
    "drawer_open": DrawerOpenTask,
    "turn_faucet": TurnFaucetTask,
    "door_open": DoorOpenTask,
}


def get_task(name: str) -> BaseTask:
    """Factory function to get a task by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]()
