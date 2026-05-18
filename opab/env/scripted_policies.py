"""
Scripted expert policies for demonstration collection.

These are NOT learned — they are hand-coded state machines that generate
expert trajectories for behavior cloning training.
"""
from __future__ import annotations

import numpy as np


# Robot-specific defaults
_ROBOT_DEFAULTS = {
    "franka": {
        "approach_height": 0.15,
        "grasp_height": -0.005,    # grip_site slightly below cube center → pads
        "lift_height": 0.12,       #   centered on cube for maximum grip overlap
        "move_speed": 0.006,
        "lift_speed": 0.002,        # very slow lift to maintain finger contact
        "grasp_wait_steps": 50,    # wait for fingers to fully close (kp=400)
        "release_wait_steps": 10,
    },
    "ur5": {
        "approach_height": 0.12,
        "grasp_height": 0.005,
        "lift_height": 0.12,
        "move_speed": 0.008,
        "grasp_wait_steps": 15,
        "release_wait_steps": 10,
    },
    "so101": {
        "approach_height": 0.05,   # hover 5cm above cube (EE starts at ~7.5cm)
        "grasp_height": 0.000,     # EE at cube center height for top-down grasp
        "lift_height": 0.06,       # lift 6cm (within workspace)
        "move_speed": 0.005,
        "lift_speed": 0.002,
        "grasp_wait_steps": 30,
        "release_wait_steps": 20,  # longer pause so open gripper is visible
        "retreat_height": 0.05,    # retreat 5cm above place position
    },
}


class ScriptedPickPlace:
    """
    State machine for pick-and-place task.

    States:
      APPROACH  → Move above the cube (hover height)
      DESCEND   → Lower to grasp height
      GRASP     → Close gripper and wait
      LIFT      → Lift the cube
      MOVE      → Move to target position (at lift height)
      PLACE     → Lower to place position
      RELEASE   → Open gripper
      RETREAT   → Move up and away
      DONE      → Episode complete

    Actions are task-space: [dx, dy, dz, gripper]
    """

    def __init__(self, robot_name: str = "franka", **kwargs):
        defaults = _ROBOT_DEFAULTS.get(robot_name, _ROBOT_DEFAULTS["franka"])
        params = {**defaults, **kwargs}

        self.approach_height = params["approach_height"]
        self.grasp_height = params["grasp_height"]
        self.lift_height = params["lift_height"]
        self.move_speed = params["move_speed"]
        self.lift_speed = params.get("lift_speed", self.move_speed)
        self.grasp_wait_steps = params["grasp_wait_steps"]
        self.release_wait_steps = params["release_wait_steps"]
        self.retreat_height = params.get("retreat_height", self.lift_height)

        self.state = "APPROACH"
        self._counter = 0
        self._cube_pos = np.zeros(3)
        self._target_pos = np.zeros(3)
        self._grasp_cube_pos = np.zeros(3)  # cube pos when grasped (fixed ref)

    def reset(self):
        """Reset state machine for a new episode."""
        self.state = "APPROACH"
        self._counter = 0
        self._grasp_cube_pos = np.zeros(3)

    def get_action(
        self, obs: dict, cube_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray | None:
        """
        Compute action based on current state and observation.

        Args:
            obs: dict with 'ee_pos', 'gripper_pos'
            cube_pos: current cube position in world frame
            target_pos: target zone position in world frame

        Returns:
            action: (4,) array [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        self._cube_pos = cube_pos.copy()
        self._target_pos = target_pos.copy()
        ee_pos = obs["ee_pos"]

        if self.state == "APPROACH":
            target = self._cube_pos.copy()
            target[2] += self.approach_height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos[:2] - target[:2]) < 0.003 and \
               abs(ee_pos[2] - target[2]) < 0.005:
                self.state = "DESCEND"
            return np.append(delta, 0.0)

        elif self.state == "DESCEND":
            target = self._cube_pos.copy()
            target[2] += self.grasp_height
            delta = self._move_toward(ee_pos, target)
            if abs(ee_pos[2] - target[2]) < 0.003:
                self.state = "GRASP"
                self._counter = 0
            # Keep fingers OPEN during descent so we don't push the cube
            return np.append(delta, 0.0)

        elif self.state == "GRASP":
            self._counter += 1
            if self._counter >= self.grasp_wait_steps:
                # Record cube position BEFORE lifting — this is our fixed
                # height reference (cube will move with EE after this)
                self._grasp_cube_pos = self._cube_pos.copy()
                self.state = "LIFT"
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.state == "LIFT":
            target = self._grasp_cube_pos.copy()
            target[2] += self.lift_height
            delta = self._move_toward(ee_pos, target, speed=self.lift_speed)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "MOVE"
            return np.append(delta, 1.0)

        elif self.state == "MOVE":
            target = self._target_pos.copy()
            target[2] = self._grasp_cube_pos[2] + self.lift_height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos[:2] - target[:2]) < 0.003:
                self.state = "PLACE"
            return np.append(delta, 1.0)

        elif self.state == "PLACE":
            target = self._target_pos.copy()
            target[2] += 0.02  # slightly above target
            delta = self._move_toward(ee_pos, target)
            if abs(ee_pos[2] - target[2]) < 0.005:
                self.state = "RELEASE"
                self._counter = 0
            return np.append(delta, 1.0)

        elif self.state == "RELEASE":
            self._counter += 1
            if self._counter >= self.release_wait_steps:
                self.state = "RETREAT"
            return np.array([0.0, 0.0, 0.0, 0.0])

        elif self.state == "RETREAT":
            target = self._target_pos.copy()
            target[2] += self.retreat_height
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 0.0)

        return None  # DONE

    def _move_toward(self, current: np.ndarray, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        """Compute delta to move toward target at fixed speed."""
        s = speed if speed is not None else self.move_speed
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < s:
            return diff
        return diff / dist * s

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"


class ScriptedStack:
    """
    State machine for cube stacking task.

    Picks cube_A and stacks it on top of cube_B.

    States:
      APPROACH  → Move above cube_A
      DESCEND   → Lower to grasp height
      GRASP     → Close gripper and wait
      LIFT      → Lift cube_A
      MOVE      → Move above cube_B
      PLACE     → Lower onto cube_B (cube_B_z + 2*cube_size)
      RELEASE   → Open gripper
      RETREAT   → Move up and away
      DONE      → Episode complete

    Actions are task-space: [dx, dy, dz, gripper]
    """

    def __init__(self, robot_name: str = "franka", cube_size: float = 0.015, **kwargs):
        defaults = _ROBOT_DEFAULTS.get(robot_name, _ROBOT_DEFAULTS["franka"])
        params = {**defaults, **kwargs}

        self.approach_height = params["approach_height"]
        self.grasp_height = params["grasp_height"]
        self.lift_height = params["lift_height"]
        self.move_speed = params["move_speed"]
        self.lift_speed = params.get("lift_speed", self.move_speed)
        self.grasp_wait_steps = params["grasp_wait_steps"]
        self.release_wait_steps = params["release_wait_steps"]
        self.retreat_height = params.get("retreat_height", self.lift_height)
        self.cube_size = cube_size

        self.state = "APPROACH"
        self._counter = 0
        self._cube_a_pos = np.zeros(3)
        self._cube_b_pos = np.zeros(3)
        self._grasp_cube_pos = np.zeros(3)

    def reset(self):
        """Reset state machine for a new episode."""
        self.state = "APPROACH"
        self._counter = 0
        self._grasp_cube_pos = np.zeros(3)

    def get_action(
        self, obs: dict, cube_a_pos: np.ndarray, cube_b_pos: np.ndarray
    ) -> np.ndarray | None:
        """
        Compute action based on current state and observation.

        Args:
            obs: dict with 'ee_pos', 'gripper_pos'
            cube_a_pos: current cube_A position (to be picked)
            cube_b_pos: current cube_B position (stack target)

        Returns:
            action: (4,) array [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        self._cube_a_pos = cube_a_pos.copy()
        self._cube_b_pos = cube_b_pos.copy()
        ee_pos = obs["ee_pos"]

        if self.state == "APPROACH":
            target = self._cube_a_pos.copy()
            target[2] += self.approach_height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos[:2] - target[:2]) < 0.003 and \
               abs(ee_pos[2] - target[2]) < 0.005:
                self.state = "DESCEND"
            return np.append(delta, 0.0)

        elif self.state == "DESCEND":
            target = self._cube_a_pos.copy()
            target[2] += self.grasp_height
            delta = self._move_toward(ee_pos, target)
            if abs(ee_pos[2] - target[2]) < 0.003:
                self.state = "GRASP"
                self._counter = 0
            return np.append(delta, 0.0)

        elif self.state == "GRASP":
            self._counter += 1
            if self._counter >= self.grasp_wait_steps:
                self._grasp_cube_pos = self._cube_a_pos.copy()
                self.state = "LIFT"
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.state == "LIFT":
            target = self._grasp_cube_pos.copy()
            target[2] += self.lift_height
            delta = self._move_toward(ee_pos, target, speed=self.lift_speed)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "MOVE"
            return np.append(delta, 1.0)

        elif self.state == "MOVE":
            # Move above cube_B
            target = self._cube_b_pos.copy()
            target[2] = self._grasp_cube_pos[2] + self.lift_height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos[:2] - target[:2]) < 0.003:
                self.state = "PLACE"
            return np.append(delta, 1.0)

        elif self.state == "PLACE":
            # Place on top of cube_B: target height = cube_B_z + 2*cube_size + small clearance
            target = self._cube_b_pos.copy()
            target[2] += 2 * self.cube_size + 0.005
            delta = self._move_toward(ee_pos, target)
            if abs(ee_pos[2] - target[2]) < 0.005:
                self.state = "RELEASE"
                self._counter = 0
            return np.append(delta, 1.0)

        elif self.state == "RELEASE":
            self._counter += 1
            if self._counter >= self.release_wait_steps:
                self.state = "RETREAT"
            return np.array([0.0, 0.0, 0.0, 0.0])

        elif self.state == "RETREAT":
            target = self._cube_b_pos.copy()
            target[2] += self.retreat_height
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 0.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        """Compute delta to move toward target at fixed speed."""
        s = speed if speed is not None else self.move_speed
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < s:
            return diff
        return diff / dist * s

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"
