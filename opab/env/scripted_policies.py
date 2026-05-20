"""
Scripted expert policies for demonstration collection.

These are NOT learned — they are hand-coded state machines that generate
expert trajectories for behavior cloning training.

Tasks:
  - ScriptedReach: Move EE to target position
  - ScriptedPickPlace: Pick cube, place at target zone
  - ScriptedPush: Push cube to target zone (no grasp)
  - ScriptedStack: Pick cube_A, stack on cube_B
  - ScriptedPegInsertion: Pick peg, insert into hole
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
        "release_wait_steps": 5,
        "place_height": 0.04,  # release cube slightly above table (avoid slam)
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
    "widowx": {
        "approach_height": 0.08,   # hover 8cm above cube
        "grasp_height": -0.003,    # slightly below cube center for good grip
        "lift_height": 0.08,       # lift 8cm (within ~30cm workspace)
        "move_speed": 0.006,
        "lift_speed": 0.002,
        "grasp_wait_steps": 25,
        "release_wait_steps": 10,
        "retreat_height": 0.06,
    },
    "lite6": {
        "approach_height": 0.10,   # hover 10cm above cube
        "grasp_height": 0.002,     # just above cube center
        "lift_height": 0.10,       # lift 10cm
        "move_speed": 0.007,
        "lift_speed": 0.003,
        "grasp_wait_steps": 20,
        "release_wait_steps": 8,
        "retreat_height": 0.08,
        "place_height": 0.03,
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
        self.place_height = params.get("place_height", 0.02)

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
            target[2] += self.place_height
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


# ============================================================
# ScriptedReach — Move EE to target position
# ============================================================

class ScriptedReach:
    """
    State machine for reach task.

    Simply moves the end-effector to a target 3D position.
    Gripper stays open throughout.

    States:
      MOVE → Move toward target
      HOLD → Hold at target for a few steps
      DONE → Episode complete

    Actions: [dx, dy, dz, gripper]
    """

    def __init__(self, robot_name: str = "franka", **kwargs):
        defaults = _ROBOT_DEFAULTS.get(robot_name, _ROBOT_DEFAULTS["franka"])
        params = {**defaults, **kwargs}
        self.move_speed = params["move_speed"]
        self.hold_steps = kwargs.get("hold_steps", 20)

        self.state = "MOVE"
        self._counter = 0

    def reset(self):
        self.state = "MOVE"
        self._counter = 0

    def get_action(
        self, obs: dict, target_pos: np.ndarray
    ) -> np.ndarray | None:
        """
        Args:
            obs: dict with 'ee_pos'
            target_pos: 3D target position for end-effector

        Returns:
            action: (4,) array [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        ee_pos = obs["ee_pos"]

        if self.state == "MOVE":
            delta = self._move_toward(ee_pos, target_pos)
            if np.linalg.norm(ee_pos - target_pos) < 0.01:
                self.state = "HOLD"
                self._counter = 0
            return np.append(delta, 0.0)

        elif self.state == "HOLD":
            self._counter += 1
            if self._counter >= self.hold_steps:
                self.state = "DONE"
            # Small corrections to maintain position
            delta = self._move_toward(ee_pos, target_pos)
            return np.append(delta, 0.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < self.move_speed:
            return diff
        return diff / dist * self.move_speed

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"


# ============================================================
# ScriptedPush — Push cube to target without grasping
# ============================================================

class ScriptedPush:
    """
    State machine for push task (non-prehensile manipulation).

    Moves behind the cube (opposite side from target), then pushes through
    the cube toward the target. Gripper stays closed to form a flat pusher.

    States:
      APPROACH    → Move above cube (hover)
      DESCEND     → Lower to push height (cube center level)
      ALIGN       → Move behind the cube (away from target)
      PUSH        → Push cube toward target
      RETREAT     → Move up and away
      DONE        → Complete

    Actions: [dx, dy, dz, gripper]
    """

    def __init__(self, robot_name: str = "franka", **kwargs):
        defaults = _ROBOT_DEFAULTS.get(robot_name, _ROBOT_DEFAULTS["franka"])
        params = {**defaults, **kwargs}

        self.approach_height = params["approach_height"]
        self.move_speed = params["move_speed"]
        self.push_offset = kwargs.get("push_offset", 0.05)  # how far behind cube to start
        self.retreat_height = params.get("retreat_height", params["lift_height"])

        self.state = "APPROACH"
        self._counter = 0
        self._cube_pos = np.zeros(3)
        self._target_pos = np.zeros(3)

    def reset(self):
        self.state = "APPROACH"
        self._counter = 0

    def get_action(
        self, obs: dict, cube_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray | None:
        """
        Args:
            obs: dict with 'ee_pos'
            cube_pos: current cube position
            target_pos: target zone position

        Returns:
            action: (4,) [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        self._cube_pos = cube_pos.copy()
        self._target_pos = target_pos.copy()
        ee_pos = obs["ee_pos"]

        if self.state == "APPROACH":
            # Move above the "behind" position (avoids contacting cube)
            push_dir = self._target_pos[:2] - self._cube_pos[:2]
            push_dir_norm = np.linalg.norm(push_dir)
            if push_dir_norm > 1e-6:
                push_dir = push_dir / push_dir_norm
            target = self._cube_pos.copy()
            target[:2] -= push_dir * self.push_offset  # behind cube in XY
            target[2] += self.approach_height           # at hover height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.005:
                self.state = "DESCEND"
            return np.append(delta, 1.0)  # gripper closed (flat pusher)

        elif self.state == "DESCEND":
            # Lower to cube height at the "behind" position
            push_dir = self._target_pos[:2] - self._cube_pos[:2]
            push_dir_norm = np.linalg.norm(push_dir)
            if push_dir_norm > 1e-6:
                push_dir = push_dir / push_dir_norm
            target = self._cube_pos.copy()
            target[:2] -= push_dir * self.push_offset
            target[2] = self._cube_pos[2]  # at cube center height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.005:
                self.state = "PUSH"
            return np.append(delta, 1.0)

        elif self.state == "ALIGN":
            # (kept for compat but skipped in new flow)
            self.state = "PUSH"
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.state == "PUSH":
            # Push through cube toward target (and slightly past)
            push_dir = self._target_pos[:2] - self._cube_pos[:2]
            push_dir_norm = np.linalg.norm(push_dir)
            if push_dir_norm > 1e-6:
                push_dir = push_dir / push_dir_norm
            target = self._target_pos.copy()
            target[2] = self._cube_pos[2]  # maintain push height
            target[:2] += push_dir * 0.02  # overshoot by 2cm

            delta = self._move_toward(ee_pos, target)
            # Done pushing when cube is near target
            cube_to_target = np.linalg.norm(self._cube_pos[:2] - self._target_pos[:2])
            if cube_to_target < 0.03:
                self.state = "RETREAT"
            return np.append(delta, 1.0)

        elif self.state == "RETREAT":
            target = ee_pos.copy()
            target[2] += self.retreat_height
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 1.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < self.move_speed:
            return diff
        return diff / dist * self.move_speed

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"


# ============================================================
# ScriptedPegInsertion — Pick peg, align, insert into hole
# ============================================================

class ScriptedPegInsertion:
    """
    State machine for peg insertion task (precision manipulation).

    Picks up a cylindrical peg and inserts it into a hole.
    Requires tight XY alignment before descent.

    States:
      APPROACH   → Move above peg
      DESCEND    → Lower to grasp peg
      GRASP      → Close gripper
      LIFT       → Lift peg
      ALIGN      → Move above hole (precise XY alignment)
      INSERT     → Lower peg into hole slowly
      RELEASE    → Open gripper
      RETREAT    → Move up
      DONE       → Complete

    Actions: [dx, dy, dz, gripper]
    """

    def __init__(self, robot_name: str = "franka", **kwargs):
        defaults = _ROBOT_DEFAULTS.get(robot_name, _ROBOT_DEFAULTS["franka"])
        params = {**defaults, **kwargs}

        self.approach_height = params["approach_height"]
        # Peg grasp: EE slightly above peg center for reachability
        self.grasp_height = kwargs.get("grasp_height", 0.015)  # 15mm above center
        # Lift high enough to clear hole rim AND stay in good workspace.
        # The arm needs height to traverse laterally without singularity issues.
        self.lift_height = params.get("lift_height", 0.12)  # full lift height
        self.move_speed = params["move_speed"]
        self.lift_speed = params.get("lift_speed", self.move_speed)
        self.insert_speed = kwargs.get("insert_speed", self.move_speed * 0.3)  # very slow insertion
        self.grasp_wait_steps = max(params["grasp_wait_steps"], 20)  # extra time for thin peg
        self.release_wait_steps = params["release_wait_steps"]
        self.retreat_height = params.get("retreat_height", self.lift_height)

        self.state = "APPROACH"
        self._counter = 0
        self._peg_pos = np.zeros(3)
        self._hole_pos = np.zeros(3)
        self._grasp_peg_pos = np.zeros(3)
        self._hole_depth_approx = 0.040  # conservative: just below hole rim

    def reset(self):
        self.state = "APPROACH"
        self._counter = 0
        self._grasp_peg_pos = np.zeros(3)

    def get_action(
        self, obs: dict, peg_pos: np.ndarray, hole_pos: np.ndarray
    ) -> np.ndarray | None:
        """
        Args:
            obs: dict with 'ee_pos'
            peg_pos: current peg position
            hole_pos: hole bottom position

        Returns:
            action: (4,) [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        self._peg_pos = peg_pos.copy()
        self._hole_pos = hole_pos.copy()
        ee_pos = obs["ee_pos"]

        if self.state == "APPROACH":
            target = self._peg_pos.copy()
            target[2] += self.approach_height
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos[:2] - target[:2]) < 0.003 and \
               abs(ee_pos[2] - target[2]) < 0.005:
                self.state = "DESCEND"
            return np.append(delta, 0.0)

        elif self.state == "DESCEND":
            target = self._peg_pos.copy()
            target[2] += self.grasp_height
            delta = self._move_toward(ee_pos, target)
            if abs(ee_pos[2] - target[2]) < 0.003:
                self.state = "GRASP"
                self._counter = 0
            return np.append(delta, 0.0)

        elif self.state == "GRASP":
            self._counter += 1
            if self._counter >= self.grasp_wait_steps:
                self._grasp_peg_pos = self._peg_pos.copy()
                self.state = "LIFT"
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.state == "LIFT":
            target = self._grasp_peg_pos.copy()
            target[2] += self.lift_height
            delta = self._move_toward(ee_pos, target, speed=self.lift_speed)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "ALIGN"
            return np.append(delta, 1.0)

        elif self.state == "ALIGN":
            # Move so the PEG (not EE) is above the hole
            # Compensate for peg-to-EE offset
            peg_ee_offset = self._peg_pos[:2] - ee_pos[:2]  # how far peg is from EE in XY
            target = self._hole_pos.copy()
            target[:2] -= peg_ee_offset  # move EE so peg ends up above hole
            target[2] = self._grasp_peg_pos[2] + self.lift_height
            delta = self._move_toward(ee_pos, target)
            # Check PEG alignment (not EE) — peg XY within 3mm of hole
            peg_hole_dist = np.linalg.norm(self._peg_pos[:2] - self._hole_pos[:2])
            if peg_hole_dist < 0.001:
                self.state = "INSERT"
            return np.append(delta, 1.0)

        elif self.state == "INSERT":
            # Descend slowly while maintaining XY alignment over hole.
            # Key insight: minimal descent needed (peg center just below rim).
            # The arm is in a better kinematic region at lower heights.
            peg_ee_offset = self._peg_pos[:2] - ee_pos[:2]
            hole_body_xy = self._hole_pos[:2]
            
            # XY: keep peg centered over hole
            xy_target = hole_body_xy - peg_ee_offset
            xy_delta = xy_target - ee_pos[:2]
            
            # Z: target just enough for peg center to cross below rim
            # Rim is at hole_pos.z + hole_depth. We want peg center ~5mm below rim.
            z_target = self._hole_pos[2] + self._hole_depth_approx - 0.005
            z_delta = z_target - ee_pos[2]
            
            # Combine with Z moving slowly
            delta = np.array([xy_delta[0], xy_delta[1], z_delta])
            # Limit total speed
            dist = np.linalg.norm(delta)
            if dist > self.insert_speed:
                delta = delta / dist * self.insert_speed
            
            # Check if peg is inserted (peg center below hole rim)
            if self._peg_pos[2] < self._hole_pos[2] + self._hole_depth_approx:
                self.state = "RELEASE"
                self._counter = 0
            return np.append(delta, 1.0)

        elif self.state == "RELEASE":
            self._counter += 1
            if self._counter >= self.release_wait_steps:
                self.state = "RETREAT"
            return np.array([0.0, 0.0, 0.0, 0.0])

        elif self.state == "RETREAT":
            target = self._hole_pos.copy()
            target[2] += self.retreat_height
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 0.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        s = speed if speed is not None else self.move_speed
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < s:
            return diff
        return diff / dist * s

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"


class ScriptedDrawerOpen:
    """
    Scripted policy for drawer_open task.
    States: APPROACH → GRASP → PULL → RELEASE → RETREAT → DONE
    """

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.state = "APPROACH"
        self._counter = 0

        # Tunable parameters
        self.move_speed = 0.006
        self.pull_speed = 0.004
        self.grasp_wait_steps = 10
        self.release_wait_steps = 5

    def get_action(self, obs: dict, handle_pos: np.ndarray) -> np.ndarray | None:
        """
        Args:
            obs: dict with 'ee_pos'
            handle_pos: current drawer handle position
        Returns:
            action: (4,) [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        ee_pos = obs["ee_pos"]
        self._handle_pos = handle_pos.copy()

        if self.state == "APPROACH":
            # Move to handle position (slightly in front, open gripper)
            target = self._handle_pos.copy()
            target[1] -= 0.02  # approach from front (Y-)
            target[2] += 0.01  # slightly above
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.005:
                self.state = "ALIGN"
            return np.append(delta, 0.0)

        elif self.state == "ALIGN":
            # Align with handle precisely
            target = self._handle_pos.copy()
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.003:
                self.state = "GRASP"
                self._counter = 0
            return np.append(delta, 0.0)

        elif self.state == "GRASP":
            self._counter += 1
            if self._counter >= self.grasp_wait_steps:
                self.state = "PULL"
            return np.array([0.0, 0.0, 0.0, 1.0])

        elif self.state == "PULL":
            # Pull in -Y direction (toward robot) while gripping
            target = ee_pos.copy()
            target[1] -= self.pull_speed
            delta = target - ee_pos
            # Check if drawer is open enough
            try:
                joint_id = self.env.model.joint("drawer_joint").id
                qpos_idx = self.env.model.jnt_qposadr[joint_id]
                drawer_pos = self.env.data.qpos[qpos_idx]
                if drawer_pos >= self.cfg.drawer_slide_range * 0.8:
                    self.state = "RELEASE"
                    self._counter = 0
            except Exception:
                pass
            return np.append(delta, 1.0)

        elif self.state == "RELEASE":
            self._counter += 1
            if self._counter >= self.release_wait_steps:
                self.state = "RETREAT"
            return np.array([0.0, 0.0, 0.0, 0.0])

        elif self.state == "RETREAT":
            target = ee_pos.copy()
            target[2] += 0.05
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 0.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        s = speed if speed is not None else self.move_speed
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < s:
            return diff
        return diff / dist * s

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"


class ScriptedTurnFaucet:
    """
    Scripted policy for turn_faucet task.
    States: APPROACH → CONTACT → PUSH → RETREAT → DONE

    Strategy: approach the lever tip, then push it tangentially to rotate 90 degrees.
    No grasping needed — just push contact.
    """

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.state = "APPROACH"
        self._counter = 0

        # Tunable parameters
        self.move_speed = 0.006
        self.push_speed = 0.003

    def get_action(self, obs: dict, lever_tip_pos: np.ndarray, faucet_base_pos: np.ndarray) -> np.ndarray | None:
        """
        Args:
            obs: dict with 'ee_pos'
            lever_tip_pos: current position of the lever tip
            faucet_base_pos: position of faucet base (pivot point)
        Returns:
            action: (4,) [dx, dy, dz, gripper], or None if DONE
        """
        if self.state == "DONE":
            return None

        ee_pos = obs["ee_pos"]
        self._tip_pos = lever_tip_pos.copy()
        self._base_pos = faucet_base_pos.copy()

        if self.state == "APPROACH":
            # Move to just behind the lever tip (push from +X toward +Y rotation)
            target = self._tip_pos.copy()
            target[2] += 0.005  # slightly above lever
            target[1] -= 0.015  # approach from behind (will push in +Y)
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.004:
                self.state = "DESCEND"
            return np.append(delta, 1.0)  # closed fist for pushing

        elif self.state == "DESCEND":
            # Lower to lever height
            target = self._tip_pos.copy()
            target[1] -= 0.005  # just behind tip
            delta = self._move_toward(ee_pos, target)
            if np.linalg.norm(ee_pos - target) < 0.003:
                self.state = "PUSH"
            return np.append(delta, 1.0)

        elif self.state == "PUSH":
            # Push tangentially — the lever is along +X from base,
            # so pushing in +Y rotates it CCW around Z axis
            # Continuously track the tip and push perpendicular to the lever
            lever_vec = self._tip_pos[:2] - self._base_pos[:2]
            lever_len = np.linalg.norm(lever_vec)
            if lever_len > 0.001:
                lever_dir = lever_vec / lever_len
                # Tangent direction (perpendicular, CCW)
                tangent = np.array([-lever_dir[1], lever_dir[0]])
            else:
                tangent = np.array([0.0, 1.0])

            # Move toward the tip, then push tangentially
            to_tip = self._tip_pos[:2] - ee_pos[:2]
            # Blend: mostly tangential push, slight correction toward tip
            push_dir = 0.8 * tangent + 0.2 * (to_tip / (np.linalg.norm(to_tip) + 1e-6))
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

            delta = np.array([push_dir[0] * self.push_speed,
                              push_dir[1] * self.push_speed,
                              0.0])

            # Check angle
            try:
                joint_id = self.env.model.joint("faucet_joint").id
                qpos_idx = self.env.model.jnt_qposadr[joint_id]
                angle = self.env.data.qpos[qpos_idx]
                if angle >= self.cfg.faucet_target_angle * 0.8:
                    self.state = "RETREAT"
            except Exception:
                pass
            return np.append(delta, 1.0)

        elif self.state == "RETREAT":
            target = ee_pos.copy()
            target[2] += 0.05
            delta = self._move_toward(ee_pos, target)
            if ee_pos[2] >= target[2] - 0.005:
                self.state = "DONE"
            return np.append(delta, 0.0)

        return None

    def _move_toward(self, current: np.ndarray, target: np.ndarray, speed: float | None = None) -> np.ndarray:
        s = speed if speed is not None else self.move_speed
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < s:
            return diff
        return diff / dist * s

    @property
    def is_done(self) -> bool:
        return self.state == "DONE"
