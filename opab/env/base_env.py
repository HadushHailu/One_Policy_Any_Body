"""
Base manipulation environment for OPAB.

Provides a unified interface across all robot embodiments.
Handles:
  - MuJoCo scene composition (robot + table + objects + cameras)
  - Task-space actions (EE delta + gripper) converted to joint commands via IK
  - Observation extraction (image, proprioception, EE pose)
  - Success detection for pick-and-place tasks

Design decisions:
  - Actions are task-space (shared across all embodiments)
  - IK is damped-least-squares (position-only, robust)
  - Images are 84x84 RGB from overhead camera
"""
from __future__ import annotations

import mujoco
import numpy as np
from pathlib import Path
from typing import Optional


# ============================================================
# XML Templates for scene composition
# ============================================================

# ============================================================
# Robot Configuration
# ============================================================

class RobotConfig:
    """Configuration for a specific robot in the environment."""

    def __init__(self, name: str, **kwargs):
        self.name = name

        if name == "franka":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/franka_emika_panda/scene.xml")
            self.ee_body_name = "hand"      # use body (no sites in menagerie Franka)
            self.n_arm_joints = 7
            self.n_gripper_joints = 2
            self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            self.gripper_joint_names = ["finger_joint1", "finger_joint2"]
            self.gripper_actuator_idx = None  # No gripper actuator in menagerie
            self.n_arm_actuators = 7          # ctrl[0..6] = arm position, ctrl[7] = arm velocity
            self.action_scale = 0.02        # max EE delta per step (m)
            self.ik_damping = 0.01
            self.ik_max_iter = 50
            self.gripper_open = 0.04
            self.gripper_closed = 0.0
            # Standard "ready" pose: arm folded, hand above table
            self.home_qpos = {
                "joint1": 0.0, "joint2": -0.785, "joint3": 0.0,
                "joint4": -2.356, "joint5": 0.0, "joint6": 1.571,
                "joint7": 0.785,
                "finger_joint1": 0.04, "finger_joint2": 0.04,
            }
            self.table_pos = np.array([0.5, 0.0, 0.2])
            self.table_half_size = np.array([0.25, 0.3, 0.2])
            self.cube_pos = np.array([0.5, 0.05, 0.42])
            self.cube_size = 0.015
            self.cube_mass = 0.05
            self.target_pos = np.array([0.5, -0.15, 0.405])
            self.cam_pos = np.array([0.5, 0.0, 1.3])
            self.cube_randomize_range = 0.05
            self.success_threshold = 0.03

        elif name == "ur5":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/universal_robots_ur5e/scene.xml")
            self.ee_body_name = "wrist_3_link"
            self.ee_site_name = "attachment_site"
            self.n_arm_joints = 6
            self.n_gripper_joints = 0  # no gripper in menagerie UR5e
            self.arm_joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            self.gripper_joint_names = []
            self.gripper_actuator_idx = None
            self.action_scale = 0.02
            self.ik_damping = 0.01
            self.ik_max_iter = 50
            self.gripper_open = 0.04
            self.gripper_closed = 0.0
            # UR5e "ready" pose: arm reaches forward and down
            self.home_qpos = {
                "shoulder_pan_joint": -1.571,
                "shoulder_lift_joint": -1.571,
                "elbow_joint": 1.571,
                "wrist_1_joint": -1.571,
                "wrist_2_joint": -1.571,
                "wrist_3_joint": 0.0,
            }
            self.table_pos = np.array([0.4, -0.3, 0.2])
            self.table_half_size = np.array([0.25, 0.3, 0.2])
            self.cube_pos = np.array([0.4, -0.25, 0.42])
            self.cube_size = 0.015
            self.cube_mass = 0.05
            self.target_pos = np.array([0.4, -0.45, 0.405])
            self.cam_pos = np.array([0.4, -0.3, 1.3])
            self.cube_randomize_range = 0.05
            self.success_threshold = 0.03

        elif name == "so101":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/robotstudio_so101/scene.xml")
            self.ee_site_name = "gripperframe"
            self.ee_body_name = "gripper"
            self.n_arm_joints = 5
            self.n_gripper_joints = 1
            self.arm_joint_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll"
            ]
            self.gripper_joint_names = ["gripper"]
            self.gripper_actuator_idx = 5  # gripper is actuator index 5
            self.action_scale = 0.02
            self.ik_damping = 0.05
            self.ik_max_iter = 30
            self.gripper_open = -0.17     # hinge joint: open angle
            self.gripper_closed = 1.5     # hinge joint: closed angle
            # SO-101 "above table" pose
            self.home_qpos = {
                "shoulder_pan": 0.0, "shoulder_lift": 0.3,
                "elbow_flex": -0.3, "wrist_flex": 0.5,
                "wrist_roll": 0.0, "gripper": -0.17,
            }
            # Robot at origin on floor. Workspace forward at x≈0.35-0.50,
            # z range 0.03-0.25. Table at floor level, cube on table.
            self.table_pos = np.array([0.35, 0.0, 0.025])
            self.table_half_size = np.array([0.15, 0.15, 0.025])
            self.cube_pos = np.array([0.38, 0.05, 0.065])
            self.cube_size = 0.015
            self.cube_mass = 0.03
            self.target_pos = np.array([0.38, -0.08, 0.055])
            self.cam_pos = np.array([0.35, 0.0, 0.65])
            self.cube_randomize_range = 0.04
            self.success_threshold = 0.03
        else:
            raise ValueError(f"Unknown robot: {name}")


# ============================================================
# Main Environment
# ============================================================

class PickPlaceEnv:
    """
    Unified pick-and-place environment.

    Accepts task-space actions: [dx, dy, dz, gripper] where:
      - dx, dy, dz: EE position deltas in meters
      - gripper: 0.0 = open, 1.0 = closed

    Works for Franka (7-DOF), UR5 (6-DOF), SO-101 (6-DOF).
    """

    def __init__(
        self,
        robot: str = "franka",
        image_size: tuple[int, int] = (84, 84),
        control_freq: float = 20.0,
        max_episode_steps: int = 300,
        seed: Optional[int] = None,
        kinematic_mode: bool = True,
    ):
        self.robot_name = robot
        self.cfg = RobotConfig(robot)
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.kinematic_mode = kinematic_mode

        # Physics = 500Hz, control = 20Hz → 25 substeps
        self.control_dt = 1.0 / control_freq

        # Load model
        self._load_model()

        # Physics substeps per control step
        self.n_substeps = max(1, int(self.control_dt / self.model.opt.timestep))

        # Cache IDs
        self._cache_ids()

        # Renderer
        self.renderer = mujoco.Renderer(self.model, *image_size)

        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    def _load_model(self):
        """Load the MuJoCo model for this robot with task objects injected."""
        project_root = Path(__file__).resolve().parents[2]

        # All robots use the same approach: load menagerie scene.xml,
        # inject task objects (table, cube, target, camera)
        scene_path = project_root / self.cfg.scene_path
        scene_dir = scene_path.parent

        # Read the original scene XML
        with open(scene_path) as f:
            xml = f.read()

        # Build objects XML to inject
        cfg = self.cfg
        objects = f"""
    <!-- OPAB: Injected task objects -->
    <body name="table" pos="{cfg.table_pos[0]} {cfg.table_pos[1]} {cfg.table_pos[2]}">
      <geom name="table_surface" type="box"
            size="{cfg.table_half_size[0]} {cfg.table_half_size[1]} {cfg.table_half_size[2]}"
            rgba="0.55 0.4 0.25 1" contype="1" conaffinity="1"
            friction="0.8 0.005 0.001" />
    </body>

    <body name="cube" pos="{cfg.cube_pos[0]} {cfg.cube_pos[1]} {cfg.cube_pos[2]}">
      <freejoint name="cube_joint" />
      <geom name="cube_geom" type="box" size="{cfg.cube_size} {cfg.cube_size} {cfg.cube_size}"
            mass="{cfg.cube_mass}" rgba="0.9 0.1 0.1 1" condim="4"
            friction="1.0 0.005 0.001" contype="1" conaffinity="1" />
    </body>

    <site name="target_zone" pos="{cfg.target_pos[0]} {cfg.target_pos[1]} {cfg.target_pos[2]}"
          size="0.04 0.04 0.002" rgba="0.1 0.8 0.1 0.4" type="box" />

    <camera name="overhead_cam" pos="{cfg.cam_pos[0]} {cfg.cam_pos[1]} {cfg.cam_pos[2]}"
            xyaxes="1 0 0 0 -1 0" fovy="50" />
"""
        # Inject before </worldbody>
        xml = xml.replace("</worldbody>", objects + "  </worldbody>")

        # Write temp file in same directory so includes/meshes resolve
        tmp_path = scene_dir / "_opab_tmp_scene.xml"
        try:
            tmp_path.write_text(xml)
            self.model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self.data = mujoco.MjData(self.model)

    def _cache_ids(self):
        """Cache body/joint/site IDs for fast access."""
        # End-effector
        if hasattr(self.cfg, 'ee_site_name') and self.cfg.ee_site_name:
            self._ee_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.ee_site_name
            )
            self._use_site_for_ee = True
        else:
            self._ee_site_id = -1
            self._use_site_for_ee = False

        if self.cfg.ee_body_name:
            self._ee_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.ee_body_name
            )
        else:
            self._ee_body_id = -1

        # Arm joint IDs and DOF addresses
        self._arm_joint_ids = []
        self._arm_dof_ids = []
        self._arm_qpos_ids = []
        for name in self.cfg.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found in model"
            self._arm_joint_ids.append(jid)
            self._arm_dof_ids.append(self.model.jnt_dofadr[jid])
            self._arm_qpos_ids.append(self.model.jnt_qposadr[jid])

        # Cube (may or may not exist depending on scene composition)
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )

        # Target zone
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_zone"
        )

        # Camera
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"
        )
        if self._cam_id < 0:
            # Try other camera names from menagerie scenes
            for cam_name in ["overhead", "fixed"]:
                self._cam_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
                )
                if self._cam_id >= 0:
                    break

    def _get_ee_pos(self) -> np.ndarray:
        """Get current end-effector position."""
        if self._use_site_for_ee and self._ee_site_id >= 0:
            return self.data.site_xpos[self._ee_site_id].copy()
        elif self._ee_body_id >= 0:
            return self.data.xpos[self._ee_body_id].copy()
        else:
            raise RuntimeError("No EE reference found")

    def _get_ee_mat(self) -> np.ndarray:
        """Get current EE rotation matrix (3x3)."""
        if self._use_site_for_ee and self._ee_site_id >= 0:
            return self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        elif self._ee_body_id >= 0:
            return self.data.xmat[self._ee_body_id].reshape(3, 3).copy()
        else:
            raise RuntimeError("No EE reference found")

    def _get_ee_quat(self) -> np.ndarray:
        """Get current EE quaternion (w, x, y, z)."""
        mat = self._get_ee_mat()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    # ============================================================
    # Inverse Kinematics (Damped Least Squares, position-only)
    # ============================================================

    def _ik_solve(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Position-only Damped Least Squares IK with adaptive damping.

        Given target EE position, compute joint angles that achieve it.
        Uses the current joint config as seed.  Damping starts high and
        decreases as the solver converges, which helps escape singularities
        while still achieving precision.

        Args:
            target_pos: (3,) desired EE position in world frame

        Returns:
            qpos: (n_arm_joints,) target joint positions
        """
        # Save state (we'll modify data for IK iterations)
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        qpos = np.array([self.data.qpos[i] for i in self._arm_qpos_ids])
        base_damping = self.cfg.ik_damping

        for it in range(self.cfg.ik_max_iter):
            # Set arm joints and forward kinematics
            for i, qpos_idx in enumerate(self._arm_qpos_ids):
                self.data.qpos[qpos_idx] = qpos[i]
            mujoco.mj_forward(self.model, self.data)

            # Current EE position
            ee_pos = self._get_ee_pos()
            error = target_pos - ee_pos
            err_norm = np.linalg.norm(error)

            if err_norm < 1e-4:
                break

            # Compute position Jacobian (3 x nv)
            jacp = np.zeros((3, self.model.nv))
            if self._use_site_for_ee and self._ee_site_id >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, None, self._ee_site_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp, None, self._ee_body_id)

            # Extract arm-joint columns only
            J = jacp[:, self._arm_dof_ids]  # (3, n_arm_joints)

            # Adaptive damping: high early (escape singularities), low later (precision)
            lam = base_damping * max(0.1, 1.0 - it / self.cfg.ik_max_iter)

            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
            JJT = J @ J.T + lam**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)

            # Step size limiting to prevent large jumps
            dq_norm = np.linalg.norm(dq)
            max_step = 0.5  # max joint angle change per iteration (rad)
            if dq_norm > max_step:
                dq = dq * (max_step / dq_norm)

            # Update joint positions
            qpos = qpos + dq

            # Clamp to joint limits
            for i, jid in enumerate(self._arm_joint_ids):
                lo, hi = self.model.jnt_range[jid]
                if lo < hi:  # Only clamp if limits are valid
                    qpos[i] = np.clip(qpos[i], lo, hi)

        # Restore original state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)

        return qpos

    # ============================================================
    # Gripper Control
    # ============================================================

    def _set_gripper(self, cmd: float):
        """
        Set gripper position.
        cmd = 0.0 → fully open
        cmd = 1.0 → fully closed
        """
        if self.robot_name == "franka":
            # Franka: no gripper actuator in menagerie — set finger qpos directly
            grip_pos = self.cfg.gripper_open * (1.0 - cmd)
            for gname in self.cfg.gripper_joint_names:
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, gname
                )
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = grip_pos

        elif self.robot_name == "so101":
            # SO-101: hinge gripper (open=negative angle, closed=positive)
            grip_pos = self.cfg.gripper_open + cmd * (
                self.cfg.gripper_closed - self.cfg.gripper_open
            )
            self.data.ctrl[self.cfg.gripper_actuator_idx] = grip_pos

        elif self.robot_name == "ur5":
            # UR5 has no gripper in the menagerie model
            pass

    def _get_gripper_pos(self) -> float:
        """Get normalized gripper position [0=open, 1=closed]."""
        if self.robot_name == "franka":
            raw = self.data.qpos[self._arm_qpos_ids[-1] + 1]  # finger_joint1
            return 1.0 - (raw / self.cfg.gripper_open)
        elif self.robot_name == "so101":
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
            )
            raw = self.data.qpos[self.model.jnt_qposadr[jid]]
            # Map from [gripper_open, gripper_closed] → [0, 1]
            span = self.cfg.gripper_closed - self.cfg.gripper_open
            if abs(span) > 1e-6:
                return (raw - self.cfg.gripper_open) / span
            return 0.0
        return 0.0

    # ============================================================
    # Environment Interface
    # ============================================================

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset environment to initial state with randomized cube position."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home pose (above table, ready to manipulate)
        if hasattr(self.cfg, 'home_qpos'):
            for joint_name, qval in self.cfg.home_qpos.items():
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = qval
                    # Also set ctrl for actuators with matching name
                    aid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                    )
                    if aid >= 0:
                        self.data.ctrl[aid] = qval

        # Randomize cube position on table surface
        if self._cube_joint_id >= 0:
            cube_qpos_adr = self.model.jnt_qposadr[self._cube_joint_id]
            r = self.cfg.cube_randomize_range
            dx = self._rng.uniform(-r, r)
            dy = self._rng.uniform(-r, r)
            self.data.qpos[cube_qpos_adr] += dx
            self.data.qpos[cube_qpos_adr + 1] += dy

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one control step.

        Args:
            action: (4,) array [dx, dy, dz, gripper]
                dx, dy, dz: EE position deltas in meters
                gripper: 0.0 = open, 1.0 = closed

        Returns:
            obs: observation dict
            reward: float (sparse: 1 if success, 0 otherwise)
            terminated: bool (success)
            truncated: bool (timeout)
            info: dict with 'success' key
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"

        ee_delta = np.clip(action[:3], -self.cfg.action_scale, self.cfg.action_scale)
        grip_cmd = np.clip(action[3], 0.0, 1.0)

        # Compute target EE position
        current_ee_pos = self._get_ee_pos()
        target_ee_pos = current_ee_pos + ee_delta

        # IK → joint positions
        target_qpos = self._ik_solve(target_ee_pos)

        if self.kinematic_mode:
            # Kinematic mode: pin arm joints to IK solution each substep.
            # Physics runs normally for objects/gripper, but arm joints
            # are overwritten after each substep so actuator forces can't
            # drift them.  This is the standard approach for scripted
            # demo collection in manipulation research.
            for i in range(min(self.cfg.n_arm_joints, self.model.nu)):
                self.data.ctrl[i] = target_qpos[i]
            # Zero velocity actuators (e.g. Franka ctrl[7])
            n_ctrl = getattr(self.cfg, 'n_arm_actuators', self.cfg.n_arm_joints)
            for i in range(self.cfg.n_arm_joints, n_ctrl):
                self.data.ctrl[i] = 0.0
        else:
            # Dynamic mode: use position actuators (realistic but slow)
            for i in range(self.cfg.n_arm_joints):
                self.data.ctrl[i] = target_qpos[i]

        # Apply gripper
        self._set_gripper(grip_cmd)

        # Step physics
        for _ in range(self.n_substeps):
            if self.kinematic_mode:
                # Pin arm joints before each substep
                for i, qpos_idx in enumerate(self._arm_qpos_ids):
                    self.data.qpos[qpos_idx] = target_qpos[i]
                    self.data.qvel[self._arm_dof_ids[i]] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        success = self._check_success()
        reward = float(success)
        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        info = {"success": success, "step": self._step_count}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        """Extract observation from current state."""
        mujoco.mj_forward(self.model, self.data)

        # Render image
        if self._cam_id >= 0:
            self.renderer.update_scene(self.data, camera=self._cam_id)
        else:
            self.renderer.update_scene(self.data)
        image = self.renderer.render()

        # Proprioception: arm joint positions
        qpos_arm = np.array([self.data.qpos[i] for i in self._arm_qpos_ids])

        # EE pose
        ee_pos = self._get_ee_pos()
        ee_quat = self._get_ee_quat()

        # Gripper
        gripper_pos = np.array([self._get_gripper_pos()])

        return {
            "image": image,                # (H, W, 3) uint8
            "proprioception": qpos_arm,    # (n_arm_joints,) float64
            "ee_pos": ee_pos,              # (3,) float64
            "ee_quat": ee_quat,            # (4,) float64 [w,x,y,z]
            "gripper_pos": gripper_pos,    # (1,) float64
        }

    def _check_success(self) -> bool:
        """Check if cube is in the target zone."""
        if self._cube_body_id < 0 or self._target_site_id < 0:
            return False

        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.data.site_xpos[self._target_site_id]

        # XY distance
        dist_xy = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        # Height check: cube above table level
        height_ok = cube_pos[2] > target_pos[2] - 0.02

        return dist_xy < self.cfg.success_threshold and height_ok

    def get_cube_pos(self) -> np.ndarray:
        """Get current cube position (for scripted policies)."""
        if self._cube_body_id >= 0:
            return self.data.xpos[self._cube_body_id].copy()
        return np.zeros(3)

    def get_target_pos(self) -> np.ndarray:
        """Get target zone position (for scripted policies)."""
        if self._target_site_id >= 0:
            return self.data.site_xpos[self._target_site_id].copy()
        return np.zeros(3)

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            del self.renderer
