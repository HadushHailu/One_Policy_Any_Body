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
            self.ee_body_name = "hand"
            self.ee_site_name = "grip_site"  # injected at fingertip center
            self.n_arm_joints = 7
            self.n_gripper_joints = 2
            self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            self.gripper_joint_names = ["finger_joint1", "finger_joint2"]
            self.gripper_actuator_idx = [8, 9]  # injected finger actuators
            self.n_arm_actuators = 8          # ctrl[0..6] = arm position, ctrl[7] = arm velocity
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
            self.cube_mass = 0.1
            self.target_pos = np.array([0.5, -0.15, 0.405])
            self.cam_pos = np.array([0.5, 0.0, 1.3])
            self.cube_randomize_range = 0.05
            self.success_threshold = 0.03

        elif name == "ur5":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/universal_robots_ur5e/scene_gripper.xml")
            self.ee_body_name = "gripper_base"
            self.ee_site_name = "pinch"
            self.n_arm_joints = 6
            self.n_gripper_joints = 8  # Robotiq 2F85 linkage joints
            self.arm_joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            self.gripper_joint_names = ["right_driver_joint"]
            self.gripper_actuator_idx = 6  # fingers_actuator
            self.action_scale = 0.02
            self.ik_damping = 0.01
            self.ik_max_iter = 50
            self.gripper_open = 0.0     # ctrl=0 → open
            self.gripper_closed = 255.0  # ctrl=255 → closed
            # UR5e "ready" pose: arm points toward +X (table direction)
            self.home_qpos = {
                "shoulder_pan_joint": -3.14,
                "shoulder_lift_joint": -1.571,
                "elbow_joint": 1.571,
                "wrist_1_joint": -1.571,
                "wrist_2_joint": -1.571,
                "wrist_3_joint": 0.0,
            }
            self.table_pos = np.array([0.45, 0.0, 0.2])
            self.table_half_size = np.array([0.2, 0.2, 0.2])
            self.cube_pos = np.array([0.45, 0.05, 0.42])
            self.cube_size = 0.015
            self.cube_mass = 0.05
            self.target_pos = np.array([0.45, -0.1, 0.405])
            self.cam_pos = np.array([0.45, 0.0, 1.3])
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
            self.gripper_open = 1.5       # hinge joint: open angle (positive)
            self.gripper_closed = -0.17   # hinge joint: closed angle (negative)
            # Top-down grasp pose (inspired by Isaac Lab SO-101 config).
            # wrist_flex=π/2 tilts the gripper 90° downward so fingers
            # open horizontally — perfect for top-down pick-and-place.
            # wrist_roll stays at 0 (no rotation needed).
            self.home_qpos = {
                "shoulder_pan": 0.0, "shoulder_lift": 0.0,
                "elbow_flex": 0.0, "wrist_flex": 1.5708,
                "wrist_roll": 0.0, "gripper": 1.5,
            }
            # Robot at origin on floor. With this pose, EE is at
            # ~[0.244, 0, 0.075] pointing straight down.
            # Place cube on ground level close to the robot.
            self.table_pos = np.array([0.20, 0.0, 0.0])
            self.table_half_size = np.array([0.10, 0.10, 0.001])
            self.cube_pos = np.array([0.20, 0.0, 0.016])
            self.cube_size = 0.015
            self.cube_mass = 0.03
            self.target_pos = np.array([0.20, -0.08, 0.016])
            self.cam_pos = np.array([0.20, 0.0, 0.55])
            self.cube_randomize_range = 0.02
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

    Tasks:
      - 'pick_place': pick cube_A, place at target zone (default)
      - 'stack': pick cube_A, stack on top of cube_B
    """

    def __init__(
        self,
        robot: str = "franka",
        image_size: tuple[int, int] = (84, 84),
        control_freq: float = 20.0,
        max_episode_steps: int = 300,
        seed: Optional[int] = None,
        kinematic_mode: bool = False,
        task: str = "pick_place",
    ):
        self.robot_name = robot
        self.cfg = RobotConfig(robot)
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.kinematic_mode = kinematic_mode
        self.task = task

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
        import re
        project_root = Path(__file__).resolve().parents[2]

        # All robots use the same approach: load menagerie scene.xml,
        # inject task objects (table, cube, target, camera)
        scene_path = project_root / self.cfg.scene_path
        scene_dir = scene_path.parent

        # Read the original scene XML
        with open(scene_path) as f:
            xml = f.read()

        # Inline <include file="..."/> so we can inject into robot bodies
        def _inline_includes(xml_str, base_dir):
            pattern = r'<include\s+file="([^"]+)"\s*/>'
            while re.search(pattern, xml_str):
                def _replace_include(m):
                    inc_path = base_dir / m.group(1)
                    inc_text = inc_path.read_text()
                    # Strip <?xml ...?> and outer <mujoco> tags from included file
                    inc_text = re.sub(r'<\?xml[^>]*\?>', '', inc_text)
                    inc_text = re.sub(r'<mujoco[^>]*>', '', inc_text, count=1)
                    inc_text = re.sub(r'</mujoco\s*>', '', inc_text, count=1)
                    return inc_text
                xml_str = re.sub(pattern, _replace_include, xml_str, count=1)
            return xml_str

        xml = _inline_includes(xml, scene_dir)

        # For Franka: inject a grip_site at the fingertip center
        # (the "hand" body is the palm — 0.1034m above the actual grasp point)
        if self.cfg.name == "franka":
            xml = xml.replace(
                '<body name="left_finger"',
                '<site name="grip_site" pos="0 0 0.1034" size="0.005" '
                'rgba="1 0 0 0.3" group="4" />\n'
                '                      <body name="left_finger"'
            )

        # For Franka: inject finger actuators (menagerie has none)
        if self.cfg.name == "franka":
            # Add position actuators for finger joints before </actuator>
            # kp=200 gives gentle but firm grip (1000 launches the cube)
            finger_actuators = (
                '\n  <position name="finger_actuator1" joint="finger_joint1" '
                'kp="400" ctrlrange="0 0.04" />'
                '\n  <position name="finger_actuator2" joint="finger_joint2" '
                'kp="400" ctrlrange="0 0.04" />'
            )
            xml = xml.replace('</actuator>', finger_actuators + '\n</actuator>')

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
            friction="1.5 0.005 0.001" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" />
    </body>

    <site name="target_zone" pos="{cfg.target_pos[0]} {cfg.target_pos[1]} {cfg.target_pos[2]}"
          size="0.04 0.04 0.002" rgba="0.1 0.8 0.1 0.4" type="box" />

    <camera name="overhead_cam" pos="{cfg.cam_pos[0]} {cfg.cam_pos[1]} {cfg.cam_pos[2]}"
            xyaxes="1 0 0 0 1 0" fovy="50" />
"""

        # For stacking task: inject a second cube (cube_B) at the target position
        if self.task == "stack":
            cube_b_pos = cfg.target_pos.copy()
            # Place cube_B on the surface (same height as cube_A default)
            cube_b_pos[2] = cfg.cube_pos[2]
            objects += f"""
    <body name="cube_b" pos="{cube_b_pos[0]} {cube_b_pos[1]} {cube_b_pos[2]}">
      <freejoint name="cube_b_joint" />
      <geom name="cube_b_geom" type="box" size="{cfg.cube_size} {cfg.cube_size} {cfg.cube_size}"
            mass="{cfg.cube_mass}" rgba="0.1 0.1 0.9 1" condim="4"
            friction="1.5 0.005 0.001" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" />
    </body>
"""
        # Inject before the LAST </worldbody> (after inlining there may be multiple)
        idx = xml.rfind("</worldbody>")
        xml = xml[:idx] + objects + "  </worldbody>" + xml[idx + len("</worldbody>"):]

        # Write temp file in same directory so includes/meshes resolve
        tmp_path = scene_dir / "_opab_tmp_scene.xml"
        try:
            tmp_path.write_text(xml)
            self.model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self.data = mujoco.MjData(self.model)

        # --- Post-load actuator tuning for sim ---
        # SO-101 menagerie default forcerange (±2.94 Nm) is too low for
        # position-tracking with kp=998 → saturates at ~0.003 rad error.
        # Raise force limits for ARM actuators so joints can track IK,
        # but keep the gripper force moderate to avoid flinging objects.
        if self.robot_name == "so101":
            for i in range(self.cfg.gripper_actuator_idx):  # arm only
                self.model.actuator_forcerange[i] = [-50.0, 50.0]
            # Moderate gripper force
            self.model.actuator_forcerange[self.cfg.gripper_actuator_idx] = [-5.0, 5.0]

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

        # Cube B (stacking target) — only present in 'stack' task
        self._cube_b_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_b"
        )
        self._cube_b_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_b_joint"
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

        # Seed from previous IK solution (ctrl) rather than actual qpos to
        # avoid re-seeding from a lagging configuration in dynamic mode.
        if hasattr(self, '_last_ik_qpos') and self._last_ik_qpos is not None:
            qpos = self._last_ik_qpos.copy()
        else:
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

        self._last_ik_qpos = qpos.copy()
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
            # Franka: use injected finger actuators
            grip_pos = self.cfg.gripper_open * (1.0 - cmd)
            for idx in self.cfg.gripper_actuator_idx:
                if idx < self.model.nu:
                    self.data.ctrl[idx] = grip_pos

        elif self.robot_name == "so101":
            # SO-101: hinge gripper (open=negative angle, closed=positive)
            grip_pos = self.cfg.gripper_open + cmd * (
                self.cfg.gripper_closed - self.cfg.gripper_open
            )
            self.data.ctrl[self.cfg.gripper_actuator_idx] = grip_pos

        elif self.robot_name == "ur5":
            # Robotiq 2F85: ctrl 0=open, 255=closed
            grip_val = self.cfg.gripper_open + cmd * (
                self.cfg.gripper_closed - self.cfg.gripper_open
            )
            self.data.ctrl[self.cfg.gripper_actuator_idx] = grip_val

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
        elif self.robot_name == "ur5":
            # Robotiq 2F85: right_driver_joint range [0, 0.8]
            # 0 = open, 0.8 = closed
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint"
            )
            raw = self.data.qpos[self.model.jnt_qposadr[jid]]
            return raw / 0.8  # normalize to [0, 1]
        return 0.0

    def _count_finger_cube_contacts(self) -> int:
        """Count contacts between finger geoms and the cube."""
        if self._cube_body_id < 0:
            return 0
        count = 0
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        for c in range(self.data.ncon):
            g1 = self.data.contact[c].geom1
            g2 = self.data.contact[c].geom2
            # One geom must be cube, the other must not be table
            if g1 == cube_geom_id or g2 == cube_geom_id:
                other = g2 if g1 == cube_geom_id else g1
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or ""
                if "table" not in other_name and "floor" not in other_name:
                    count += 1
        return count

    # ============================================================
    # Environment Interface
    # ============================================================

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset environment to initial state with randomized cube position."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization (if attached by make_env with DR enabled)
        if hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.reset_to_nominal()
            self.domain_randomizer.randomize(self._rng)

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

        # Randomize cube_B position (stacking task)
        if self._cube_b_joint_id >= 0:
            cb_qpos_adr = self.model.jnt_qposadr[self._cube_b_joint_id]
            r = self.cfg.cube_randomize_range
            dx = self._rng.uniform(-r, r)
            dy = self._rng.uniform(-r, r)
            self.data.qpos[cb_qpos_adr] += dx
            self.data.qpos[cb_qpos_adr + 1] += dy

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._ee_target = self._get_ee_pos().copy()  # track desired EE position
        self._last_ik_qpos = None  # IK will seed from current qpos on first call
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

        # Accumulate desired EE position (not relative to current, which lags)
        self._ee_target = self._ee_target + ee_delta

        # Prevent target from drifting too far from actual EE
        # (if IK can't reach the target, it accumulates indefinitely)
        actual_ee = self._get_ee_pos()
        drift = self._ee_target - actual_ee
        # SO-101 needs more drift room — arm tracks slowly with rate-limited joints
        max_drift = 0.15 if self.robot_name == "so101" else 0.05
        drift_norm = np.linalg.norm(drift)
        if drift_norm > max_drift:
            self._ee_target = actual_ee + drift * (max_drift / drift_norm)

        # IK → joint positions from desired target
        target_qpos = self._ik_solve(self._ee_target)

        # Joint-space rate limiting for SO-101:
        # With kp=998 and forcerange ±50 Nm the max non-saturating
        # joint delta is 50/998 ≈ 0.05 rad.  Clamp ctrl changes to
        # stay below that so actuators track smoothly.
        if self.robot_name == "so101":
            actual_arm_qpos = np.array(
                [self.data.qpos[i] for i in self._arm_qpos_ids]
            )
            max_joint_delta = 0.04  # rad per control step
            delta = target_qpos - actual_arm_qpos
            delta = np.clip(delta, -max_joint_delta, max_joint_delta)
            target_qpos = actual_arm_qpos + delta
            # Re-seed IK from rate-limited target (avoids divergence)
            self._last_ik_qpos = target_qpos.copy()

        # Set arm actuator targets
        for i in range(min(self.cfg.n_arm_joints, self.model.nu)):
            self.data.ctrl[i] = target_qpos[i]
        # Zero velocity actuators (e.g. Franka ctrl[7])
        n_ctrl = getattr(self.cfg, 'n_arm_actuators', self.cfg.n_arm_joints)
        for i in range(self.cfg.n_arm_joints, n_ctrl):
            self.data.ctrl[i] = 0.0

        # Apply gripper
        self._set_gripper(grip_cmd)

        # Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        success = self._check_success()
        reward = float(success)
        # Don't terminate on success immediately — let the policy complete
        # the full place+release+retreat cycle for realistic behavior.
        terminated = False
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
        """Check if cube is in the target zone (pick_place) or stacked (stack)."""
        if self.task == "stack":
            return self._check_stack_success()

        if self._cube_body_id < 0 or self._target_site_id < 0:
            return False

        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.data.site_xpos[self._target_site_id]

        # XY distance
        dist_xy = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        # Height check: cube must be near target height (placed, not mid-air)
        height_ok = abs(cube_pos[2] - target_pos[2]) < 0.03

        return dist_xy < self.cfg.success_threshold and height_ok

    def _check_stack_success(self) -> bool:
        """Check if cube_A is stacked on top of cube_B."""
        if self._cube_body_id < 0 or self._cube_b_body_id < 0:
            return False

        cube_a_pos = self.data.xpos[self._cube_body_id]
        cube_b_pos = self.data.xpos[self._cube_b_body_id]

        # cube_A should be directly above cube_B
        dist_xy = np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2])
        # Expected height = cube_B_z + 2 * cube_size (both cube half-sizes)
        expected_z = cube_b_pos[2] + 2 * self.cfg.cube_size
        height_ok = abs(cube_a_pos[2] - expected_z) < 0.02

        return dist_xy < self.cfg.success_threshold and height_ok

    def get_cube_pos(self) -> np.ndarray:
        """Get current cube position (for scripted policies)."""
        if self._cube_body_id >= 0:
            return self.data.xpos[self._cube_body_id].copy()
        return np.zeros(3)

    def get_cube_b_pos(self) -> np.ndarray:
        """Get current cube_B position (for stacking policy)."""
        if self._cube_b_body_id >= 0:
            return self.data.xpos[self._cube_b_body_id].copy()
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
