"""
Modular manipulation environment for OPAB.

Thin orchestrator that composes:
  - Robot (from opab.robots) — gripper control, XML patches, IK params
  - Task (from opab.tasks) — object XML, success checks, ID caching
  - Placement (from opab.config) — per-(robot, task) positions/sizes
  - IK solver (from opab.env.ik_solver) — stateless DLS algorithm

Provides the same public API as the original monolithic PickPlaceEnv
so all existing scripts (record_all_tasks.py, etc.) continue working.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from opab.robots import get_robot
from opab.robots.base_robot import BaseRobot
from opab.tasks import get_task
from opab.tasks.base_task import BaseTask, TaskPlacement
from opab.placement import get_placement
from opab.env.ik_solver import solve_ik, orientation_error


class ManipulationEnv:
    """
    Unified manipulation environment (modular architecture).

    Accepts task-space actions: [dx, dy, dz, d_yaw, gripper] where:
      - dx, dy, dz: EE position deltas in meters
      - d_yaw: EE yaw rotation delta in radians (around world Z)
      - gripper: 0.0 = open, 1.0 = closed

    Works for all robots (Franka, UR5, Lite6, WidowX, SO-101)
    and all tasks (pick_place, push, stack, peg_insertion, drawer_open,
    turn_faucet, door_open).
    """

    def __init__(
        self,
        robot: str = "franka",
        image_size: tuple[int, int] = (128, 128),
        control_freq: float = 20.0,
        max_episode_steps: int = 300,
        seed: Optional[int] = None,
        kinematic_mode: bool = False,
        task: str = "pick_place",
    ):
        self.robot_name = robot
        self.task_name = task
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.kinematic_mode = kinematic_mode

        # Instantiate modular components
        self._robot: BaseRobot = get_robot(robot)
        self._task: BaseTask = get_task(task)
        self._placement: TaskPlacement = get_placement(robot, task)

        # For backwards compatibility — expose as self.cfg and self.task
        self.cfg = self._robot
        self.task = task

        # Physics = 500Hz, control = 20Hz → 25 substeps
        self.control_dt = 1.0 / control_freq

        # Load model
        self._load_model()

        # Physics substeps per control step
        self.n_substeps = max(1, int(self.control_dt / self.model.opt.timestep))

        # Cache IDs
        self._cache_ids()

        # Ensure framebuffer is large enough
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, image_size[0])
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, image_size[1])

        # Renderer
        self.renderer = mujoco.Renderer(self.model, *image_size)

        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    # ============================================================
    # Model Loading
    # ============================================================

    def _load_model(self):
        """Load the MuJoCo model: robot scene + task objects."""
        project_root = Path(__file__).resolve().parents[2]

        scene_path = project_root / self._robot.scene_path
        scene_dir = scene_path.parent

        with open(scene_path) as f:
            xml = f.read()

        # Phase 1: Pre-inline mods (include swaps — e.g., Lite6 gripper)
        xml = self._robot.pre_inline_xml(xml, scene_dir)

        # Inline includes
        xml = self._inline_includes(xml, scene_dir)

        # Phase 2: Post-inline mods (body injections — e.g., Franka grip_site, UR5 Robotiq)
        xml = self._robot.modify_xml(xml, scene_dir)

        # Robot z-offset
        xml = self._apply_z_offset(xml)

        # Override floor
        xml = self._override_floor(xml)

        # Inject table
        xml = self._inject_table(xml)

        # Inject camera
        xml = self._inject_camera(xml)

        # Inject task-specific assets
        task_assets = self._task.generate_asset_xml(self._placement)
        if task_assets:
            asset_close_idx = xml.rfind("</asset>")
            if asset_close_idx >= 0:
                xml = xml[:asset_close_idx] + task_assets + "  </asset>" + xml[asset_close_idx + len("</asset>"):]

        # Inject task objects
        objects_xml = self._task.generate_object_xml(self._placement)
        idx = xml.rfind("</worldbody>")
        xml = xml[:idx] + objects_xml + "  </worldbody>" + xml[idx + len("</worldbody>"):]

        # Write temp file (meshes resolve relative to scene_dir)
        tmp_path = scene_dir / "_opab_tmp_scene.xml"
        try:
            tmp_path.write_text(xml)
            self.model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self.data = mujoco.MjData(self.model)

        # Post-load tuning
        self._robot.post_load_tuning(self.model)

        # Physics standardization
        self.model.opt.timestep = 0.002
        self.model.opt.iterations = 100
        self.model.opt.cone = 0
        self.n_substeps = max(1, int(self.control_dt / self.model.opt.timestep))

    @staticmethod
    def _inline_includes(xml: str, base_dir: Path) -> str:
        """Inline <include file="..."/> tags so we can inject into robot bodies."""
        pattern = r'<include\s+file="([^"]+)"\s*/>'
        while re.search(pattern, xml):
            def _replace_include(m):
                inc_path = base_dir / m.group(1)
                inc_text = inc_path.read_text()
                inc_text = re.sub(r'<\?xml[^>]*\?>', '', inc_text)
                inc_text = re.sub(r'<mujoco[^>]*>', '', inc_text, count=1)
                inc_text = re.sub(r'</mujoco\s*>', '', inc_text, count=1)
                return inc_text
            xml = re.sub(pattern, _replace_include, xml, count=1)
        return xml

    def _apply_z_offset(self, xml: str) -> str:
        """Apply robot z-offset to root body position."""
        z_off = self._robot.robot_z_offset
        pattern = r'(<body\s+name="[^"]*"[^>]*pos=")([^"]*)"'
        match = re.search(pattern, xml)
        if match:
            pos_str = match.group(2)
            parts = pos_str.split()
            if len(parts) == 3:
                parts[2] = str(float(parts[2]) + z_off)
                new_pos = " ".join(parts)
                xml = xml[:match.start(2)] + new_pos + xml[match.end(2):]
        return xml

    def _override_floor(self, xml: str) -> str:
        """Override menagerie floor with our standard floor."""
        xml = re.sub(
            r'<geom[^>]*name="floor"[^/]*/>', '', xml
        )
        return xml

    def _inject_table(self, xml: str) -> str:
        """Inject table geometry before </worldbody>."""
        p = self._placement
        table_xml = f"""
    <body name="table" pos="{p.table_pos[0]} {p.table_pos[1]} {p.table_pos[2]}">
      <geom name="table_top" type="box"
            size="{p.table_half_size[0]} {p.table_half_size[1]} {p.table_half_size[2]}"
            rgba="0.4 0.35 0.3 1" contype="1" conaffinity="1"
            friction="0.8 0.005 0.0001"/>
      <geom name="table_floor" type="plane" size="2 2 0.01" pos="0 0 -{p.table_half_size[2]}"
            rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
    </body>
"""
        idx = xml.rfind("</worldbody>")
        xml = xml[:idx] + table_xml + xml[idx:]
        return xml

    def _inject_camera(self, xml: str) -> str:
        """Inject overhead camera before </worldbody>."""
        p = self._placement
        cam_xml = f"""
    <camera name="overhead_cam" pos="{p.cam_pos[0]} {p.cam_pos[1]} {p.cam_pos[2]}"
            euler="0 0 0" mode="fixed" fovy="45"/>
"""
        idx = xml.rfind("</worldbody>")
        xml = xml[:idx] + cam_xml + xml[idx:]
        return xml

    # ============================================================
    # ID Caching
    # ============================================================

    def _cache_ids(self):
        """Cache body/joint/site IDs for fast access."""
        # End-effector
        if self._robot.ee_site_name:
            self._ee_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self._robot.ee_site_name
            )
            self._use_site_for_ee = self._ee_site_id >= 0
        else:
            self._ee_site_id = -1
            self._use_site_for_ee = False

        if self._robot.ee_body_name:
            self._ee_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self._robot.ee_body_name
            )
        else:
            self._ee_body_id = -1

        # Arm joints
        self._arm_joint_ids = []
        self._arm_dof_ids = []
        self._arm_qpos_ids = []
        for name in self._robot.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found in model"
            self._arm_joint_ids.append(jid)
            self._arm_dof_ids.append(self.model.jnt_dofadr[jid])
            self._arm_qpos_ids.append(self.model.jnt_qposadr[jid])

        # Task-specific IDs
        self._task_ids = self._task.cache_ids(self.model)

        # Camera
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"
        )
        if self._cam_id < 0:
            for cam_name in ["overhead", "fixed"]:
                self._cam_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
                )
                if self._cam_id >= 0:
                    break

    # ============================================================
    # EE Helpers
    # ============================================================

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
    # IK
    # ============================================================

    def _ik_solve(self, target_pos: np.ndarray) -> np.ndarray:
        """Solve IK for target position with current orientation goal."""
        return solve_ik(
            model=self.model,
            data=self.data,
            target_pos=target_pos,
            target_mat=self._target_ee_mat,
            ee_site_id=self._ee_site_id,
            ee_body_id=self._ee_body_id,
            use_site_for_ee=self._use_site_for_ee,
            arm_joint_ids=self._arm_joint_ids,
            arm_dof_ids=self._arm_dof_ids,
            arm_qpos_ids=self._arm_qpos_ids,
            n_arm_joints=self._robot.n_arm_joints,
            ik_damping=self._robot.ik_damping,
            ik_max_iter=self._robot.ik_max_iter,
        )

    # ============================================================
    # Environment Interface
    # ============================================================

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization
        if hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.reset_to_nominal()
            self.domain_randomizer.randomize(self._rng)

        # Set robot to home pose
        if hasattr(self._robot, 'home_qpos'):
            for joint_name, qval in self._robot.home_qpos.items():
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = qval
                    aid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                    )
                    if aid >= 0:
                        self.data.ctrl[aid] = qval

        # Task-specific randomization
        self._task.randomize_reset(
            self.model, self.data, self._task_ids, self._placement, self._rng
        )

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._ee_target = self._get_ee_pos().copy()
        self._target_ee_mat = self._get_ee_mat()
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one control step.

        Args:
            action: (5,) array [dx, dy, dz, d_yaw, gripper]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (5,), f"Expected action shape (5,), got {action.shape}"

        ee_delta = np.clip(action[:3], -self._robot.action_scale, self._robot.action_scale)
        yaw_delta = np.clip(action[3], -0.1, 0.1)
        grip_cmd = np.clip(action[4], 0.0, 1.0)

        # Accumulate desired EE position
        self._ee_target = self._ee_target + ee_delta

        # Apply yaw rotation
        if abs(yaw_delta) > 1e-6:
            c, s = np.cos(yaw_delta), np.sin(yaw_delta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self._target_ee_mat = Rz @ self._target_ee_mat

        # IK
        target_qpos = self._ik_solve(self._ee_target)

        # Set arm actuator targets
        n_arm = self._robot.n_arm_joints
        for i in range(min(n_arm, self.model.nu)):
            self.data.ctrl[i] = target_qpos[i]
        # Zero velocity actuators
        n_ctrl = self._robot.effective_n_arm_actuators
        for i in range(n_arm, n_ctrl):
            self.data.ctrl[i] = 0.0

        # Gripper
        self._robot.set_gripper(self.model, self.data, grip_cmd)

        # Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        success = self._check_success()
        reward = float(success)
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        info = {"success": success, "step": self._step_count}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        """Extract observation from current state."""
        mujoco.mj_forward(self.model, self.data)

        if self._cam_id >= 0:
            self.renderer.update_scene(self.data, camera=self._cam_id)
        else:
            self.renderer.update_scene(self.data)
        image = self.renderer.render()

        qpos_arm = np.array([self.data.qpos[i] for i in self._arm_qpos_ids])
        ee_pos = self._get_ee_pos()
        ee_quat = self._get_ee_quat()
        gripper_pos = np.array([self._robot.get_gripper_pos(self.model, self.data)])

        return {
            "image": image,
            "proprioception": qpos_arm,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_pos": gripper_pos,
        }

    def _check_success(self) -> bool:
        """Delegate to task's success check."""
        return self._task.check_success(
            self.model, self.data, self._task_ids, self._placement
        )

    # ============================================================
    # Helper methods for scripted policies (backwards compatibility)
    # ============================================================

    def get_cube_pos(self) -> np.ndarray:
        """Get current cube position."""
        body_id = self._task_ids.get("cube_body", -1)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)

    def get_cube_b_pos(self) -> np.ndarray:
        """Get current cube_B position (stacking)."""
        body_id = self._task_ids.get("cube_b_body", -1)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)

    def get_target_pos(self) -> np.ndarray:
        """Get target zone position."""
        site_id = self._task_ids.get("target_site", -1)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)

    def get_peg_pos(self) -> np.ndarray:
        """Get current peg position."""
        body_id = self._task_ids.get("peg_body", -1)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)

    def get_hole_pos(self) -> np.ndarray:
        """Get hole bottom position."""
        site_id = self._task_ids.get("hole_bottom_site", -1)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)

    def get_faucet_handle_pos(self) -> np.ndarray:
        """Get faucet handle site position."""
        site_id = self._task_ids.get("faucet_handle_site", -1)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)

    def get_faucet_angle(self) -> float:
        """Get current faucet joint angle (radians)."""
        joint_id = self._task_ids.get("faucet_joint", -1)
        if joint_id >= 0:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

    def get_door_angle(self) -> float:
        """Get current door hinge angle (radians)."""
        joint_id = self._task_ids.get("door_hinge", -1)
        if joint_id >= 0:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

    def get_door_handle_pos(self) -> np.ndarray:
        """Get door handle site position."""
        site_id = self._task_ids.get("door_handle_site", -1)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)

    def _get_gripper_pos(self) -> float:
        """Backwards-compat: get normalized gripper position."""
        return self._robot.get_gripper_pos(self.model, self.data)

    def _count_finger_cube_contacts(self) -> int:
        """Count contacts between finger geoms and the cube."""
        cube_body_id = self._task_ids.get("cube_body", -1)
        if cube_body_id < 0:
            return 0
        count = 0
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        for c in range(self.data.ncon):
            g1 = self.data.contact[c].geom1
            g2 = self.data.contact[c].geom2
            if g1 == cube_geom_id or g2 == cube_geom_id:
                other = g2 if g1 == cube_geom_id else g1
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or ""
                if "table" not in other_name and "floor" not in other_name:
                    count += 1
        return count

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            del self.renderer


# Backwards-compatible alias
PickPlaceEnv = ManipulationEnv
