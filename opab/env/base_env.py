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
            self.action_scale = 0.01        # max EE delta per step (m) — standard 1cm/step
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
            # Franka: table-mounted at z=0.4 (like robosuite/Furniture-Bench)
            # Robot base bolted on table, objects on same surface.
            # 80cm depth x 120cm width (robosuite/Furniture-Bench standard).
            self.robot_z_offset = 0.4
            self.table_pos = np.array([0.35, 0.0, 0.2])
            self.table_half_size = np.array([0.40, 0.60, 0.2])
            self.cube_pos = np.array([0.45, 0.05, 0.42])
            self.cube_size = 0.02
            self.cube_mass = 0.05
            self.target_pos = np.array([0.45, -0.05, 0.405])
            self.cam_pos = np.array([0.25, 0.0, 1.3])
            self.cube_randomize_range = 0.0  # Disabled for v1 (fixed positions, BC standard)
            self.success_threshold = 0.025
            # Peg insertion parameters
            self.peg_radius = 0.010
            self.peg_half_length = 0.050
            self.hole_clearance = 0.002
            self.hole_depth = 0.045

            # Drawer task parameters
            self.drawer_pos = np.array([0.45, 0.10, 0.42])
            self.drawer_size = np.array([0.05, 0.04, 0.03])
            self.drawer_slide_range = 0.08  # 8cm pull
            # Faucet/dial task parameters
            self.faucet_pos = np.array([0.45, -0.10, 0.42])
            self.faucet_scale = 1.0
            self.faucet_target_angle = -1.2  # radians (~70 degrees)
            # Button press parameters
            self.button_pos = np.array([0.45, 0.10, 0.42])
            self.button_scale = 1.0
            # Door open parameters
            self.door_pos = np.array([0.50, 0.15, 0.42])
            self.door_scale = 1.0
            # Lever pull parameters
            self.lever_pos = np.array([0.45, 0.15, 0.42])
            self.lever_scale = 1.0
            # Sweep parameters
            self.sweep_obj_pos = np.array([0.45, 0.0, 0.42])
            self.sweep_target_pos = np.array([0.45, -0.25, 0.405])

        elif name == "ur5":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/universal_robots_ur5e/scene.xml")
            self.ee_body_name = "robotiq_base"
            self.ee_site_name = "robotiq_pinch"
            self.n_arm_joints = 6
            self.n_gripper_joints = 2  # Robotiq 2F-85 (left/right driver)
            self.arm_joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            self.gripper_joint_names = [
                "robotiq_right_driver_joint", "robotiq_left_driver_joint"
            ]
            self.gripper_actuator_idx = 6  # robotiq_fingers_actuator (after 6 arm actuators)
            self.action_scale = 0.01    # max EE delta per step (m) — standard 1cm/step
            self.ik_damping = 0.01
            self.ik_max_iter = 50
            self.gripper_open = 0.0     # ctrl=0 → open (Robotiq range 0-255)
            self.gripper_closed = 255.0 # ctrl=255 → fully closed
            # UR5e "ready" pose: arm reaches forward, gripper pointing perfectly down
            # Note: shoulder_lift + elbow + wrist_1 must = -pi/2 for vertical gripper
            self.home_qpos = {
                "shoulder_pan_joint": -3.14,
                "shoulder_lift_joint": -1.45,
                "elbow_joint": 0.9,
                "wrist_1_joint": -1.0208,
                "wrist_2_joint": -1.571,
                "wrist_3_joint": 0.0,
            }
            # UR5: table-mounted at z=0.4 (like Franka/robosuite).
            # 80cm depth x 120cm width (robot on the 120cm long side).
            self.robot_z_offset = 0.4
            self.table_pos = np.array([0.35, 0.0, 0.2])
            self.table_half_size = np.array([0.40, 0.60, 0.2])
            self.cube_pos = np.array([0.45, 0.15, 0.42])
            self.cube_size = 0.02
            self.cube_mass = 0.05
            self.target_pos = np.array([0.45, -0.15, 0.405])
            self.cam_pos = np.array([0.30, 0.0, 1.3])
            self.cube_randomize_range = 0.0  # Disabled for v1 (fixed positions, BC standard)
            self.success_threshold = 0.025
            # Peg insertion parameters
            self.peg_radius = 0.010
            self.peg_half_length = 0.050
            self.hole_clearance = 0.002
            self.hole_depth = 0.045

            # Drawer task parameters
            self.drawer_pos = np.array([0.40, 0.10, 0.42])
            self.drawer_size = np.array([0.05, 0.04, 0.03])
            self.drawer_slide_range = 0.08
            # Faucet/dial task parameters
            self.faucet_pos = np.array([0.40, -0.15, 0.42])
            self.faucet_scale = 1.0
            self.faucet_target_angle = -1.2
            # Button press parameters
            self.button_pos = np.array([0.40, 0.10, 0.42])
            self.button_scale = 1.0
            # Door open parameters
            self.door_pos = np.array([0.45, 0.15, 0.42])
            self.door_scale = 1.0
            # Lever pull parameters
            self.lever_pos = np.array([0.40, 0.15, 0.42])
            self.lever_scale = 1.0
            # Sweep parameters
            self.sweep_obj_pos = np.array([0.40, 0.0, 0.42])
            self.sweep_target_pos = np.array([0.40, -0.25, 0.405])

        elif name == "widowx":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/trossen_wx250s/scene.xml")
            self.ee_site_name = "grip_site"  # injected at finger tips
            self.ee_body_name = "wx250s/gripper_link"
            self.n_arm_joints = 6
            self.n_gripper_joints = 1
            self.arm_joint_names = [
                "waist", "shoulder", "elbow",
                "forearm_roll", "wrist_angle", "wrist_rotate"
            ]
            self.gripper_joint_names = ["left_finger"]
            self.gripper_actuator_idx = 6  # gripper actuator after 6 arm actuators
            self.action_scale = 0.01
            self.ik_damping = 0.02
            self.ik_max_iter = 50
            self.gripper_open = 0.037   # slide joint max (open)
            self.gripper_closed = 0.015  # slide joint min (closed)
            # Home pose from keyframe: arm folded, gripper pointing down
            self.home_qpos = {
                "waist": 0.0, "shoulder": -0.96, "elbow": 1.16,
                "forearm_roll": 0.0, "wrist_angle": -0.3, "wrist_rotate": 0.0,
                "left_finger": 0.037, "right_finger": -0.037,
            }
            # WidowX: table-mounted (Bridge V2 style). 76cm depth x 122cm width.
            # Robot on the 122cm long side, 20cm from back edge.
            self.robot_z_offset = 0.4   # inject into root body pos
            self.table_pos = np.array([0.18, 0.0, 0.2])
            self.table_half_size = np.array([0.38, 0.61, 0.2])
            self.cube_pos = np.array([0.22, 0.12, 0.418])
            self.cube_size = 0.018
            self.cube_mass = 0.03
            self.target_pos = np.array([0.22, -0.12, 0.405])
            self.cam_pos = np.array([0.15, 0.0, 1.2])
            self.cube_randomize_range = 0.0  # Disabled for v1 (fixed positions, BC standard)
            self.success_threshold = 0.0225
            # Peg insertion parameters
            self.peg_radius = 0.008
            self.peg_half_length = 0.030
            self.hole_clearance = 0.002
            self.hole_depth = 0.040

            # Drawer task parameters
            self.drawer_pos = np.array([0.28, 0.06, 0.42])
            self.drawer_size = np.array([0.04, 0.035, 0.025])
            self.drawer_slide_range = 0.06  # 6cm pull
            # Faucet/dial task parameters
            self.faucet_pos = np.array([0.22, -0.06, 0.42])
            self.faucet_scale = 0.9
            self.faucet_target_angle = -1.2
            # Button press parameters
            self.button_pos = np.array([0.22, 0.06, 0.065])
            self.button_scale = 0.75
            # Door open parameters
            self.door_pos = np.array([0.28, 0.08, 0.065])
            self.door_scale = 0.75
            # Lever pull parameters
            self.lever_pos = np.array([0.22, 0.08, 0.065])
            self.lever_scale = 0.75
            # Sweep parameters
            self.sweep_obj_pos = np.array([0.22, 0.0, 0.065])
            self.sweep_target_pos = np.array([0.22, -0.12, 0.055])

        elif name == "lite6":
            self.scene_path = kwargs.get("scene_path",
                "assets/mujoco_menagerie/ufactory_lite6/scene.xml")
            self.ee_site_name = "end_effector"
            self.ee_body_name = "gripper_body"
            self.n_arm_joints = 6
            self.n_gripper_joints = 1
            self.arm_joint_names = [
                "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6"
            ]
            self.gripper_joint_names = ["gripper_left_finger"]
            self.gripper_actuator_idx = 6  # motor actuator after 6 arm position actuators
            self.action_scale = 0.01
            self.ik_damping = 0.01
            self.ik_max_iter = 50
            self.gripper_open = -10.0   # motor ctrl=-10 -> open (push fingers apart)
            self.gripper_closed = 10.0   # motor ctrl=+10 -> close (push fingers together)
            # Home pose: arm reaching forward and down
            self.home_qpos = {
                "joint1": 0.0, "joint2": 0.0, "joint3": 1.57,
                "joint4": 0.0, "joint5": 1.57, "joint6": 0.0,
                "gripper_left_finger": 0.0, "gripper_right_finger": 0.0,
            }
            # Lite6: table-mounted at z=0.4 (like WidowX/Franka).
            # 50cm x 60cm table (proportional to 440mm reach).
            self.robot_z_offset = 0.4
            self.table_pos = np.array([0.25, 0.0, 0.2])
            self.table_half_size = np.array([0.30, 0.60, 0.2])
            self.cube_pos = np.array([0.30, 0.10, 0.418])
            self.cube_size = 0.018
            self.cube_mass = 0.04
            self.target_pos = np.array([0.30, -0.10, 0.405])
            self.cam_pos = np.array([0.20, 0.0, 1.2])
            self.cube_randomize_range = 0.0  # Disabled for v1 (fixed positions, BC standard)
            self.success_threshold = 0.0225
            # Peg insertion parameters
            self.peg_radius = 0.008
            self.peg_half_length = 0.030
            self.hole_clearance = 0.002
            self.hole_depth = 0.040

            # Drawer task parameters
            self.drawer_pos = np.array([0.20, 0.06, 0.42])
            self.drawer_size = np.array([0.035, 0.03, 0.02])
            self.drawer_slide_range = 0.05
            # Faucet/dial task parameters
            self.faucet_pos = np.array([0.30, -0.08, 0.42])
            self.faucet_scale = 1.0
            self.faucet_target_angle = -1.2
            # Button press parameters
            self.button_pos = np.array([0.30, 0.08, 0.41])
            self.button_scale = 0.85
            # Door open parameters
            self.door_pos = np.array([0.35, 0.12, 0.41])
            self.door_scale = 0.85
            # Lever pull parameters
            self.lever_pos = np.array([0.30, 0.12, 0.41])
            self.lever_scale = 0.85
            # Sweep parameters
            self.sweep_obj_pos = np.array([0.30, 0.0, 0.41])
            self.sweep_target_pos = np.array([0.30, -0.15, 0.41])

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
            self.action_scale = 0.01    # max EE delta per step (m) — standard 1cm/step
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
            # SO-101: table-mounted at z=0.2 (shorter table for small robot).
            # 60cm depth x 100cm width (robot on the 100cm long side).
            self.robot_z_offset = 0.2
            self.table_pos = np.array([0.25, 0.0, 0.1])
            self.table_half_size = np.array([0.30, 0.50, 0.1])
            self.cube_pos = np.array([0.15, 0.06, 0.213])
            self.cube_size = 0.012  # 2.4cm edge — must fit 25mm gripper aperture
            self.cube_mass = 0.03
            self.target_pos = np.array([0.15, -0.06, 0.205])
            self.cam_pos = np.array([0.10, 0.0, 0.7])
            self.cube_randomize_range = 0.0  # Disabled for v1 (fixed positions, BC standard)
            self.success_threshold = 0.01875
            # Peg insertion parameters
            self.peg_radius = 0.006
            self.peg_half_length = 0.050
            self.hole_clearance = 0.002
            self.hole_depth = 0.020

            # Drawer task parameters
            self.drawer_pos = np.array([0.23, 0.06, 0.42])
            self.drawer_size = np.array([0.035, 0.03, 0.02])
            self.drawer_slide_range = 0.05
            # Faucet/dial task parameters
            self.faucet_pos = np.array([0.18, -0.06, 0.22])
            self.faucet_scale = 0.7
            self.faucet_target_angle = -1.2
            # Button press parameters
            self.button_pos = np.array([0.18, 0.06, 0.021])
            self.button_scale = 0.60
            # Door open parameters
            self.door_pos = np.array([0.22, 0.08, 0.021])
            self.door_scale = 0.60
            # Lever pull parameters
            self.lever_pos = np.array([0.18, 0.08, 0.021])
            self.lever_scale = 0.60
            # Sweep parameters
            self.sweep_obj_pos = np.array([0.18, 0.0, 0.021])
            self.sweep_target_pos = np.array([0.18, -0.10, 0.015])
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
        image_size: tuple[int, int] = (128, 128),
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

        # Ensure framebuffer is large enough for requested image size
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, image_size[0])
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, image_size[1])

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


        # For Lite6: swap lite6.xml include for the gripper-equipped version
        # Must happen BEFORE _inline_includes() expands the include
        if self.cfg.name == "lite6":
            xml = xml.replace(
                '<include file="lite6.xml"/>',
                '<include file="lite6_gripper_narrow.xml"/>'
            )

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
                'kp="800" ctrlrange="0 0.04" />'
                '\n  <position name="finger_actuator2" joint="finger_joint2" '
                'kp="800" ctrlrange="0 0.04" />'
            )
            xml = xml.replace('</actuator>', finger_actuators + '\n</actuator>')

        # For UR5: inject Robotiq 2F-85 gripper at wrist_3_link
        if self.cfg.name == "ur5":
            xml = self._inject_robotiq_gripper(xml, scene_dir)

        # Raise robot base to table height (for robots where robot_z_offset is set)
        if hasattr(self.cfg, 'robot_z_offset') and self.cfg.robot_z_offset > 0:
            import re as _re
            # Find full root body opening tag (up to the closing >)
            body_pattern = r'(<body\s+name="(?:\w+/)?(?:base_link|link_base|link0|base)"[^>]*>)'
            match = _re.search(body_pattern, xml)
            if match:
                body_tag = match.group(1)
                z = self.cfg.robot_z_offset
                if 'pos=' in body_tag:
                    new_tag = _re.sub(r'pos="[^"]*"', f'pos="0 0 {z}"', body_tag)
                else:
                    # Insert pos before the closing >
                    new_tag = body_tag[:-1] + f' pos="0 0 {z}">'
                xml = xml.replace(body_tag, new_tag, 1)

        # For WidowX: inject grip_site at the midpoint between finger tips
        # gripper_link body -> 0.066m to finger base + 0.042m to fingertip midpoint
        if self.cfg.name == "widowx":
            inject_site = '<site name="grip_site" pos="0.108 0 0" size="0.005" rgba="0 1 0 0.3" group="4"/>'
            target_body = '<body name="gripper_link"'
            if target_body in xml:
                idx_b = xml.find(target_body)
                end_tag = xml.find('>', idx_b)
                xml = xml[:end_tag+1] + '\n          ' + inject_site + xml[end_tag+1:]


        # Override floor: replace checker material with flat gray
        # All menagerie scenes have: <geom name="floor" ... material="groundplane"/>
        xml = re.sub(
            r'(<geom\s+name="floor"[^/]*?)material="groundplane"([^/]*/\s*>)',
            r'\1rgba="0.5 0.5 0.5 1"\2',
            xml
        )

        # Build objects XML to inject
        cfg = self.cfg
        # Table mesh scaling: non-uniform to match desired table dimensions.
        # Original mesh: X=0.762m wide, Y=1.2192m deep, Z surface at 0.74m
        table_surface_z = cfg.table_pos[2] + cfg.table_half_size[2]
        scale_x = (cfg.table_half_size[0] * 2) / 0.762  # desired width / mesh width
        scale_y = (cfg.table_half_size[1] * 2) / 1.2192  # desired depth / mesh depth
        scale_z = table_surface_z / 0.74 if table_surface_z > 0.01 else 0.01

        # Inject table mesh assets into XML <asset> section
        project_root = Path(__file__).resolve().parents[2]
        table_asset_dir = project_root / "assets" / "table"
        table_assets = (
            f'  <mesh name="tabletop" file="{table_asset_dir}/tabletop.obj" scale="{scale_x} {scale_y} {scale_z}" />\n'
            f'  <mesh name="tablelegs" file="{table_asset_dir}/tablelegs.obj" scale="{scale_x} {scale_y} {scale_z}" />\n'
            f'  <texture name="table_tex" type="2d" file="{table_asset_dir}/small_meta_table_diffuse.png" />\n'
            f'  <material name="table_wood" texture="table_tex" />\n'
        )
        # Insert before the LAST </asset> (inlined XML may have multiple)
        if "</asset>" in xml:
            last_idx = xml.rfind("</asset>")
            xml = xml[:last_idx] + table_assets + xml[last_idx:]
        else:
            xml = xml.replace("</mujoco>", "  <asset>\n" + table_assets + "  </asset>\n</mujoco>")

        objects = f"""
    <!-- OPAB: Injected task objects -->
    <body name="table" pos="{cfg.table_pos[0]} {cfg.table_pos[1]} 0">
      <geom name="table_visual_top" type="mesh" mesh="tabletop"
            material="table_wood" contype="0" conaffinity="0" />
      <geom name="table_visual_legs" type="mesh" mesh="tablelegs"
            material="table_wood" contype="0" conaffinity="0" />
      <geom name="table_surface" type="box"
            size="{cfg.table_half_size[0]} {cfg.table_half_size[1]} 0.01"
            pos="0 0 {table_surface_z}"
            rgba="0 0 0 0" contype="1" conaffinity="1"
            friction="1.0 0.005 0.0001" />
    </body>

    <site name="target_zone" pos="{cfg.target_pos[0]} {cfg.target_pos[1]} {cfg.target_pos[2]}"
          size="{cfg.cube_size * 2} 0.002" rgba="0.1 0.8 0.1 0.5" type="cylinder" />

    <camera name="overhead_cam" pos="{cfg.cam_pos[0]} {cfg.cam_pos[1]} {cfg.cam_pos[2]}"
            xyaxes="0 -1 0 0.966 0 0.259" fovy="60" />
    <camera name="topdown_cam" pos="{cfg.cam_pos[0]} {cfg.cam_pos[1]} {cfg.cam_pos[2]}"
            xyaxes="1 0 0 0 1 0" fovy="60" />
    <camera name="angled_cam" pos="{cfg.cam_pos[0] + 0.6} {cfg.cam_pos[1] - 0.5} {cfg.cam_pos[2] - 0.4}"
            xyaxes="0.64 0.77 0 -0.41 0.35 0.84" fovy="45" />
"""

        # Task-specific objects
        if self.task == "reach":
            # Reach: just a target sphere, no cube needed
            objects += f"""
    <site name="reach_target" pos="{cfg.target_pos[0]} {cfg.target_pos[1]} {cfg.target_pos[2] + 0.05}"
          size="0.02" rgba="0.9 0.2 0.2 0.8" type="sphere" />
"""

        elif self.task in ("pick_place", "push", "stack"):
            # These tasks need a cube
            objects += f"""
    <body name="cube" pos="{cfg.cube_pos[0]} {cfg.cube_pos[1]} {cfg.cube_pos[2]}">
      <freejoint name="cube_joint" />
      <geom name="cube_geom" type="box" size="{cfg.cube_size} {cfg.cube_size} {cfg.cube_size}"
            mass="{cfg.cube_mass}" rgba="0.9 0.1 0.1 1" condim="4"
            friction="1.0 0.005 0.0001" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" />
    </body>
"""

        elif self.task == "peg_insertion":
            # Peg (free cylinder on table) + hole (physical socket with walls)
            # Per-robot scaled: peg_radius, peg_half_length, hole_clearance, hole_depth
            table_top_z = cfg.table_pos[2] + cfg.table_half_size[2]
            peg_r = cfg.peg_radius
            peg_hl = cfg.peg_half_length
            clearance = cfg.hole_clearance
            hole_depth = cfg.hole_depth
            hole_inner_r = peg_r + clearance
            hole_outer_r = hole_inner_r + 0.005  # 5mm wall thickness

            # Peg spawns upright on table (center at table + half_length)
            peg_z = table_top_z + peg_hl
            peg_pos = np.array([cfg.cube_pos[0], cfg.cube_pos[1], peg_z])

            # Hole elevated above table so downward insertion is meaningful
            hole_z = table_top_z + hole_depth
            hole_pos = np.array([cfg.target_pos[0], cfg.target_pos[1], hole_z])

            objects += f"""
    <body name="peg" pos="{peg_pos[0]} {peg_pos[1]} {peg_pos[2]}">
      <freejoint name="peg_joint" />
      <geom name="peg_geom" type="cylinder" size="{peg_r} {peg_hl}"
            mass="0.02" rgba="0.2 0.6 0.9 1" condim="6"
            friction="2.0 0.1 0.01" contype="1" conaffinity="1" />
    </body>

    <body name="hole_body" pos="{hole_pos[0]} {hole_pos[1]} {hole_pos[2]}">
      <!-- Hollow cylinder: 32 thin box segments forming a smooth ring -->
      <geom name="hole_seg_0" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 1.000000} {((hole_inner_r + hole_outer_r)/2) * 0.000000} 0"
            euler="0 0 0.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_1" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.980785} {((hole_inner_r + hole_outer_r)/2) * 0.195090} 0"
            euler="0 0 11.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_2" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.923880} {((hole_inner_r + hole_outer_r)/2) * 0.382683} 0"
            euler="0 0 22.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_3" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.831470} {((hole_inner_r + hole_outer_r)/2) * 0.555570} 0"
            euler="0 0 33.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_4" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.707107} {((hole_inner_r + hole_outer_r)/2) * 0.707107} 0"
            euler="0 0 45.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_5" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.555570} {((hole_inner_r + hole_outer_r)/2) * 0.831470} 0"
            euler="0 0 56.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_6" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.382683} {((hole_inner_r + hole_outer_r)/2) * 0.923880} 0"
            euler="0 0 67.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_7" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.195090} {((hole_inner_r + hole_outer_r)/2) * 0.980785} 0"
            euler="0 0 78.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_8" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.000000} {((hole_inner_r + hole_outer_r)/2) * 1.000000} 0"
            euler="0 0 90.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_9" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.195090} {((hole_inner_r + hole_outer_r)/2) * 0.980785} 0"
            euler="0 0 101.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_10" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.382683} {((hole_inner_r + hole_outer_r)/2) * 0.923880} 0"
            euler="0 0 112.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_11" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.555570} {((hole_inner_r + hole_outer_r)/2) * 0.831470} 0"
            euler="0 0 123.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_12" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.707107} {((hole_inner_r + hole_outer_r)/2) * 0.707107} 0"
            euler="0 0 135.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_13" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.831470} {((hole_inner_r + hole_outer_r)/2) * 0.555570} 0"
            euler="0 0 146.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_14" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.923880} {((hole_inner_r + hole_outer_r)/2) * 0.382683} 0"
            euler="0 0 157.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_15" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.980785} {((hole_inner_r + hole_outer_r)/2) * 0.195090} 0"
            euler="0 0 168.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_16" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -1.000000} {((hole_inner_r + hole_outer_r)/2) * 0.000000} 0"
            euler="0 0 180.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_17" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.980785} {((hole_inner_r + hole_outer_r)/2) * -0.195090} 0"
            euler="0 0 191.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_18" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.923880} {((hole_inner_r + hole_outer_r)/2) * -0.382683} 0"
            euler="0 0 202.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_19" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.831470} {((hole_inner_r + hole_outer_r)/2) * -0.555570} 0"
            euler="0 0 213.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_20" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.707107} {((hole_inner_r + hole_outer_r)/2) * -0.707107} 0"
            euler="0 0 225.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_21" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.555570} {((hole_inner_r + hole_outer_r)/2) * -0.831470} 0"
            euler="0 0 236.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_22" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.382683} {((hole_inner_r + hole_outer_r)/2) * -0.923880} 0"
            euler="0 0 247.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_23" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.195090} {((hole_inner_r + hole_outer_r)/2) * -0.980785} 0"
            euler="0 0 258.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_24" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * -0.000000} {((hole_inner_r + hole_outer_r)/2) * -1.000000} 0"
            euler="0 0 270.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_25" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.195090} {((hole_inner_r + hole_outer_r)/2) * -0.980785} 0"
            euler="0 0 281.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_26" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.382683} {((hole_inner_r + hole_outer_r)/2) * -0.923880} 0"
            euler="0 0 292.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_27" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.555570} {((hole_inner_r + hole_outer_r)/2) * -0.831470} 0"
            euler="0 0 303.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_28" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.707107} {((hole_inner_r + hole_outer_r)/2) * -0.707107} 0"
            euler="0 0 315.0"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_29" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.831470} {((hole_inner_r + hole_outer_r)/2) * -0.555570} 0"
            euler="0 0 326.25"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_30" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.923880} {((hole_inner_r + hole_outer_r)/2) * -0.382683} 0"
            euler="0 0 337.5"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <geom name="hole_seg_31" type="box"
            pos="{((hole_inner_r + hole_outer_r)/2) * 0.980785} {((hole_inner_r + hole_outer_r)/2) * -0.195090} 0"
            euler="0 0 348.75"
            size="{((hole_inner_r + hole_outer_r)/2) * 0.098491} {(hole_outer_r - hole_inner_r) / 2} {hole_depth}"
            rgba="0.4 0.4 0.4 1" contype="1" conaffinity="1" />
      <!-- Visual inner cylinder (semi-transparent) -->
      <geom name="hole_void" type="cylinder" size="{hole_inner_r} {hole_depth + 0.001}"
            rgba="0.1 0.1 0.1 0.2" contype="0" conaffinity="0" />
      <!-- Success site at bottom -->
      <site name="hole_bottom" pos="0 0 -{hole_depth}" size="0.005" rgba="0 1 0 0.5" />
      <!-- Floor -->
      <geom name="hole_floor" type="cylinder" size="{hole_inner_r} 0.002"
            pos="0 0 -{hole_depth}"
            rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1" />
    </body>
"""


        elif self.task == "drawer_open":
            # ManiSkill/RoboCasa-style cabinet with sliding drawer
            # Cabinet housing is static; inner drawer slides on prismatic joint
            table_top_z = cfg.table_pos[2] + cfg.table_half_size[2]
            
            # Cabinet dimensions (scaled per robot)
            cab_w = cfg.drawer_size[0]   # half-width (X)
            cab_d = cfg.drawer_size[1]   # half-depth (Y) 
            cab_h = cfg.drawer_size[2]   # half-height (Z)
            wall = 0.008                 # wall thickness
            slide_range = cfg.drawer_slide_range
            
            # Position: cabinet sits on table, open face toward robot (-Y)
            cab_pos = cfg.drawer_pos.copy()
            cab_pos[2] = table_top_z + cab_h
            
            # Inner drawer dimensions
            inner_w = cab_w - wall - 0.001
            inner_d = cab_d - wall - 0.001
            inner_h = cab_h - wall - 0.001
            
            # Handle: bar handle on front panel, 4cm proud
            handle_offset = 0.035
            handle_half_len = cab_w * 0.6

            objects += f"""
    <!-- Cabinet housing (static) -->
    <body name="cabinet" pos="{cab_pos[0]} {cab_pos[1]} {cab_pos[2]}">
      <!-- Top -->
      <geom name="cab_top" type="box" size="{cab_w} {cab_d} {wall/2}"
            pos="0 0 {cab_h}" rgba="0.55 0.35 0.2 1" contype="1" conaffinity="1" />
      <!-- Bottom -->
      <geom name="cab_bottom" type="box" size="{cab_w} {cab_d} {wall/2}"
            pos="0 0 -{cab_h}" rgba="0.55 0.35 0.2 1" contype="1" conaffinity="1" />
      <!-- Back wall -->
      <geom name="cab_back" type="box" size="{cab_w} {wall/2} {cab_h - wall}"
            pos="0 {cab_d} 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />
      <!-- Left wall -->
      <geom name="cab_left" type="box" size="{wall/2} {cab_d} {cab_h - wall}"
            pos="-{cab_w} 0 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />
      <!-- Right wall -->
      <geom name="cab_right" type="box" size="{wall/2} {cab_d} {cab_h - wall}"
            pos="{cab_w} 0 0" rgba="0.5 0.3 0.15 1" contype="1" conaffinity="1" />

      <!-- Sliding drawer body -->
      <body name="drawer_body" pos="0 0 0">
        <joint name="drawer_joint" type="slide" axis="0 -1 0"
               range="0 {slide_range}" damping="0.5" frictionloss="0.1" />
        <!-- Inner bottom -->
        <geom name="drawer_bottom" type="box" size="{inner_w} {inner_d} {wall/2}"
              pos="0 0 -{cab_h - wall}" mass="0.1" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner back -->
        <geom name="drawer_inner_back" type="box" size="{inner_w} {wall/2} {inner_h}"
              pos="0 {inner_d} 0" mass="0.05" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner left -->
        <geom name="drawer_inner_left" type="box" size="{wall/2} {inner_d} {inner_h}"
              pos="-{inner_w} 0 0" mass="0.03" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Inner right -->
        <geom name="drawer_inner_right" type="box" size="{wall/2} {inner_d} {inner_h}"
              pos="{inner_w} 0 0" mass="0.03" rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" />
        <!-- Front panel (face of drawer) -->
        <geom name="drawer_front" type="box" size="{cab_w - 0.001} {wall/2} {cab_h - 0.001}"
              pos="0 -{cab_d} 0" mass="0.08" rgba="0.6 0.4 0.25 1"
              contype="1" conaffinity="1" />
        <!-- Bar handle (horizontal, centered on front panel) -->
        <geom name="handle_bar" type="capsule" size="0.006"
              fromto="-{handle_half_len} -{cab_d + handle_offset} 0 {handle_half_len} -{cab_d + handle_offset} 0"
              mass="0.02" rgba="0.75 0.75 0.75 1" condim="6"
              friction="2.0 0.1 0.01" contype="1" conaffinity="1" />
        <!-- Handle connectors -->
        <geom name="handle_conn_l" type="cylinder" size="0.005 {handle_offset/2}"
              pos="-{handle_half_len} -{cab_d + handle_offset/2} 0" euler="1.5708 0 0"
              rgba="0.75 0.75 0.75 1" contype="1" conaffinity="1" />
        <geom name="handle_conn_r" type="cylinder" size="0.005 {handle_offset/2}"
              pos="{handle_half_len} -{cab_d + handle_offset/2} 0" euler="1.5708 0 0"
              rgba="0.75 0.75 0.75 1" contype="1" conaffinity="1" />
        <site name="drawer_handle_site" pos="0 -{cab_d + handle_offset} 0" size="0.005" rgba="1 1 0 0.5" />
      </body>
    </body>
"""

        elif self.task == "turn_faucet":
            # Turn faucet: modern single-lever gooseneck faucet (primitives)
            # Design: chrome column + gooseneck spout + side-mounted lever handle
            # Handle points -Y at rest (perpendicular to spout in +X), rotates around Z
            fpos = cfg.faucet_pos
            s = cfg.faucet_scale  # uniform scale factor (1.0 = 13.6cm tall)

            # Inject faucet materials into <asset> section
            faucet_materials = """
    <material name="faucet_chrome" rgba="0.75 0.78 0.82 1"
              specular="0.9" shininess="0.95" reflectance="0.5" />
    <material name="faucet_dark" rgba="0.30 0.32 0.35 1"
              specular="0.8" shininess="0.9" reflectance="0.4" />
    <material name="faucet_highlight" rgba="0.85 0.87 0.90 1"
              specular="0.95" shininess="0.98" reflectance="0.6" />
"""
            asset_close_idx = xml.rfind("</asset>")
            if asset_close_idx >= 0:
                xml = xml[:asset_close_idx] + faucet_materials + "  </asset>" + xml[asset_close_idx + len("</asset>"):]

            objects += f"""
    <body name="faucet_base" pos="{fpos[0]} {fpos[1]} {fpos[2]}">
      <!-- Base mounting ring -->
      <geom name="faucet_base_ring" type="cylinder" size="{0.025*s} {0.005*s}"
            pos="0 0 0" material="faucet_dark" mass="0.5"
            contype="1" conaffinity="1" />
      <!-- Lower column (wider) -->
      <geom name="faucet_col_lo" type="cylinder" size="{0.018*s} {0.025*s}"
            pos="0 0 {0.030*s}" material="faucet_chrome" mass="0.3"
            contype="1" conaffinity="1" />
      <!-- Upper column (narrower, tapered look) -->
      <geom name="faucet_col_hi" type="cylinder" size="{0.014*s} {0.030*s}"
            pos="0 0 {0.085*s}" material="faucet_chrome" mass="0.3"
            contype="1" conaffinity="1" />
      <!-- Neck dome (transition sphere) -->
      <geom name="faucet_neck" type="sphere" size="{0.016*s}"
            pos="0 0 {0.115*s}" material="faucet_chrome" mass="0.1"
            contype="1" conaffinity="1" />
      <!-- Gooseneck spout (arcs forward in +X, then curves down) -->
      <geom name="faucet_spout1" type="capsule" size="{0.008*s}"
            fromto="0 0 {0.115*s}  {0.020*s} 0 {0.125*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout2" type="capsule" size="{0.008*s}"
            fromto="{0.020*s} 0 {0.125*s}  {0.045*s} 0 {0.128*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout3" type="capsule" size="{0.007*s}"
            fromto="{0.045*s} 0 {0.128*s}  {0.065*s} 0 {0.120*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <geom name="faucet_spout4" type="capsule" size="{0.006*s}"
            fromto="{0.065*s} 0 {0.120*s}  {0.072*s} 0 {0.105*s}"
            material="faucet_chrome" contype="0" conaffinity="0" />
      <!-- Aerator tip -->
      <geom name="faucet_tip" type="cylinder" size="{0.007*s} {0.003*s}"
            pos="{0.072*s} 0 {0.102*s}" material="faucet_dark"
            contype="0" conaffinity="0" />
      <!-- Handle lever (articulated, points -Y at rest = perpendicular to spout) -->
      <body name="faucet_switch" pos="0 0 {0.120*s}">
        <joint name="faucet_joint" type="hinge" axis="0 0 1"
               range="-1.5708 1.5708" damping="0.15" frictionloss="0.03" />
        <!-- Handle base hub -->
        <geom name="faucet_hub" type="cylinder" size="{0.012*s} {0.008*s}"
              material="faucet_dark" mass="0.05"
              contype="1" conaffinity="1" />
        <!-- Lever arm (points -Y, perpendicular to spout) -->
        <geom name="faucet_lever" type="capsule" size="{0.006*s}"
              fromto="0 0 {0.005*s}  0 {-0.060*s} {0.015*s}"
              material="faucet_highlight" mass="0.03"
              contype="1" conaffinity="1" friction="2.0 0.1 0.01" />
        <!-- Grip ball at lever tip -->
        <geom name="faucet_grip" type="sphere" size="{0.009*s}"
              pos="0 {-0.065*s} {0.016*s}" material="faucet_highlight" mass="0.02"
              contype="1" conaffinity="1" friction="2.0 0.1 0.01" />
        <!-- Target site at lever tip -->
        <site name="faucet_handle_site" pos="0 {-0.065*s} {0.016*s}"
              size="0.004" rgba="0 1 0 0.3" />
      </body>
    </body>
"""

        elif self.task == "button_press":
            # Button press: housing box with cylindrical plunger (topdown press)
            bpos = cfg.button_pos
            bs = cfg.button_scale

            objects += f"""
    <body name="button_housing" pos="{bpos[0]} {bpos[1]} {bpos[2]}">
      <!-- Housing box -->
      <geom name="btn_base" type="box" size="{0.030*bs} {0.030*bs} {0.020*bs}"
            rgba="0.30 0.30 0.35 1" mass="0.5"
            contype="1" conaffinity="1"/>
      <!-- Button plunger (slides down on press) -->
      <body name="button_plunger" pos="0 0 {0.020*bs}">
        <joint name="button_joint" type="slide" axis="0 0 -1"
               range="0 {0.030*bs}" damping="5" stiffness="50"/>
        <geom name="btn_cap" type="cylinder" size="{0.015*bs} {0.006*bs}"
              rgba="0.90 0.15 0.15 1" mass="0.02"
              contype="1" conaffinity="1"/>
        <site name="btn_press_site" pos="0 0 0" size="0.002"/>
      </body>
      <!-- Target site (fully pressed pos) -->
      <site name="btn_target" pos="0 0 {-0.005*bs}" size="0.002"
            rgba="0 1 0 0.3"/>
    </body>
"""

        elif self.task == "door_open":
            # Door open: frame with hinged panel + handle
            dpos = cfg.door_pos
            ds = cfg.door_scale

            objects += f"""
    <body name="door_frame" pos="{dpos[0]} {dpos[1]} {dpos[2]}">
      <!-- Frame post (hinge side) -->
      <geom name="door_frame_post" type="box" size="{0.010*ds} {0.010*ds} {0.080*ds}"
            pos="{-0.080*ds} 0 {0.080*ds}" rgba="0.40 0.35 0.30 1"
            contype="1" conaffinity="1"/>
      <!-- Door panel (rotates around Z hinge) -->
      <body name="door_panel" pos="{-0.080*ds} 0 {0.080*ds}">
        <joint name="door_hinge" type="hinge" axis="0 0 1"
               range="0 1.57" damping="2" stiffness="0"/>
        <geom name="door_panel_geom" type="box"
              size="{0.075*ds} {0.005*ds} {0.075*ds}"
              pos="{0.075*ds} 0 0" rgba="0.85 0.80 0.70 1"
              contype="1" conaffinity="1"/>
        <!-- Door handle -->
        <body name="door_handle_body" pos="{0.120*ds} {0.015*ds} 0">
          <geom name="door_handle_geom" type="capsule" size="{0.008*ds}"
                fromto="0 0 {-0.020*ds} 0 0 {0.020*ds}"
                rgba="0.70 0.70 0.75 1"
                contype="1" conaffinity="1"/>
          <site name="door_handle_site" pos="0 0 0" size="0.002"/>
        </body>
      </body>
    </body>
"""

        elif self.task == "lever_pull":
            # Lever pull: base with hinged lever arm (horizontal to vertical)
            lpos = cfg.lever_pos
            ls = cfg.lever_scale

            objects += f"""
    <body name="lever_base" pos="{lpos[0]} {lpos[1]} {lpos[2]}">
      <!-- Base pedestal -->
      <geom name="lever_base_geom" type="box" size="{0.025*ls} {0.025*ls} {0.040*ls}"
            pos="0 0 {0.040*ls}" rgba="0.25 0.25 0.28 1" mass="0.5"
            contype="1" conaffinity="1"/>
      <!-- Lever arm (rotates around X-axis, starts horizontal) -->
      <body name="lever_arm" pos="0 0 {0.080*ls}">
        <joint name="lever_hinge" type="hinge" axis="1 0 0"
               range="0 1.57" damping="0.3" armature="0.005" stiffness="0"/>
        <!-- Arm capsule pointing in +Y (starts horizontal) -->
        <geom name="lever_arm_geom" type="capsule" size="{0.010*ls}"
              fromto="0 0 0 0 {0.120*ls} 0"
              rgba="0.60 0.60 0.65 1" mass="0.05"
              contype="1" conaffinity="1"/>
        <!-- Tip sphere -->
        <geom name="lever_tip_geom" type="sphere" size="{0.015*ls}"
              pos="0 {0.120*ls} 0" rgba="1.0 0.60 0.0 1" mass="0.02"
              contype="1" conaffinity="1"/>
        <site name="lever_tip_site" pos="0 {0.120*ls} 0" size="0.002"/>
      </body>
    </body>
"""

        elif self.task == "sweep":
            # Sweep: free-body puck + target zone
            spos = cfg.sweep_obj_pos
            stgt = cfg.sweep_target_pos

            objects += f"""
    <body name="sweep_obj" pos="{spos[0]} {spos[1]} {spos[2]}">
      <freejoint name="sweep_obj_joint"/>
      <geom name="sweep_obj_geom" type="cylinder" size="0.020 0.012"
            mass="0.05" rgba="0.20 0.50 0.90 1"
            friction="0.5 0.005 0.0001" solref="0.01 1"
            contype="1" conaffinity="1"/>
      <site name="sweep_obj_site" pos="0 0 0" size="0.002"/>
    </body>

    <site name="sweep_target" pos="{stgt[0]} {stgt[1]} {stgt[2]}"
          size="0.03 0.03 0.001" type="box" rgba="0.2 0.9 0.2 0.4"/>
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
            friction="1.0 0.005 0.0001" contype="1" conaffinity="1"
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

        # WidowX: boost forcerange for arm actuators (Dynamixel tracking)
        if self.robot_name == "widowx":
            for i in range(self.cfg.gripper_actuator_idx):
                self.model.actuator_forcerange[i] = [-50.0, 50.0]
            self.model.actuator_forcerange[self.cfg.gripper_actuator_idx] = [-5.0, 5.0]

        # --- Post-load physics standardization ---
        # Ensure consistent timestep, solver iterations, and friction cone
        # across ALL robots (menagerie XMLs may differ).
        self.model.opt.timestep = 0.002       # 500Hz physics (standard)
        self.model.opt.iterations = 100       # Newton solver iterations
        self.model.opt.cone = 0               # Pyramidal friction cone
        # Recompute n_substeps after timestep override
        self.n_substeps = max(1, int(self.control_dt / self.model.opt.timestep))

    def _inject_robotiq_gripper(self, xml: str, scene_dir) -> str:
        """Inject Robotiq 2F-85 gripper body into UR5e wrist_3_link."""
        from pathlib import Path
        import os

        project_root = Path(__file__).resolve().parents[2]
        robotiq_mesh_dir = project_root / "assets" / "mujoco_menagerie" / "robotiq_2f85" / "assets"

        # MuJoCo resolves mesh paths relative to the 'meshdir' set in <compiler>.
        # UR5e has meshdir="assets", so paths are relative to <scene_dir>/assets/
        ur5_mesh_base = scene_dir / "assets"
        rel_mesh = os.path.relpath(robotiq_mesh_dir, ur5_mesh_base)

        # 1. Inject Robotiq assets before closing </asset> (or add <asset> section)
        robotiq_assets = f"""
    <!-- Robotiq 2F-85 assets -->
    <material name="robotiq_metal" rgba="0.58 0.58 0.58 1"/>
    <material name="robotiq_silicone" rgba="0.1882 0.1882 0.1882 1"/>
    <material name="robotiq_gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="robotiq_black" rgba="0.149 0.149 0.149 1"/>
    <mesh name="robotiq_base_mount" file="{rel_mesh}/base_mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_base" file="{rel_mesh}/base.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_driver" file="{rel_mesh}/driver.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_coupler" file="{rel_mesh}/coupler.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_follower" file="{rel_mesh}/follower.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_pad" file="{rel_mesh}/pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_silicone_pad" file="{rel_mesh}/silicone_pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="robotiq_spring_link" file="{rel_mesh}/spring_link.stl" scale="0.001 0.001 0.001"/>
"""
        # Find last </asset> and inject before it
        asset_close_idx = xml.rfind("</asset>")
        if asset_close_idx >= 0:
            xml = xml[:asset_close_idx] + robotiq_assets + "  </asset>" + xml[asset_close_idx + len("</asset>"):]
        else:
            # No asset section — inject one
            worldbody_idx = xml.find("<worldbody")
            xml = xml[:worldbody_idx] + f"<asset>{robotiq_assets}</asset>\n" + xml[worldbody_idx:]

        # 2. Inject gripper body under wrist_3_link (after attachment_site)
        robotiq_body = """
                  <!-- Robotiq 2F-85 gripper mounted at attachment_site -->
                  <body name="robotiq_base_mount" pos="0 0.107 0" quat="-0.707107 0.707107 0 0">
                    <geom type="mesh" mesh="robotiq_base_mount" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                    <geom type="mesh" mesh="robotiq_base_mount" group="3"/>
                    <body name="robotiq_base" pos="0 0 0.0038" quat="1 0 0 -1">
                      <inertial mass="0.777441" pos="0 -2.70394e-05 0.0354675" quat="1 -0.00152849 0 0"
                          diaginertia="0.000260285 0.000225381 0.000152708"/>
                      <geom type="mesh" mesh="robotiq_base" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                      <geom type="mesh" mesh="robotiq_base" group="3"/>
                      <site name="robotiq_pinch" pos="0 0 0.145" type="sphere" group="5" rgba="0.9 0.9 0.9 1" size="0.005"/>
                      <!-- Right-hand side 4-bar linkage -->
                      <body name="robotiq_right_driver" pos="0 0.0306011 0.054904">
                        <inertial mass="0.00899563" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                            diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>
                        <joint name="robotiq_right_driver_joint" axis="1 0 0" range="0 0.8" armature="0.005" damping="0.1"
                            solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                        <geom type="mesh" mesh="robotiq_driver" material="robotiq_gray" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_driver" group="3"/>
                        <body name="robotiq_right_coupler" pos="0 0.0315 -0.0041">
                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                              diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>
                          <joint name="robotiq_right_coupler_joint" axis="1 0 0" range="-1.57 0" armature="0.001"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_coupler" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_coupler" group="3"/>
                        </body>
                      </body>
                      <body name="robotiq_right_spring_link" pos="0 0.0132 0.0609">
                        <inertial mass="0.0221642" pos="0 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                            diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>
                        <joint name="robotiq_right_spring_link_joint" axis="1 0 0" range="-0.29670597283 0.8"
                            armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
                        <geom type="mesh" mesh="robotiq_spring_link" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_spring_link" group="3"/>
                        <body name="robotiq_right_follower" pos="0 0.055 0.0375">
                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                              diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>
                          <joint name="robotiq_right_follower_joint" axis="1 0 0" range="-0.872664 0.872664"
                              armature="0.001" pos="0 -0.018 0.0065"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_follower" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_follower" group="3"/>
                          <body name="robotiq_right_pad" pos="0 -0.0189 0.01352">
                            <geom name="robotiq_right_pad1" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375"
                                mass="0" friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1" group="3"/>
                            <geom name="robotiq_right_pad2" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375"
                                mass="0" friction="0.6" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1" group="3"/>
                            <inertial mass="0.0035" pos="0 -0.0025 0.0185" quat="0.707107 0 0 0.707107"
                                diaginertia="4.73958e-07 3.64583e-07 1.23958e-07"/>
                            <geom type="mesh" mesh="robotiq_pad" contype="0" conaffinity="0" group="2"/>
                            <geom type="mesh" mesh="robotiq_silicone_pad" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          </body>
                        </body>
                      </body>
                      <!-- Left-hand side 4-bar linkage -->
                      <body name="robotiq_left_driver" pos="0 -0.0306011 0.054904" quat="0 0 0 1">
                        <inertial mass="0.00899563" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0"
                            diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>
                        <joint name="robotiq_left_driver_joint" axis="1 0 0" range="0 0.8" armature="0.005" damping="0.1"
                            solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                        <geom type="mesh" mesh="robotiq_driver" material="robotiq_gray" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_driver" group="3"/>
                        <body name="robotiq_left_coupler" pos="0 0.0315 -0.0041">
                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636"
                              diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>
                          <joint name="robotiq_left_coupler_joint" axis="1 0 0" range="-1.57 0" armature="0.001"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_coupler" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_coupler" group="3"/>
                        </body>
                      </body>
                      <body name="robotiq_left_spring_link" pos="0 -0.0132 0.0609" quat="0 0 0 1">
                        <inertial mass="0.0221642" pos="0 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403"
                            diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>
                        <joint name="robotiq_left_spring_link_joint" axis="1 0 0" range="-0.29670597283 0.8"
                            armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
                        <geom type="mesh" mesh="robotiq_spring_link" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                        <geom type="mesh" mesh="robotiq_spring_link" group="3"/>
                        <body name="robotiq_left_follower" pos="0 0.055 0.0375">
                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0"
                              diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>
                          <joint name="robotiq_left_follower_joint" axis="1 0 0" range="-0.872664 0.872664"
                              armature="0.001" pos="0 -0.018 0.0065"
                              solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
                          <geom type="mesh" mesh="robotiq_follower" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          <geom type="mesh" mesh="robotiq_follower" group="3"/>
                          <body name="robotiq_left_pad" pos="0 -0.0189 0.01352">
                            <geom name="robotiq_left_pad1" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375"
                                mass="0" friction="0.7" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1" group="3"/>
                            <geom name="robotiq_left_pad2" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375"
                                mass="0" friction="0.6" solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1" group="3"/>
                            <inertial mass="0.0035" pos="0 -0.0025 0.0185" quat="1 0 0 1"
                                diaginertia="4.73958e-07 3.64583e-07 1.23958e-07"/>
                            <geom type="mesh" mesh="robotiq_pad" contype="0" conaffinity="0" group="2"/>
                            <geom type="mesh" mesh="robotiq_silicone_pad" material="robotiq_black" contype="0" conaffinity="0" group="2"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
"""
        # Insert the gripper body just before the closing </body> of wrist_3_link
        # (after the attachment_site line)
        xml = xml.replace(
            '<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>',
            '<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>'
            + robotiq_body
        )

        # 3. Inject contact exclusions
        robotiq_contacts = """
  <contact>
    <exclude body1="robotiq_base" body2="robotiq_left_driver"/>
    <exclude body1="robotiq_base" body2="robotiq_right_driver"/>
    <exclude body1="robotiq_base" body2="robotiq_left_spring_link"/>
    <exclude body1="robotiq_base" body2="robotiq_right_spring_link"/>
    <exclude body1="robotiq_right_coupler" body2="robotiq_right_follower"/>
    <exclude body1="robotiq_left_coupler" body2="robotiq_left_follower"/>
  </contact>
"""
        # 4. Inject tendon for symmetric actuation
        robotiq_tendon = """
  <tendon>
    <fixed name="robotiq_split">
      <joint joint="robotiq_right_driver_joint" coef="0.5"/>
      <joint joint="robotiq_left_driver_joint" coef="0.5"/>
    </fixed>
  </tendon>
"""
        # 5. Inject equality constraints (4-bar linkage + parallel finger motion)
        robotiq_equality = """
  <equality>
    <connect anchor="0 0 0" body1="robotiq_right_follower" body2="robotiq_right_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <connect anchor="0 0 0" body1="robotiq_left_follower" body2="robotiq_left_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="robotiq_right_driver_joint" joint2="robotiq_left_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <!-- Couple spring_link to driver for parallel finger motion (no curling) -->
    <joint joint1="robotiq_right_spring_link_joint" joint2="robotiq_right_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <joint joint1="robotiq_left_spring_link_joint" joint2="robotiq_left_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>
"""
        # 6. Inject actuator (biastype="affine" gaintype="fixed" needed for position-servo behavior)
        robotiq_actuator = """
  <actuator>
    <general name="robotiq_fingers_actuator" tendon="robotiq_split" forcerange="-5 5" ctrlrange="0 255"
        gaintype="fixed" gainprm="0.3137255 0 0" biastype="affine" biasprm="0 -100 -10"/>
  </actuator>
"""
        # Add all before </mujoco>
        mujoco_close_idx = xml.rfind("</mujoco>")
        xml = (xml[:mujoco_close_idx]
               + robotiq_contacts + robotiq_tendon + robotiq_equality + robotiq_actuator
               + "</mujoco>")

        return xml

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

        # Cube (may or may not exist depending on task)
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

        # Reach target site
        self._reach_target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "reach_target"
        )

        # Peg insertion objects
        self._peg_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "peg"
        )
        self._peg_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_joint"
        )
        self._hole_bottom_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "hole_bottom"
        )

        # Faucet objects (turn_faucet task)
        self._faucet_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "faucet_joint"
        )
        self._faucet_handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "faucet_handle_site"
        )
        self._faucet_switch_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "faucet_switch"
        )

        # Button press objects
        self._button_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "button_joint"
        )
        self._button_plunger_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "button_plunger"
        )

        # Door open objects
        self._door_hinge_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge"
        )
        self._door_handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "door_handle_site"
        )

        # Lever pull objects
        self._lever_hinge_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lever_hinge"
        )
        self._lever_tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lever_tip_site"
        )

        # Sweep objects
        self._sweep_obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sweep_obj"
        )
        self._sweep_target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "sweep_target"
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
    # Inverse Kinematics (Damped Least Squares, 6-DOF)
    # ============================================================

    @staticmethod
    def _orientation_error(R_current: 'np.ndarray', R_desired: 'np.ndarray') -> 'np.ndarray':
        """
        Compute orientation error as a 3D vector (axis-angle representation).

        Uses the skew-symmetric part of R_err = R_desired @ R_current^T:
            error = 0.5 * [R_err[2,1] - R_err[1,2],
                           R_err[0,2] - R_err[2,0],
                           R_err[1,0] - R_err[0,1]]

        This is the standard orientation error used in operational space control
        (Siciliano, Robotics: Modelling, Planning and Control, Ch. 3.7).

        Returns:
            (3,) orientation error vector (zero when aligned)
        """
        R_err = R_desired @ R_current.T
        error = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        return error

    def _ik_solve(self, target_pos: 'np.ndarray') -> 'np.ndarray':
        """
        6-DOF Damped Least Squares IK (position + full orientation).

        Maintains both the desired EE position AND the reference
        orientation (captured at reset). This fully constrains the
        kinematic chain for 6-DOF arms (Lite6, UR5, SO-101) and uses
        DLS redundancy resolution for 7-DOF arms (Franka).

        For 5-DOF arms (WidowX), orientation weight is reduced to allow
        position priority.

        Args:
            target_pos: (3,) desired EE position in world frame

        Returns:
            qpos: (n_arm_joints,) target joint positions
        """
        # Save state (we'll modify data for IK iterations)
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        # Always seed from actual joint state — with 6-DOF IK the solution
        # is unique, so there is no ambiguity to resolve by using a stale seed.
        # This prevents IK-vs-reality divergence.
        qpos = np.array([self.data.qpos[i] for i in self._arm_qpos_ids])
        base_damping = self.cfg.ik_damping

        # Full orientation control for robots with >= 6 DOF
        # For 5-DOF (WidowX), use reduced orientation weight
        use_orientation = (self.cfg.n_arm_joints >= 5)
        if self.cfg.n_arm_joints >= 6:
            ori_weight = 1.0  # full weight for 6+ DOF
        else:
            ori_weight = 0.3  # reduced for underdetermined systems

        # Reference orientation (captured at reset)
        R_desired = self._target_ee_mat

        for it in range(self.cfg.ik_max_iter):
            # Set arm joints and forward kinematics
            for i, qpos_idx in enumerate(self._arm_qpos_ids):
                self.data.qpos[qpos_idx] = qpos[i]
            mujoco.mj_forward(self.model, self.data)

            # Current EE position
            ee_pos = self._get_ee_pos()
            pos_error = target_pos - ee_pos
            pos_err_norm = np.linalg.norm(pos_error)

            # Full orientation error (3-DOF)
            if use_orientation:
                R_current = self._get_ee_mat()
                ori_error = self._orientation_error(R_current, R_desired) * ori_weight
                ori_err_norm = np.linalg.norm(ori_error)
            else:
                ori_error = np.zeros(3)
                ori_err_norm = 0.0

            # Convergence check
            if pos_err_norm < 1e-4 and ori_err_norm < 1e-3:
                break

            # Compute Jacobians (3 x nv each)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if self._use_site_for_ee and self._ee_site_id >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._ee_body_id)

            # Extract arm-joint columns only
            Jp = jacp[:, self._arm_dof_ids]  # (3, n_arm_joints)

            if use_orientation:
                Jr = jacr[:, self._arm_dof_ids]  # (3, n_arm_joints)
                # Stack into 6-DOF task
                J = np.vstack([Jp, ori_weight * Jr])  # (6, n_arm_joints)
                error = np.concatenate([pos_error, ori_error])
            else:
                J = Jp
                error = pos_error

            # Adaptive damping: high early (escape singularities), low later
            lam = base_damping * max(0.1, 1.0 - it / self.cfg.ik_max_iter)

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} error
            n_task = J.shape[0]
            JJT = J @ J.T + lam**2 * np.eye(n_task)
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
                if lo < hi:
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
            # UR5 with Robotiq 2F-85: ctrl range [0, 255], 0=open, 255=closed
            grip_val = cmd * self.cfg.gripper_closed  # 0→0, 1→255
            self.data.ctrl[self.cfg.gripper_actuator_idx] = grip_val

        elif self.robot_name == "widowx":
            # WidowX: position actuator on left_finger slide joint
            # open=0.037 (max), closed=0.015 (min)
            grip_pos = self.cfg.gripper_open + cmd * (
                self.cfg.gripper_closed - self.cfg.gripper_open
            )
            self.data.ctrl[self.cfg.gripper_actuator_idx] = grip_pos

        elif self.robot_name == "lite6":
            # Lite6: motor actuator, ctrl=0 -> open, ctrl=-10 -> closed
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
            # Robotiq 2F-85: read right_driver_joint position [0=open, 0.8=closed]
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "robotiq_right_driver_joint"
            )
            if jid >= 0:
                raw = self.data.qpos[self.model.jnt_qposadr[jid]]
                return raw / 0.8  # normalize to [0, 1]
            return 0.0

        elif self.robot_name == "widowx":
            # WidowX: read left_finger slide joint position
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger"
            )
            if jid >= 0:
                raw = self.data.qpos[self.model.jnt_qposadr[jid]]
                span = self.cfg.gripper_closed - self.cfg.gripper_open
                if abs(span) > 1e-6:
                    return (raw - self.cfg.gripper_open) / span
            return 0.0

        elif self.robot_name == "lite6":
            # Lite6: read gripper_left_finger slide joint
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_finger"
            )
            if jid >= 0:
                raw = self.data.qpos[self.model.jnt_qposadr[jid]]
                # Range: -0.025 (open/apart) to ~0 (closed/together)
                return (raw + 0.025) / 0.025
            return 0.0

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

        # Randomize reach target (move the site)
        if self.task == "reach" and self._reach_target_site_id >= 0:
            r = self.cfg.cube_randomize_range
            self.model.site_pos[self._reach_target_site_id][0] += self._rng.uniform(-r, r)
            self.model.site_pos[self._reach_target_site_id][1] += self._rng.uniform(-r, r)

        # Randomize peg start position
        if self._peg_joint_id >= 0:
            peg_qpos_adr = self.model.jnt_qposadr[self._peg_joint_id]
            r = self.cfg.cube_randomize_range * 0.5  # smaller range for precision task
            self.data.qpos[peg_qpos_adr] += self._rng.uniform(-r, r)
            self.data.qpos[peg_qpos_adr + 1] += self._rng.uniform(-r, r)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._ee_target = self._get_ee_pos().copy()  # track desired EE position
        self._target_ee_mat = self._get_ee_mat()  # reference orientation for 6-DOF IK
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

        # Accumulate desired EE position
        self._ee_target = self._ee_target + ee_delta

        # IK → joint positions (6-DOF IK with orientation constraint
        # produces unique, continuous solutions — no drift clamp or
        # rate limiter needed)
        target_qpos = self._ik_solve(self._ee_target)

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
        """Check task-specific success condition."""
        if self.task == "reach":
            return self._check_reach_success()
        elif self.task == "push":
            return self._check_push_success()
        elif self.task == "stack":
            return self._check_stack_success()
        elif self.task == "peg_insertion":
            return self._check_peg_insertion_success()

        elif self.task == "drawer_open":
            return self._check_drawer_open_success()
        elif self.task == "turn_faucet":
            return self._check_turn_faucet_success()
        elif self.task == "button_press":
            return self._check_button_press_success()
        elif self.task == "door_open":
            return self._check_door_open_success()
        elif self.task == "lever_pull":
            return self._check_lever_pull_success()
        elif self.task == "sweep":
            return self._check_sweep_success()

        # Default: pick_place
        if self._cube_body_id < 0 or self._target_site_id < 0:
            return False

        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.data.site_xpos[self._target_site_id]

        # XY distance
        dist_xy = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        # Height check: cube must be near target height (placed, not mid-air)
        height_ok = abs(cube_pos[2] - target_pos[2]) < 0.03

        return dist_xy < self.cfg.success_threshold and height_ok

    def _check_reach_success(self) -> bool:
        """Check if EE reached the target position."""
        if self._reach_target_site_id < 0:
            return False
        ee_pos = self._get_ee_pos()
        target_pos = self.data.site_xpos[self._reach_target_site_id]
        dist = np.linalg.norm(ee_pos - target_pos)
        return dist < 0.02  # 2cm tolerance

    def _check_push_success(self) -> bool:
        """Check if cube was pushed to the target zone."""
        if self._cube_body_id < 0 or self._target_site_id < 0:
            return False
        cube_pos = self.data.xpos[self._cube_body_id]
        target_pos = self.data.site_xpos[self._target_site_id]
        dist_xy = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        # Push: cube stays on table surface (no height change required)
        return dist_xy < self.cfg.success_threshold

    def _check_peg_insertion_success(self) -> bool:
        """Check if peg is inserted into the hole.

        Success requires:
        1. Peg center XY within hole inner radius (actually inside, not just above)
        2. Peg center Z below the hole body center (inserted past the rim)
        """
        if self._peg_body_id < 0 or self._hole_bottom_site_id < 0:
            return False
        cfg = self.cfg
        peg_pos = self.data.xpos[self._peg_body_id]
        hole_body_pos = self.data.body("hole_body").xpos
        hole_inner_r = cfg.peg_radius + cfg.hole_clearance

        # XY: peg center must be within the hole opening
        dist_xy = np.linalg.norm(peg_pos[:2] - hole_body_pos[:2])
        inside_xy = dist_xy < hole_inner_r

        # Z: peg center must be at or below hole top (within 2mm tolerance)
        inserted_z = peg_pos[2] < hole_body_pos[2] + 0.002

        return inside_xy and inserted_z


    def _check_drawer_open_success(self) -> bool:
        """Check if drawer is pulled open past 80% of its range."""
        try:
            joint_id = self.model.joint("drawer_joint").id
            qpos_idx = self.model.jnt_qposadr[joint_id]
            drawer_pos = self.data.qpos[qpos_idx]
            target = self.cfg.drawer_slide_range * 0.8
            return drawer_pos >= target
        except Exception:
            return False

    def _check_turn_faucet_success(self) -> bool:
        """Check if faucet has been turned past 80% of target angle.
        
        The faucet starts at 0 and target is negative (turning handle down).
        Success when current_angle <= target * 0.8.
        """
        if self._faucet_joint_id < 0:
            return False
        current_angle = self.get_faucet_angle()
        target = self.cfg.faucet_target_angle
        # Target is negative, so success is when angle <= 80% of target
        return current_angle <= target * 0.8

    def get_faucet_handle_pos(self) -> np.ndarray:
        """Get faucet handle site world position (for scripted policy)."""
        if self._faucet_handle_site_id >= 0:
            return self.data.site_xpos[self._faucet_handle_site_id].copy()
        return np.zeros(3)

    def get_faucet_angle(self) -> float:
        """Get current faucet joint angle (radians)."""
        if self._faucet_joint_id >= 0:
            qpos_addr = self.model.jnt_qposadr[self._faucet_joint_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

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

    def get_reach_target_pos(self) -> np.ndarray:
        """Get reach target position."""
        if self._reach_target_site_id >= 0:
            return self.data.site_xpos[self._reach_target_site_id].copy()
        return np.zeros(3)

    def get_peg_pos(self) -> np.ndarray:
        """Get current peg position (for peg insertion policy)."""
        if self._peg_body_id >= 0:
            return self.data.xpos[self._peg_body_id].copy()
        return np.zeros(3)

    def get_hole_pos(self) -> np.ndarray:
        """Get hole bottom position (for peg insertion policy)."""
        if self._hole_bottom_site_id >= 0:
            return self.data.site_xpos[self._hole_bottom_site_id].copy()
        return np.zeros(3)

    # ------------------------------------------------------------------
    # Button press
    # ------------------------------------------------------------------

    def get_button_displacement(self) -> float:
        """Get current button joint displacement (meters pressed)."""
        if self._button_joint_id >= 0:
            qpos_addr = self.model.jnt_qposadr[self._button_joint_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

    def _check_button_press_success(self) -> bool:
        """Success when button is pressed >= 66% of its travel range."""
        if self._button_joint_id < 0:
            return False
        displacement = self.get_button_displacement()
        max_travel = 0.030 * self.cfg.button_scale
        return displacement >= max_travel * 0.66

    # ------------------------------------------------------------------
    # Door open
    # ------------------------------------------------------------------

    def get_door_angle(self) -> float:
        """Get current door hinge angle (radians)."""
        if self._door_hinge_id >= 0:
            qpos_addr = self.model.jnt_qposadr[self._door_hinge_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

    def get_door_handle_pos(self) -> np.ndarray:
        """Get door handle site position."""
        if self._door_handle_site_id >= 0:
            return self.data.site_xpos[self._door_handle_site_id].copy()
        return np.zeros(3)

    def _check_door_open_success(self) -> bool:
        """Success when door is opened >= 60 degrees (~1.047 rad)."""
        if self._door_hinge_id < 0:
            return False
        return self.get_door_angle() >= 1.047

    # ------------------------------------------------------------------
    # Lever pull
    # ------------------------------------------------------------------

    def get_lever_angle(self) -> float:
        """Get current lever hinge angle (radians, 0=horizontal, pi/2=vertical)."""
        if self._lever_hinge_id >= 0:
            qpos_addr = self.model.jnt_qposadr[self._lever_hinge_id]
            return float(self.data.qpos[qpos_addr])
        return 0.0

    def get_lever_tip_pos(self) -> np.ndarray:
        """Get lever tip site position."""
        if self._lever_tip_site_id >= 0:
            return self.data.site_xpos[self._lever_tip_site_id].copy()
        return np.zeros(3)

    def _check_lever_pull_success(self) -> bool:
        """Success when lever is pulled >= 75 degrees (~1.31 rad)."""
        if self._lever_hinge_id < 0:
            return False
        return self.get_lever_angle() >= 1.31

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def get_sweep_obj_pos(self) -> np.ndarray:
        """Get sweep object body position."""
        if self._sweep_obj_body_id >= 0:
            return self.data.xpos[self._sweep_obj_body_id].copy()
        return np.zeros(3)

    def _check_sweep_success(self) -> bool:
        """Success when object is within 3cm of target zone (XY)."""
        if self._sweep_obj_body_id < 0 or self._sweep_target_site_id < 0:
            return False
        obj_pos = self.data.xpos[self._sweep_obj_body_id][:2]
        target_pos = self.data.site_xpos[self._sweep_target_site_id][:2]
        dist = np.linalg.norm(obj_pos - target_pos)
        return dist <= 0.03

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            del self.renderer
