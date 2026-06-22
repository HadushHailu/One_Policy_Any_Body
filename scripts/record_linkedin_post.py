#!/usr/bin/env python3
"""Generate a LinkedIn showcase video for One Policy, Any Body.

Creates a montage showing:
  1. Lite6 — all 7 tasks, 3-camera grid (sideview, agentview, topdown)
  2. UR5  — door_open, drawer_open, turn_faucet, 3-camera grid

Output: posts/opab_linkedin_showcase.mp4

Usage:
    python scripts/record_linkedin_post.py
    python scripts/record_linkedin_post.py --no-title    # skip title cards
    python scripts/record_linkedin_post.py --task pick_place  # single task only
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opab.env.base_env import PickPlaceEnv

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "posts"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Video layout settings
# ---------------------------------------------------------------
GRID_W, GRID_H = 1920, 480       # 3 panels side-by-side (each 640x480)
PANEL_W, PANEL_H = 640, 480      # Individual camera panel
RENDER_W, RENDER_H = 640, 480    # MuJoCo render resolution (must fit model offscreen buffer)
FPS = 30
TITLE_DURATION_S = 2.0           # Title card duration in seconds

# Camera names to use (3-camera grid)
CAMERA_NAMES = ["sideview_0", "agentview_0", "topdown_0"]

# Camera fallbacks per robot
CAMERA_FALLBACKS = {
    "lite6": {
        "sideview_0": {"lookat": [0.15, 0.0, 0.45], "distance": 0.70, "azimuth": 150, "elevation": -22},
    },
    "ur5": {
        "sideview_0": {"lookat": [0.30, 0.0, 0.50], "distance": 0.85, "azimuth": 150, "elevation": -22},
    },
}

# Task display names
TASK_LABELS = {
    "pick_place": "Pick & Place",
    "push": "Push",
    "stack": "Stack",
    "peg_insertion": "Peg Insertion",
    "drawer_open": "Drawer Open",
    "turn_faucet": "Turn Faucet",
    "door_open": "Door Open",
}


def make_env(robot, task):
    env = PickPlaceEnv(robot=robot, task=task, seed=42)
    env.reset(seed=42)
    return env


def setup_camera(env, cam_name, robot):
    """Configure a MjvCamera for the given camera name."""
    cam = mujoco.MjvCamera()
    robot_cameras = getattr(env.cfg, 'cameras', {})

    if cam_name == "sideview_0":
        fallback = CAMERA_FALLBACKS.get(robot, {}).get("sideview_0", {})
        cam_cfg = robot_cameras.get("sideview_0", fallback)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cam_cfg["lookat"]
        cam.distance = cam_cfg["distance"]
        cam.azimuth = cam_cfg["azimuth"]
        cam.elevation = cam_cfg["elevation"]
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam.fixedcamid < 0:
            # Fallback to sideview with slightly different angle
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            fallback = CAMERA_FALLBACKS.get(robot, {}).get("sideview_0", {})
            cam_cfg = robot_cameras.get("sideview_0", fallback)
            cam.lookat[:] = cam_cfg["lookat"]
            cam.distance = cam_cfg["distance"]
            if cam_name == "agentview_0":
                cam.azimuth = cam_cfg["azimuth"] + 40
                cam.elevation = cam_cfg["elevation"] - 10
            elif cam_name == "topdown_0":
                cam.azimuth = 180
                cam.elevation = -89
                cam.distance = cam_cfg["distance"] * 0.8
            else:
                cam.azimuth = cam_cfg["azimuth"]
                cam.elevation = cam_cfg["elevation"]
    return cam


def create_title_card(text, subtitle="", width=GRID_W, height=GRID_H):
    """Create a title card frame with text overlay."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark gradient background
    for y in range(height):
        val = int(20 + 15 * (y / height))
        frame[y, :] = [val, val, val + 5]

    # Main title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2
    y = (height - text_size[1]) // 2 - 30
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Subtitle
    if subtitle:
        sub_scale = 0.9
        sub_thickness = 2
        sub_size = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)[0]
        sx = (width - sub_size[0]) // 2
        sy = y + text_size[1] + 50
        cv2.putText(frame, subtitle, (sx, sy), font, sub_scale, (180, 200, 255), sub_thickness, cv2.LINE_AA)

    return frame


def render_robot_snapshot(robot, task="pick_place"):
    """Render a frame of a robot mid-task so the arm is extended and visible."""
    try:
        env = PickPlaceEnv(robot=robot, task=task, seed=42)
        env.reset(seed=42)

        # Run steps so the arm is extended and visible
        # WidowX: just use home pose (arm is already visible at reset)
        if robot != "widowx":
            ee_name = get_ee_site_name(env, robot)
            ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
            target = env.get_cube_pos() if hasattr(env, 'get_cube_pos') else None
            if target is not None:
                for _ in range(150):
                    ee = env.data.site_xpos[ee_sid].copy()
                    d = np.clip((target + [0, 0, 0.03] - ee) * 0.3, -0.01, 0.01)
                    env.step(np.array([d[0], d[1], d[2], 0.0, 0.0]))
            else:
                for _ in range(150):
                    env.step(np.array([0., 0., 0., 0., 0.]))

        renderer = mujoco.Renderer(env.model, RENDER_H, RENDER_W)
        cam = setup_camera(env, "sideview_0", robot)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, cam)
        frame = renderer.render().copy()
        renderer.close()
        env.close()
        return frame
    except Exception as e:
        print(f"  Warning: Could not render {robot}: {e}")
        # Return a placeholder frame with robot name
        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(robot, font, 1.0, 2)[0]
        x = (RENDER_W - text_size[0]) // 2
        y = (RENDER_H + text_size[1]) // 2
        cv2.putText(frame, robot, (x, y), font, 1.0, (150, 150, 150), 2, cv2.LINE_AA)
        return frame


def render_task_snapshot(task, robot="lite6"):
    """Render a frame of a task mid-execution so the arm and objects are visible."""
    try:
        env = PickPlaceEnv(robot=robot, task=task, seed=42)
        env.reset(seed=42)

        # Run ~150 steps so the arm is reaching toward objects
        ee_name = get_ee_site_name(env, robot)
        ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        # Move toward the task object
        if task in ("pick_place", "push", "stack"):
            target = env.get_cube_pos() + np.array([0, 0, 0.03])
        elif task == "peg_insertion":
            target = env.get_peg_pos() + np.array([0, 0, 0.03])
        elif task == "drawer_open":
            sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'drawer_handle_site')
            target = env.data.site_xpos[sid].copy() + np.array([0, 0, 0.02])
        elif task == "turn_faucet":
            sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'faucet_handle_site')
            target = env.data.site_xpos[sid].copy() + np.array([0, 0, 0.02])
        elif task == "door_open":
            sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
            target = env.data.site_xpos[sid].copy() + np.array([0, 0, 0.02])
        else:
            target = None

        if target is not None:
            for _ in range(150):
                ee = env.data.site_xpos[ee_sid].copy()
                d = np.clip((target - ee) * 0.3, -0.01, 0.01)
                env.step(np.array([d[0], d[1], d[2], 0.0, 0.0]))
        else:
            for _ in range(150):
                env.step(np.array([0., 0., 0., 0., 0.]))

        renderer = mujoco.Renderer(env.model, RENDER_H, RENDER_W)
        cam = setup_camera(env, "sideview_0", robot)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, cam)
        frame = renderer.render().copy()
        renderer.close()
        env.close()
        return frame
    except Exception as e:
        print(f"  Warning: Could not render task {task}: {e}")
        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = TASK_LABELS.get(task, task)
        text_size = cv2.getTextSize(label, font, 1.0, 2)[0]
        x = (RENDER_W - text_size[0]) // 2
        y = (RENDER_H + text_size[1]) // 2
        cv2.putText(frame, label, (x, y), font, 1.0, (150, 150, 150), 2, cv2.LINE_AA)
        return frame


def create_overview_slide(duration_s=4.0):
    """Create an overview slide: left = all robots, right = all tasks.
    
    Returns list of identical frames for the given duration.
    """
    all_robots = ["franka", "ur5", "widowx", "lite6", "so101"]
    all_tasks = ["pick_place", "push", "stack", "peg_insertion", "drawer_open", "turn_faucet", "door_open"]
    
    robot_labels = {
        "franka": "Franka Panda",
        "ur5": "UR5",
        "widowx": "WidowX",
        "lite6": "Lite6",
        "so101": "SO-101",
    }

    # Render robot snapshots
    print("  Rendering robot snapshots...")
    robot_frames = []
    for robot in all_robots:
        frame = render_robot_snapshot(robot)
        # Add label
        add_label_overlay(frame, robot_labels.get(robot, robot), position="bottom")
        robot_frames.append(frame)

    # Render task snapshots
    print("  Rendering task snapshots...")
    task_frames = []
    for task in all_tasks:
        frame = render_task_snapshot(task)
        add_label_overlay(frame, TASK_LABELS.get(task, task), position="bottom")
        task_frames.append(frame)

    # Layout: left half = robots grid, right half = tasks grid
    # Left: arrange robots in a grid (e.g., 2 cols x 3 rows, fitting 5)
    # Right: arrange tasks in a grid (e.g., 2 cols x 4 rows, fitting 7)
    half_w = GRID_W // 2
    
    # --- Left side: Robots ---
    # Grid: 3 cols x 2 rows for 5 robots
    r_cols, r_rows = 3, 2
    thumb_w = half_w // r_cols
    thumb_h = GRID_H // r_rows
    left_panel = np.zeros((GRID_H, half_w, 3), dtype=np.uint8)
    
    # Header
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left_panel, "ROBOTS", (half_w // 2 - 60, 30), font, 0.9, (180, 200, 255), 2, cv2.LINE_AA)
    
    # Offset grid down to leave space for header
    grid_y_offset = 40
    usable_h = GRID_H - grid_y_offset
    thumb_h = usable_h // r_rows
    
    for idx, rf in enumerate(robot_frames):
        col = idx % r_cols
        row = idx // r_cols
        thumb = cv2.resize(rf, (thumb_w, thumb_h))
        y_start = grid_y_offset + row * thumb_h
        x_start = col * thumb_w
        left_panel[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = thumb

    # --- Right side: Tasks ---
    # Grid: 4 cols x 2 rows for 7 tasks
    t_cols, t_rows = 4, 2
    thumb_w_t = half_w // t_cols
    right_panel = np.zeros((GRID_H, half_w, 3), dtype=np.uint8)
    
    cv2.putText(right_panel, "TASKS", (half_w // 2 - 50, 30), font, 0.9, (180, 200, 255), 2, cv2.LINE_AA)
    
    thumb_h_t = usable_h // t_rows
    
    for idx, tf in enumerate(task_frames):
        col = idx % t_cols
        row = idx // t_cols
        thumb = cv2.resize(tf, (thumb_w_t, thumb_h_t))
        y_start = grid_y_offset + row * thumb_h_t
        x_start = col * thumb_w_t
        right_panel[y_start:y_start + thumb_h_t, x_start:x_start + thumb_w_t] = thumb

    # Combine left + right
    slide = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    slide[:, :half_w] = left_panel
    slide[:, half_w:] = right_panel
    
    # Add vertical divider line
    cv2.line(slide, (half_w, 0), (half_w, GRID_H), (100, 100, 120), 2)

    return slide


def add_label_overlay(frame, label, position="bottom"):
    """Add a semi-transparent label to a frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    if position == "bottom":
        # Bottom-center label with background
        pad = 8
        bx = (w - text_size[0]) // 2 - pad
        by = h - 35
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by - text_size[1] - pad), (bx + text_size[0] + 2*pad, by + pad), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, label, (bx + pad, by), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    elif position == "top":
        # Top-left label (black font for camera view names)
        cv2.putText(frame, label, (10, 30), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return frame


def get_ee_site_name(env, robot):
    """Get the end-effector site name."""
    if hasattr(env.cfg, 'ee_site_name'):
        return env.cfg.ee_site_name
    return 'end_effector'


def run_scripted_policy(env, task, robot):
    """Run the scripted policy for a task and return. Adapted from record_all_tasks scripts."""
    ee_name = get_ee_site_name(env, robot)
    ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)

    def ee_pos():
        return env.data.site_xpos[ee_sid].copy()

    def move_to(target, steps=25, grip=0.0):
        for _ in range(steps):
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], 0.0, grip]))

    def grip_close(steps=8):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 1.]))

    def grip_open(steps=8):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 0.]))

    def idle(steps=5):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 0.]))

    if task == "pick_place":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        if robot == "ur5":
            move_to(cube_pos + [0, 0, 0.08], steps=30)
            move_to(cube_pos + [0, 0, 0.03], steps=25)
            grip_close(15)
            move_to(cube_pos + [0, 0, 0.10], steps=20, grip=1.0)
            move_to(target_pos + [0, 0, 0.08], steps=25, grip=1.0)
            move_to(target_pos + [0, 0, 0.04], steps=20, grip=1.0)
            grip_open(8)
            move_to(target_pos + [0, 0, 0.10], steps=15)
        else:
            move_to(cube_pos + [0, 0, 0.05], steps=25)
            move_to(cube_pos + [0, 0, -0.005], steps=25)
            grip_close(10)
            move_to(cube_pos + [0, 0, 0.08], steps=15, grip=1.0)
            move_to(target_pos + [0, 0, 0.06], steps=20, grip=1.0)
            move_to(target_pos + [0, 0, 0.013], steps=20, grip=1.0)
            grip_open(8)
            move_to(target_pos + [0, 0, 0.06], steps=10)

    elif task == "push":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        cs = env.cfg.cube_size
        push_dir = (target_pos[:2] - cube_pos[:2])
        push_dist = np.linalg.norm(push_dir)
        push_dir = push_dir / (push_dist + 1e-8)
        behind_offset = 0.04 if robot == "lite6" else 0.05
        behind_xy = cube_pos[:2] - push_dir * behind_offset
        contact_z = cube_pos[2] + 0.008 if robot == "lite6" else cube_pos[2] + 0.005
        grip_close(5)
        move_to(np.array([behind_xy[0], behind_xy[1], contact_z + 0.05]), steps=25, grip=1.0)
        move_to(np.array([behind_xy[0], behind_xy[1], contact_z]), steps=25, grip=1.0)
        total_push = behind_offset + push_dist - cs
        step_size = 0.005 if robot == "lite6" else 0.003
        n_push_steps = int(total_push / step_size)
        for i in range(n_push_steps):
            env.step(np.array([push_dir[0]*step_size, push_dir[1]*step_size, 0.0, 0.0, 1.0]))
        move_to(ee_pos() + [0, 0, 0.05], steps=10, grip=1.0)

    elif task == "stack":
        cube_a_pos = env.get_cube_pos()
        cube_b_pos = env.get_cube_b_pos()
        cs = env.cfg.cube_size
        if robot == "ur5":
            move_to(cube_a_pos + [0, 0, 0.08], steps=30)
            move_to(cube_a_pos + [0, 0, 0.03], steps=25)
            grip_close(15)
            move_to(cube_a_pos + [0, 0, 0.10], steps=20, grip=1.0)
            move_to(cube_b_pos + [0, 0, 2*cs + 0.08], steps=25, grip=1.0)
            move_to(cube_b_pos + [0, 0, 2*cs + 0.015], steps=30, grip=1.0)
            grip_open(8)
            move_to(ee_pos() + [0, 0, 0.06], steps=10)
        else:
            move_to(cube_a_pos + [0, 0, 0.05], steps=25)
            move_to(cube_a_pos + [0, 0, -0.005], steps=25)
            grip_close(10)
            move_to(cube_a_pos + [0, 0, 0.08], steps=15, grip=1.0)
            move_to(cube_b_pos + [0, 0, 2*cs + 0.06], steps=20, grip=1.0)
            move_to(cube_b_pos + [0, 0, 2*cs + 0.003], steps=20, grip=1.0)
            grip_open(8)
            move_to(ee_pos() + [0, 0, 0.05], steps=10)

    elif task == "peg_insertion":
        peg_pos = env.get_peg_pos()
        hole_pos = env.get_hole_pos()
        hole_depth = env.cfg.hole_depth
        move_to(peg_pos + [0, 0, 0.05], steps=20)
        move_to(peg_pos + [0, 0, 0.003], steps=15)
        grip_close(10)
        move_to(peg_pos + [0, 0, 0.10], steps=25, grip=1.0)
        high_above_hole = np.array([hole_pos[0], hole_pos[1], ee_pos()[2]])
        move_to(high_above_hole, steps=50, grip=1.0)
        move_to(hole_pos + np.array([0, 0, 2*hole_depth + 0.005]), steps=35, grip=1.0)
        grip_open(5)
        idle(30)
        move_to(ee_pos() + [0, 0, 0.05], steps=10)

    elif task == "drawer_open":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'drawer_handle_site')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        drawer_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'drawer_joint')
        move_to(handle_pos + [0, 0, 0.04], steps=25)
        move_to(handle_pos + [0, 0, -0.005], steps=30)
        grip_close(10)
        for i in range(80):
            cur_handle = env.data.site_xpos[handle_sid].copy()
            error = cur_handle - ee_pos()
            dx = np.clip(error[0] * 0.5, -0.003, 0.003)
            dz = np.clip(error[2] * 0.5, -0.003, 0.003)
            env.step(np.array([dx, -0.003, dz, 0.0, 1.0]))
            dpos = env.data.qpos[env.model.jnt_qposadr[drawer_jid]]
            if dpos >= 0.045:
                break
        grip_open(6)
        move_to(ee_pos() + [0, -0.03, 0.03], steps=10)

    elif task == "turn_faucet":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'faucet_handle_site')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        faucet_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'faucet_joint')
        faucet_qadr = env.model.jnt_qposadr[faucet_jid]
        switch_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'faucet_switch')
        switch_pos = env.data.xpos[switch_bid].copy()
        lever_dir = handle_pos[:2] - switch_pos[:2]
        lever_angle = np.arctan2(lever_dir[1], lever_dir[0])
        target_yaw = lever_angle
        for s in range(20):
            target = handle_pos + np.array([0, 0, 0.03])
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], target_yaw * 0.1, 0.0]))
        for s in range(15):
            target = handle_pos + np.array([0, 0, -0.003])
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], 0.0, 0.0]))
        grip_close(10)
        for i in range(60):
            cur_handle = env.data.site_xpos[handle_sid].copy()
            switch_pos = env.data.xpos[switch_bid].copy()
            radius_vec = cur_handle[:2] - switch_pos[:2]
            tangent = np.array([radius_vec[1], -radius_vec[0]])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            error = cur_handle - ee_pos()
            error[2] = 0
            dx = tangent[0] * 0.004 + np.clip(error[0] * 0.5, -0.008, 0.008)
            dy = tangent[1] * 0.004 + np.clip(error[1] * 0.5, -0.008, 0.008)
            env.step(np.array([dx, dy, 0.0, 0.0, 1.0]))
            angle = env.data.qpos[faucet_qadr]
            if angle <= -1.0:
                break
        grip_open(8)
        move_to(ee_pos() + [0, 0, 0.04], steps=10)

    elif task == "door_open":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
        hinge_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'door_hinge')
        hinge_qadr = env.model.jnt_qposadr[hinge_jid]
        hinge_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'door_panel')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        grip_target = handle_pos + np.array([-0.012, 0, -0.005])
        move_to(grip_target + [0, 0, 0.04], steps=25)
        move_to(grip_target, steps=15)
        grip_close(10)
        for i in range(120):
            angle = env.data.qpos[hinge_qadr]
            if angle >= 1.48:
                break
            cur_handle = env.data.site_xpos[handle_sid].copy()
            hinge_pos = env.data.xpos[hinge_bid].copy()
            radius_vec = cur_handle[:2] - hinge_pos[:2]
            tangent = np.array([-radius_vec[1], radius_vec[0]])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            error = cur_handle - ee_pos()
            error[2] = 0
            dx = tangent[0] * 0.004 + np.clip(error[0] * 0.5, -0.008, 0.008)
            dy = tangent[1] * 0.004 + np.clip(error[1] * 0.5, -0.008, 0.008)
            env.step(np.array([dx, dy, 0.0, 0.02, 1.0]))
        grip_open(5)
        move_to(ee_pos() + [0, 0, 0.04], steps=10)


def record_task_3cam(env, task, robot):
    """Record a task from 3 camera views simultaneously, return list of frame lists."""
    renderers = []
    cameras = []

    for cam_name in CAMERA_NAMES:
        renderer = mujoco.Renderer(env.model, RENDER_H, RENDER_W)
        cam = setup_camera(env, cam_name, robot)
        renderers.append(renderer)
        cameras.append(cam)

    # Collect frames from all cameras
    all_frames = [[] for _ in CAMERA_NAMES]

    ee_name = get_ee_site_name(env, robot)
    ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)

    # Monkey-patch env.step to capture frames after each step
    original_step = env.step

    def step_and_capture(action):
        obs, reward, done, truncated, info = original_step(action)
        mujoco.mj_forward(env.model, env.data)
        for i, (renderer, cam) in enumerate(zip(renderers, cameras)):
            renderer.update_scene(env.data, cam)
            frame = renderer.render().copy()
            all_frames[i].append(frame)
        return obs, reward, done, truncated, info

    env.step = step_and_capture

    # Capture initial frame
    mujoco.mj_forward(env.model, env.data)
    for i, (renderer, cam) in enumerate(zip(renderers, cameras)):
        renderer.update_scene(env.data, cam)
        frame = renderer.render().copy()
        all_frames[i].append(frame)

    # Run policy
    run_scripted_policy(env, task, robot)

    # Restore original step
    env.step = original_step

    # Clean up renderers
    for r in renderers:
        r.close()

    return all_frames


def compose_grid_frame(frames_per_cam, frame_idx, task_label, robot_label):
    """Compose a single 3-panel grid frame with labels."""
    grid = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

    cam_labels = ["Side View", "Agent View", "Top Down"]

    for i, (cam_frames, cam_label) in enumerate(zip(frames_per_cam, cam_labels)):
        # Get frame (repeat last if we're past the end)
        idx = min(frame_idx, len(cam_frames) - 1)
        panel = cam_frames[idx].copy()

        # Add camera label at top
        add_label_overlay(panel, cam_label, position="top")

        # Place in grid
        x_offset = i * PANEL_W
        grid[:PANEL_H, x_offset:x_offset + PANEL_W] = panel

    # Add task + robot label at the bottom center of the full grid
    label = f"{robot_label}  |  {task_label}"
    add_label_overlay(grid, label, position="bottom")

    return grid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LinkedIn showcase video")
    parser.add_argument("--no-title", action="store_true", help="Skip title cards")
    parser.add_argument("--task", default=None, help="Record only one task")
    parser.add_argument("--output", default=None, help="Output filename")
    args = parser.parse_args()

    output_path = args.output or str(OUTPUT_DIR / "opab_linkedin_showcase.mp4")

    # Define what to record
    lite6_tasks = ["pick_place", "push", "stack", "peg_insertion", "drawer_open", "turn_faucet", "door_open"]
    ur5_tasks = ["drawer_open"]

    if args.task:
        lite6_tasks = [args.task] if args.task in lite6_tasks else []
        ur5_tasks = [args.task] if args.task in ur5_tasks else []

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (GRID_W, GRID_H))

    def write_title(text, subtitle="", duration=TITLE_DURATION_S):
        if args.no_title:
            return
        frame = create_title_card(text, subtitle)
        for _ in range(int(FPS * duration)):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # --- Opening title ---
    write_title("One Policy, Any Embodiment", "Zero-Shot Generalist Manipulation Across Embodiments", duration=3.0)

    # --- Overview slide: robots + tasks ---
    if not args.no_title:
        print("[Overview] Rendering robots & tasks overview...")
        overview_frame = create_overview_slide()
        for _ in range(int(FPS * 4.0)):
            out.write(cv2.cvtColor(overview_frame, cv2.COLOR_RGB2BGR))

    # --- Lite6 section ---
    if lite6_tasks:
        write_title("UFACTORY Lite6", f"{len(lite6_tasks)} Tasks | 3 Camera Views")

        for task in lite6_tasks:
            print(f"[Lite6] Recording: {task}")
            env = make_env("lite6", task)
            frames_per_cam = record_task_3cam(env, task, "lite6")
            env.close()

            # Find max frame count
            max_frames = max(len(f) for f in frames_per_cam)
            task_label = TASK_LABELS.get(task, task)

            for fi in range(max_frames):
                grid = compose_grid_frame(frames_per_cam, fi, task_label, "Lite6")
                out.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            # Brief pause between tasks (0.5s black)
            pause_frame = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
            for _ in range(int(FPS * 0.3)):
                out.write(pause_frame)

    # --- UR5 section ---
    if ur5_tasks:
        write_title("UR5 + Robotiq 2F-85", f"{len(ur5_tasks)} Tasks | 3 Camera Views")

        for task in ur5_tasks:
            print(f"[UR5] Recording: {task}")
            env = make_env("ur5", task)
            frames_per_cam = record_task_3cam(env, task, "ur5")
            env.close()

            max_frames = max(len(f) for f in frames_per_cam)
            task_label = TASK_LABELS.get(task, task)

            for fi in range(max_frames):
                grid = compose_grid_frame(frames_per_cam, fi, task_label, "UR5")
                out.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            # Brief pause between tasks
            pause_frame = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
            for _ in range(int(FPS * 0.3)):
                out.write(pause_frame)

    # --- End card ---
    write_title("One Policy, Any Embodiment", "More embodiments coming soon...", duration=2.5)

    out.release()
    print(f"\nVideo saved: {output_path}")
    print(f"  Resolution: {GRID_W}x{GRID_H} @ {FPS}fps")


if __name__ == "__main__":
    main()
