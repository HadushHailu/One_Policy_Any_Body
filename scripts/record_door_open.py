#!/usr/bin/env python3
"""Record lite6 door_open demo video."""
import sys
from pathlib import Path
import numpy as np
import mujoco
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opab.env.base_env import PickPlaceEnv

env = PickPlaceEnv(robot='lite6', task='door_open', seed=42)
obs = env.reset(seed=42)

renderer = mujoco.Renderer(env.model, 480, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.25, 0.0, 0.45]
cam.distance = 0.75
cam.azimuth = 145
cam.elevation = -20

Path("videos").mkdir(exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/lite6_door_open.mp4', fourcc, 30, (640, 480))

def render():
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, cam)
    out.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
hinge_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'door_hinge')
hinge_qadr = env.model.jnt_qposadr[hinge_jid]

handle_pos = env.data.site_xpos[handle_sid].copy()
print(f'Handle site: {handle_pos}, EE: {env.data.site_xpos[ee_sid]}')

# The handle bar extends in -X from the stem. Target further along the bar
# (away from stem) for a firmer grip around the bar's middle
# Handle site is already at -0.020*ds along bar, shift another -0.012 in X
# Also go slightly lower (-Z) so bar sits in the center of the finger pads
# rather than at the fingertips
grip_target = handle_pos + np.array([-0.012, 0, -0.005])
print(f'Grip target (mid-bar, lower): {grip_target}')

# Phase 1: Move above grip target
above = grip_target + np.array([0, 0, 0.04])
for i in range(25):
    d = np.clip((above - env.data.site_xpos[ee_sid]) * 0.25, -0.01, 0.01)
    env.step(np.array([d[0], d[1], d[2], 0.0, 0.0])); render()

# Phase 2: Lower to bar
for i in range(15):
    d = np.clip((grip_target - env.data.site_xpos[ee_sid]) * 0.25, -0.01, 0.01)
    env.step(np.array([d[0], d[1], d[2], 0.0, 0.0])); render()

dist = np.linalg.norm(env.data.site_xpos[ee_sid] - grip_target)
print(f'At bar - dist: {dist*1000:.1f}mm')

# Phase 3: Close gripper
for i in range(10):
    env.step(np.array([0., 0., 0., 0., 1.])); render()

# Phase 4: Pull door open - track handle arc
# Instead of a fixed linear pull, follow the handle's circular arc at each step.
# Compute tangential direction from current handle pos relative to hinge.
hinge_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'door_panel')
prev_angle = 0.0
stall_count = 0
for i in range(120):
    angle = env.data.qpos[hinge_qadr]
    # Stop pulling once door is near fully open (85°)
    if angle >= 1.48:  # ~85 degrees - let it settle at 90
        break
    if i > 15 and abs(angle - prev_angle) < 0.001:
        stall_count += 1
        if stall_count > 10:
            break
    else:
        stall_count = 0
    prev_angle = angle

    # Get current handle position and compute tangent to arc (perpendicular to radius)
    cur_handle = env.data.site_xpos[handle_sid].copy()
    hinge_pos = env.data.xpos[hinge_bid].copy()  # hinge world pos
    radius_vec = cur_handle[:2] - hinge_pos[:2]  # XY vector from hinge to handle
    # Tangent for CCW rotation (opening): rotate radius 90° CCW -> (-ry, rx)
    tangent = np.array([-radius_vec[1], radius_vec[0]])
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    # Position correction: keep EE locked onto the handle (dominant term)
    ee_pos = env.data.site_xpos[ee_sid]
    error = cur_handle - ee_pos
    error[2] = 0  # don't chase Z (keep grip height stable)

    # Use mostly position tracking with a small tangential nudge
    # This keeps the gripper firmly on the bar while gently pushing the door open
    pull_strength = 0.004  # gentle tangential push
    track_gain = 0.5       # strong position tracking to stay on bar
    dx = tangent[0] * pull_strength + np.clip(error[0] * track_gain, -0.008, 0.008)
    dy = tangent[1] * pull_strength + np.clip(error[1] * track_gain, -0.008, 0.008)
    # Yaw tracks the door angle to keep fingers aligned with bar
    yaw_delta = 0.02
    env.step(np.array([dx, dy, 0.0, yaw_delta, 1.])); render()

angle_deg = np.degrees(env.data.qpos[hinge_qadr])
print(f'Door angle: {angle_deg:.1f} deg')

# Phase 5: Retract
for i in range(10):
    env.step(np.array([0., 0., 0.005, 0., 0.])); render()

out.release()
renderer.close()
print('Done: videos/lite6_door_open.mp4')
