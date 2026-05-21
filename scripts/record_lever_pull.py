#!/usr/bin/env python3
"""Record lite6 lever_pull demo video."""
import sys
from pathlib import Path
import numpy as np
import mujoco
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opab.env.base_env import PickPlaceEnv

env = PickPlaceEnv(robot='lite6', task='lever_pull', seed=42)
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
out = cv2.VideoWriter('videos/lite6_lever_pull.mp4', fourcc, 30, (640, 480))

def render():
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, cam)
    out.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
tip_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'lever_tip_site')
hinge_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'lever_hinge')
hinge_qadr = env.model.jnt_qposadr[hinge_jid]

# Target the arm shaft (thinner capsule, much easier to grip than sphere tip)
arm_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'lever_arm')
tip_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'lever_tip_site')
arm_body_pos = env.data.xpos[arm_bid].copy()
shaft_target = arm_body_pos + np.array([0, 0.06, 0])  # 6cm along shaft
print(f'Shaft target: {shaft_target}')

# Phase 1: Approach above shaft + rotate yaw 90°
above = shaft_target + np.array([0, 0, 0.05])
for i in range(30):
    d = np.clip((above - env.data.site_xpos[ee_sid]) * 0.25, -0.01, 0.01)
    yaw = 0.1 if i < 16 else 0.0
    env.step(np.array([d[0], d[1], d[2], yaw, 0.0])); render()

# Phase 2: Lower to shaft
for i in range(15):
    d = np.clip((shaft_target - env.data.site_xpos[ee_sid]) * 0.25, -0.01, 0.01)
    env.step(np.array([d[0], d[1], d[2], 0.0, 0.0])); render()

print(f'Dist to shaft: {np.linalg.norm(env.data.site_xpos[ee_sid]-shaft_target)*1000:.1f}mm')

# Phase 3: Grasp
for i in range(10):
    env.step(np.array([0., 0., 0., 0., 1.])); render()

# Phase 4: Pull along tangent to the arc (stop if lever stalls)
prev_angle = 0.0
stall_count = 0
for i in range(70):
    angle = env.data.qpos[hinge_qadr]
    # Stop if lever hasn't moved in 5 steps
    if i > 5 and abs(angle - prev_angle) < 0.001:
        stall_count += 1
        if stall_count > 5:
            break
    else:
        stall_count = 0
    prev_angle = angle
    dy = -np.sin(angle + 0.1) * 0.006
    dz = -np.cos(angle + 0.1) * 0.006
    env.step(np.array([0., dy, dz, 0., 1.])); render()

print(f'Lever angle: {np.degrees(env.data.qpos[hinge_qadr]):.1f} deg')

# Phase 5: Release and retract
for i in range(10):
    env.step(np.array([0., 0., 0.005, 0., 0.])); render()

out.release()
renderer.close()
print('Done: videos/lite6_lever_pull.mp4')
