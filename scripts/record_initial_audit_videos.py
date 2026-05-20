#!/usr/bin/env python3
"""Record 3 videos per robot×task combo for initial audit.

Saves to videos/initial_audit/{robot}_{task}_1.mp4, _2.mp4, _3.mp4
Uses the agent-view camera (overhead) at 480×480.

Usage:
    python scripts/record_initial_audit_videos.py
"""
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import (
    make_env, ScriptedPickPlace, ScriptedStack, ScriptedReach,
    ScriptedPush, ScriptedPegInsertion,
)
from opab.env.base_env import RobotConfig


ROBOTS = ["franka", "ur5", "widowx", "lite6", "so101"]
TASKS = ["reach", "pick_place", "push", "stack", "peg_insertion"]
RENDER_SIZE = (480, 480)
MAX_STEPS = 300
FPS = 30
N_VIDEOS = 3


def get_policy(task: str, robot: str):
    if task == "reach":
        return ScriptedReach(robot_name=robot)
    elif task == "push":
        return ScriptedPush(robot_name=robot)
    elif task == "stack":
        cube_size = RobotConfig(robot).cube_size
        return ScriptedStack(robot_name=robot, cube_size=cube_size)
    elif task == "peg_insertion":
        return ScriptedPegInsertion(robot_name=robot)
    else:
        return ScriptedPickPlace(robot_name=robot)


def get_action(policy, task, env, obs):
    if task == "reach":
        target_pos = env.get_reach_target_pos()
        return policy.get_action(obs, target_pos)
    elif task == "push":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        return policy.get_action(obs, cube_pos, target_pos)
    elif task == "stack":
        cube_pos = env.get_cube_pos()
        cube_b_pos = env.get_cube_b_pos()
        return policy.get_action(obs, cube_pos, cube_b_pos)
    elif task == "peg_insertion":
        peg_pos = env.get_peg_pos()
        hole_pos = env.get_hole_pos()
        return policy.get_action(obs, peg_pos, hole_pos)
    else:
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        return policy.get_action(obs, cube_pos, target_pos)


def render_frame(env):
    """Render from the recording-view (side-angle camera)."""
    renderer = mujoco.Renderer(env.model, *RENDER_SIZE)
    mujoco.mj_forward(env.model, env.data)

    # Use a free camera with side/angled view (recording view)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    if env.robot_name == "so101":
        cam.lookat[:] = [0.20, 0.0, 0.05]
        cam.distance = 0.5
        cam.azimuth = 135
        cam.elevation = -25
    elif env.robot_name == "widowx":
        cam.lookat[:] = [0.22, 0.0, 0.42]
        cam.distance = 0.7
        cam.azimuth = 135
        cam.elevation = -25
    elif env.robot_name == "lite6":
        cam.lookat[:] = [0.30, 0.0, 0.42]
        cam.distance = 0.75
        cam.azimuth = 135
        cam.elevation = -25
    else:
        # Franka/UR5 — standard tabletop view
        cam.lookat[:] = [0.5, 0.0, 0.45]
        cam.distance = 0.9
        cam.azimuth = 135
        cam.elevation = -25

    renderer.update_scene(env.data, cam)
    frame = renderer.render()
    del renderer
    return frame


def record_episode(env, policy, task, seed):
    """Record one episode and return (frames, success, steps)."""
    obs = env.reset(seed=seed)
    policy.reset()

    frames = []
    terminated = truncated = False
    steps = 0
    info = {}

    # Capture first frame
    frames.append(render_frame(env))

    while not terminated and not truncated and steps < MAX_STEPS:
        action = get_action(policy, task, env, obs)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # Capture every frame for smooth video
        if steps % 2 == 0:  # every 2 steps = 15fps effective, or use 1 for 30fps-ish
            frames.append(render_frame(env))

    success = info.get("success", False) if steps > 0 else False
    return frames, success, steps


def save_video(frames, path, fps=FPS):
    """Save frames as MP4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    output_dir = Path("videos/initial_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    total_videos = len(ROBOTS) * len(TASKS) * N_VIDEOS
    done = 0

    print(f"Recording {total_videos} videos to {output_dir}/")
    print(f"{'='*60}")

    for robot in ROBOTS:
        for task in TASKS:
            env = make_env(robot=robot, task=task, image_size=RENDER_SIZE)

            for vid_idx in range(N_VIDEOS):
                seed = vid_idx * 100
                policy = get_policy(task, robot)

                frames, success, steps = record_episode(env, policy, task, seed)

                filename = f"{robot}_{task}_{vid_idx + 1}.mp4"
                filepath = output_dir / filename
                save_video(frames, filepath)

                done += 1
                status = "✓" if success else "✗"
                print(f"  [{done:3d}/{total_videos}] {status} {filename:40s} "
                      f"| {len(frames)} frames, {steps} steps, success={success}")

            env.close()

        print()

    elapsed = time.time() - total_start
    print(f"{'='*60}")
    print(f"Done! {total_videos} videos saved to {output_dir}/")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
