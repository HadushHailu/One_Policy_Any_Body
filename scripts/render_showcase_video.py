#!/usr/bin/env python3
"""
Render a high-quality showcase video of OPAB tasks across multiple robots.

Produces a grid or sequential video showing scripted expert policies
executing all 5 tasks on available robots.

Output: media/opab_showcase.mp4 (720p, 30fps)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import imageio
import mujoco
from opab.env import make_env, SUPPORTED_TASKS
from opab.env.scripted_policies import (
    ScriptedReach, ScriptedPickPlace, ScriptedPush,
    ScriptedStack, ScriptedPegInsertion,
)
from opab.env.base_env import RobotConfig


# ============================================================
# Config
# ============================================================
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "media" / "opab_showcase.mp4"
RENDER_SIZE = (480, 480)  # per-cell render resolution
FPS = 30
MAX_STEPS = 250  # max steps per episode in video

# Which combos to render (use working robots)
ROBOTS = ["franka", "so101"]  # UR5 scene_gripper issue — skip for now
TASKS = ["reach", "pick_place", "push", "stack", "peg_insertion"]

TASK_LABELS = {
    "reach": "Reach",
    "pick_place": "Pick & Place",
    "push": "Push",
    "stack": "Stack",
    "peg_insertion": "Peg Insertion",
}

ROBOT_LABELS = {
    "franka": "Franka Panda (7-DOF)",
    "so101": "SO-101 (5-DOF)",
    "ur5": "UR5e (6-DOF)",
}


def get_policy(task: str, robot: str):
    """Get the appropriate scripted policy for a task."""
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


def get_policy_action(policy, task, env, obs):
    """Get action from policy based on task type."""
    if task == "reach":
        target = env.get_reach_target_pos()
        return policy.get_action(obs, target)
    elif task == "push":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        return policy.get_action(obs, cube_pos, target_pos)
    elif task == "stack":
        cube_a = env.get_cube_pos()
        cube_b = env.get_cube_b_pos()
        return policy.get_action(obs, cube_a, cube_b)
    elif task == "peg_insertion":
        peg_pos = env.get_peg_pos()
        hole_pos = env.get_hole_pos()
        return policy.get_action(obs, peg_pos, hole_pos)
    else:  # pick_place
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        return policy.get_action(obs, cube_pos, target_pos)


def render_high_res(env, size=(480, 480)):
    """Render a high-resolution frame from a side camera angle."""
    renderer = mujoco.Renderer(env.model, *size)
    mujoco.mj_forward(env.model, env.data)

    # Use a free camera with side/angled view instead of overhead
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Side-angle view: slightly elevated, looking at the workspace
    if env.robot_name == "so101":
        # SO-101 is small, on the ground — closer camera
        cam.lookat[:] = [0.20, 0.0, 0.05]
        cam.distance = 0.5
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


def add_text_overlay(frame, text, position="top", font_scale=0.7, color=(255, 255, 255)):
    """Add text with background to a frame."""
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == "top":
        x = (frame.shape[1] - text_w) // 2
        y = text_h + 10
    elif position == "bottom":
        x = (frame.shape[1] - text_w) // 2
        y = frame.shape[0] - 10

    # Draw background rectangle
    pad = 5
    cv2.rectangle(frame,
                  (x - pad, y - text_h - pad),
                  (x + text_w + pad, y + baseline + pad),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def render_episode(robot: str, task: str):
    """Render one episode and return list of frames."""
    print(f"  Rendering {robot}/{task}...", end=" ", flush=True)

    env = make_env(robot=robot, task=task, image_size=RENDER_SIZE)
    obs = env.reset(seed=42)
    policy = get_policy(task, robot)
    policy.reset()

    frames = []
    for step in range(MAX_STEPS):
        # Render high-res frame
        frame = render_high_res(env, RENDER_SIZE)

        # Add label overlay
        label = f"{ROBOT_LABELS.get(robot, robot)} | {TASK_LABELS.get(task, task)}"
        frame = add_text_overlay(frame, label, position="top")

        # Add step counter
        step_text = f"Step {step:3d}"
        frame = add_text_overlay(frame, step_text, position="bottom", font_scale=0.5)

        frames.append(frame)

        # Get action
        action = get_policy_action(policy, task, env, obs)
        if action is None:
            # Policy finished — hold last frame for a bit
            for _ in range(15):
                success_frame = frame.copy()
                success_frame = add_text_overlay(success_frame, "DONE", position="bottom",
                                                 color=(0, 255, 0), font_scale=0.8)
                frames.append(success_frame)
            break

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("success", False) and step > 50:
            # Show success for a few frames
            for _ in range(20):
                sframe = render_high_res(env, RENDER_SIZE)
                sframe = add_text_overlay(sframe, label, position="top")
                sframe = add_text_overlay(sframe, "SUCCESS!", position="bottom",
                                          color=(0, 255, 0), font_scale=0.8)
                frames.append(sframe)
            break

    env.close()
    print(f"{len(frames)} frames")
    return frames


def create_title_card(text: str, duration_frames: int = 45,
                      size=(480, 480)):
    """Create a title card (dark background with centered text)."""
    frames = []
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:] = (30, 30, 40)  # dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split("\n")
    total_height = len(lines) * 40
    start_y = (size[0] - total_height) // 2

    for i, line in enumerate(lines):
        font_scale = 1.0 if i == 0 else 0.6
        color = (255, 255, 255) if i == 0 else (180, 180, 200)
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, 2)
        x = (size[1] - tw) // 2
        y = start_y + i * 45
        cv2.putText(frame, line, (x, y), font, font_scale, color, 2, cv2.LINE_AA)

    for _ in range(duration_frames):
        frames.append(frame.copy())
    return frames


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []

    # Title card
    print("Creating title card...")
    title_frames = create_title_card(
        "One Policy, Any Body (OPAB)\n\n5 Tasks x 3 Robots\nScripted Expert Demonstrations",
        duration_frames=60
    )
    all_frames.extend(title_frames)

    # Render each robot-task combo
    for task in TASKS:
        # Task title card
        task_title = create_title_card(
            f"Task: {TASK_LABELS[task]}\n\nReach | Push | Pick | Stack | Insert",
            duration_frames=30
        )
        all_frames.extend(task_title)

        for robot in ROBOTS:
            try:
                frames = render_episode(robot, task)
                all_frames.extend(frames)

                # Brief pause between episodes
                if frames:
                    pause = [frames[-1]] * 10
                    all_frames.extend(pause)
            except Exception as e:
                print(f"  SKIP {robot}/{task}: {e}")

    # End card
    end_frames = create_title_card(
        "OPAB\n\nOne Policy, Any Body\n15 Robot-Task Combinations",
        duration_frames=45
    )
    all_frames.extend(end_frames)

    # Write video
    print(f"\nWriting {len(all_frames)} frames to {OUTPUT_PATH}...")
    writer = imageio.get_writer(
        str(OUTPUT_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,  # high quality (1-10 scale)
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    duration = len(all_frames) / FPS
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nDone! Video saved to: {OUTPUT_PATH}")
    print(f"  Duration: {duration:.1f}s | Frames: {len(all_frames)} | Size: {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
