#!/usr/bin/env python3
"""Record agent-view videos for stack and peg_insertion tasks (all 5 robots).

Saves to videos/agent-view/{robot}_{task}_1.mp4
Uses the overhead agent-view camera at 480×480.
"""
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import (
    make_env, ScriptedStack, ScriptedPegInsertion,
)
from opab.env.base_env import RobotConfig


ROBOTS = ["franka", "ur5", "widowx", "lite6", "so101"]
TASKS = ["stack", "peg_insertion"]
RENDER_SIZE = (480, 480)
MAX_STEPS = 300
FPS = 30


def get_policy(task, robot):
    if task == "stack":
        cube_size = RobotConfig(robot).cube_size
        return ScriptedStack(robot_name=robot, cube_size=cube_size)
    else:
        return ScriptedPegInsertion(robot_name=robot)


def get_action(policy, task, env, obs):
    if task == "stack":
        cube_pos = env.get_cube_pos()
        cube_b_pos = env.get_cube_b_pos()
        return policy.get_action(obs, cube_pos, cube_b_pos)
    else:
        peg_pos = env.get_peg_pos()
        hole_pos = env.get_hole_pos()
        return policy.get_action(obs, peg_pos, hole_pos)


def render_agent_view(env):
    """Render from the overhead agent-view camera."""
    renderer = mujoco.Renderer(env.model, *RENDER_SIZE)
    mujoco.mj_forward(env.model, env.data)
    if env._cam_id >= 0:
        renderer.update_scene(env.data, camera=env._cam_id)
    else:
        renderer.update_scene(env.data)
    frame = renderer.render()
    del renderer
    return frame


def save_video(frames, path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    output_dir = Path("videos/agent-view")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(ROBOTS) * len(TASKS)
    done = 0

    print(f"Recording {total} agent-view videos to {output_dir}/")
    print(f"{'='*60}")

    for robot in ROBOTS:
        for task in TASKS:
            env = make_env(robot=robot, task=task, image_size=RENDER_SIZE)
            policy = get_policy(task, robot)

            obs = env.reset(seed=42)
            policy.reset()

            frames = []
            terminated = truncated = False
            steps = 0
            info = {}

            frames.append(render_agent_view(env))

            while not terminated and not truncated and steps < MAX_STEPS:
                action = get_action(policy, task, env, obs)
                if action is None:
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if steps % 2 == 0:
                    frames.append(render_agent_view(env))

            success = info.get("success", False) if steps > 0 else False
            filename = f"{robot}_{task}_1.mp4"
            save_video(frames, output_dir / filename)

            done += 1
            status = "✓" if success else "✗"
            print(f"  [{done:2d}/{total}] {status} {filename:35s} | {len(frames)} frames, {steps} steps")

            env.close()

    print(f"\n{'='*60}")
    print(f"Done! {total} videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
